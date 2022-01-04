import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import regularizers
from tensorflow.keras import initializers
from tensorflow.keras import backend as K
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.engine import network

from evaluate import mse
from layers import DenseForSparse
from layers import TransposedSharedDense
from layers import AddGaussianLoss, ReparameterizeGaussian
from layers import AddBernoulliLoss, ReparameterizeBernoulli


class MLP(network.Network):
    '''
        Multilayer Perceptron (MLP). 
    '''
    def __init__(self, 
                 hidden_sizes,
                 activations,
                 input_size=None,
                 l2_normalize=True,
                 dropout_rate=0.5,
                 **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.m_name = kwargs.get("name", None)

        self.dense_list = []
        self.dropout_rate = dropout_rate
        self.l2_normalize = l2_normalize

        for i, (size, activation) in enumerate(zip(hidden_sizes, activations)):
            self.dense_list.append(
                layers.Dense(size, activation=activation,
                    kernel_initializer='glorot_uniform',
                    bias_initializer='zeros',
                    name="mlp_dense_{}".format(i)
            ))

    def build(self, input_shapes):
        x_in = layers.Input(input_shapes[1:])
        h_out = x_in
        if self.dropout_rate:
            if self.l2_normalize:
                h_out = layers.Lambda(tf.nn.l2_normalize, arguments={"axis":1})(h_out)
            h_out = layers.Dropout(self.dropout_rate)(h_out)
        if len(self.dense_list) > 0:
            h_out = self.dense_list[0](h_out)
            for dense in self.dense_list[1:]:
                h_out = dense(h_out)
        self._init_graph_network(x_in, h_out, self.m_name)
        super(MLP, self).build(input_shapes)


class SamplerGaussian(network.Network):
    '''
        Sample from the variational Gaussian, and add its KL 
        with the prior to loss
    '''
    def __init__(self, **kwargs):
        super(SamplerGaussian, self).__init__(**kwargs)
        self.m_name = kwargs.get("name", None)
        self.rep_gauss = ReparameterizeGaussian()

    def build(self, input_shapes):
        mean = layers.Input(input_shapes[0][1:])
        std  = layers.Input(input_shapes[1][1:])
        sample  = self.rep_gauss([mean, std])
        self._init_graph_network([mean, std], [sample], name=self.m_name)
        super(SamplerGaussian, self).build(input_shapes)


class CollaborativeLatentCore(network.Network):
    '''
        The latent core for collaborative VAE
    '''
    def __init__(self,  
                 latent_size, 
                 **kwargs):
        super(CollaborativeLatentCore, self).__init__(**kwargs)
        self.m_name = kwargs.get("name", None)
        self.dense_mean = layers.Dense(latent_size, name="mean")
        self.dense_std  = layers.Dense(latent_size, name="logstd")
        self.clip = layers.Lambda(lambda x:K.clip(x, -20, 2), name="clip")
        self.exp = layers.Lambda(lambda x:K.exp(x), name="exp")
        self.z_sampler = SamplerGaussian(name="z_sampler")

    def build(self, input_shapes):
        x_in = layers.Input(input_shapes[1:])
        mean = self.dense_mean(x_in)
        std  = self.exp(self.clip(self.dense_std(x_in)))
        z_b = self.z_sampler([mean, std])
        self._init_graph_network([x_in], [mean, std, z_b], name=self.m_name)
        super(CollaborativeLatentCore, self).build(input_shapes)


class ExposureVariationalAutoencoder():
    '''
       	The exposure variational autoencoder
    '''
    def __init__(self, 
                 input_dim,
                 hidden_sizes,
                 latent_size,
                 encoder_activs,
                 decoder_activs,
                 dropout_rate=0.5):
        self.input_dim = input_dim
        self.latent_size = latent_size
        self.encoder = MLP(hidden_sizes, activations=encoder_activs, dropout_rate=dropout_rate, name="Encoder")
        self.encoder.build(input_shapes=[None, input_dim])
        self.latent_core = CollaborativeLatentCore(latent_size, name="Latent")
        self.decoder = MLP(hidden_sizes[::-1]+[input_dim], activations=decoder_activs, 
                           l2_normalize=False, dropout_rate=None, name="Decoder")

    def build_vae_train(self):
        '''
            Get the training form of the vae model
        '''
        if not hasattr(self, "vae_train"):
            r_in = layers.Input(shape=[self.input_dim,], name="ratings")
            h_mid = self.encoder(r_in)
            mu, std, z_b = self.latent_core(h_mid)
            r_rec = self.decoder(z_b)
            self.vae_train = models.Model(inputs=r_in, outputs=r_rec)
            self.add_gauss_loss = AddGaussianLoss()
            kl_loss = self.add_gauss_loss([mu, std])
            self.vae_train.add_loss(kl_loss)
        return self.vae_train

    def build_vae_infer(self):
        '''
            Get the posterior mean of latent variable for a user
        '''
        if not hasattr(self, "vae_infer"):
            r_in = layers.Input(shape=[self.input_dim,], name="ratings")
            h_mid = self.encoder(r_in)
            mu_b, _, _ = self.latent_core(h_mid)
            self.vae_infer = models.Model(inputs=r_in, outputs=mu_b)
        return self.vae_infer

    def build_vae_eval(self):
        '''
            For evaluation, use the mean deterministically
        '''
        if not hasattr(self, "vae_eval"):
            r_in = layers.Input(shape=[self.input_dim,], name="ratings")
            h_mid = self.encoder(r_in)
            mu_b, _, _ = self.latent_core(h_mid)
            r_rec = self.decoder(mu_b)
            self.vae_eval = models.Model(inputs=r_in, outputs=r_rec)
        return self.vae_eval

    def build_vae_gen(self):
        '''
            Generate inputs from user latent variable
        '''
        if not hasattr(self, "vae_gen"):
            z_in = layers.Input(shape=[self.latent_size,], name="latent_var")
            r_out = self.decoder(z_in)
            self.vae_gen = models.Model(inputs=z_in, outputs=r_out)
        return self.vae_gen

    def load_weights(self, weight_path):
        '''
            Load weights from a pretrained vae
        '''
        vae_model = self.build_vae_train()
        vae_model.load_weights(weight_path)


if __name__ == "__main__":
    pass