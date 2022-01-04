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


class RatingOutcomeVariationalAutoencoder():
    '''
       	The collaborative variational autoencoder for outcome prediction
    '''
    def __init__(self, 
                 name_dim_dict,
                 hidden_sizes,
                 latent_size,
                 encoder_activs,
                 decoder_activs,
                 dropout_rate=0.5):
        self.name_dim_dict = name_dim_dict
        self.latent_size = latent_size
        if "exposure" in self.name_dim_dict.keys() or "features" in self.name_dim_dict.keys():
            self.encoder = MLP(hidden_sizes, activations=encoder_activs, 
                               dropout_rate=dropout_rate, name="Encoder")
            self.latent_core = CollaborativeLatentCore(latent_size, name="Latent")
        self.num_items = self.name_dim_dict["ratings"]
        self.decoder = MLP(hidden_sizes[::-1]+[self.num_items], activations=decoder_activs, 
                           l2_normalize=False, dropout_rate=None, name="Decoder")

    def build_outcome_model(self):
        '''
            Get the training form of the outcome prediction model
        '''
        if not hasattr(self, "vae_train"):
            ### Placeholder for substitute confounders
            conf_dim = self.name_dim_dict["subs_conf"]
            subs_in = layers.Input(shape=[conf_dim], name="susb_conf")
            ### Opitional predictor variables/treatment assignment
            pred_ins = []
            if "exposure" in self.name_dim_dict.keys():
                exp_dim = self.name_dim_dict["exposure"]
                pred_ins.append(layers.Input(shape=[exp_dim,], name="exposure"))
            if "features" in self.name_dim_dict.keys():
                feat_dim = self.name_dim_dict["features"]
                pred_ins.append(layers.Input(shape=[feat_dim, ], name="features"))
            ### Control the substitute confounders
            if pred_ins == []:
                ### All-one treatment, simplified model
                z_dcf_train = subs_in
                z_dcf_val = subs_in
            else:
                ### Forward pass for predictors
                pr_in = layers.Concatenate(axis=-1)(pred_ins)
                h_mid = self.encoder(pr_in)
                mu, std, z_b = self.latent_core(h_mid)
                #z_dcf_train = layers.Add()([z_b, subs_in])
                #z_dcf_val =layers.Add()([mu, subs_in])
                z_dcf_train = layers.Concatenate(axis=-1)([z_b, subs_in])
                z_dcf_val = layers.Concatenate(axis=-1)([mu, subs_in])
            r_rec_train = self.decoder(z_dcf_train)
            self.vae_train = models.Model(inputs=[subs_in]+pred_ins, outputs=r_rec_train)
            if pred_ins != []:
                self.add_gauss_loss = AddGaussianLoss()
                kl_loss = self.add_gauss_loss([mu, std])
                self.vae_train.add_loss(kl_loss)
            r_rec_val = self.decoder(z_dcf_val)
            self.vae_val = models.Model(inputs=[subs_in]+pred_ins, outputs=r_rec_val)
        return self.vae_train, self.vae_val

    def load_weights(self, weight_path):
        '''
            Load weights from a pretrained vae
        '''
        self.build_outcome_model()
        self.vae_train.load_weights(weight_path)


if __name__ == "__main__":
    pass