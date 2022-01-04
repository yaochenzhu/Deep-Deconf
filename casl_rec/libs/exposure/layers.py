import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras import backend as K

from tensorflow_probability import distributions as tfp


class DenseForSparse(layers.Layer):
    '''
        Dense layer where the input is a sparse matrix.
    '''
    def __init__(self, in_dim, out_dim, activation=None, **kwargs):
        super(DenseForSparse, self).__init__(**kwargs)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation = activations.get(activation)

        self.kernel = self.add_weight(
            "kernel", shape=(self.in_dim, self.out_dim),
            trainable=True, initializer="glorot_uniform",
        )
        self.bias = self.add_weight(
            "bias", shape=(self.out_dim),
            trainable=True, initializer="zeros",
        )

    def call(self, inputs):
        if isinstance(inputs, tf.SparseTensor):
            outputs = tf.add(tf.sparse.matmul(inputs, self.kernel), self.bias)
        else:
            outputs = tf.add(tf.matmul(inputs, self.kernel), self.bias)
        return self.activation(outputs)


class TransposedSharedDense(layers.Layer):
    '''
        Dense layer that shares weights (transposed) and bias 
        with another dense layer.
    '''

    def __init__(self, weights, activation=None, **kwargs):
        super(TransposedSharedDense, self).__init__(**kwargs)
        assert(len(weights) in [1, 2]), \
            "Specify the [kernel] or the [kernel] and [bias]."
        self.W = weights[0]

        if len(weights) == 1:
            b_shape = self.W.shape.as_list()[0]
            self.b = self.add_weight(shape=(b_shape),
                name="bias",
                trainable=True,
                initializer="zeros")
        else:
            self.b = weights[1]
        self.activate = activations.get(activation)

    def call(self, inputs):
        return self.activate(K.dot(inputs, K.transpose(self.W))+self.b)


class AddGaussianLoss(layers.Layer):
    '''
        Add the KL divergence between the variational 
        Gaussian distribution and the prior to loss.
    '''
    def __init__(self, 
                 **kwargs):
        super(AddGaussianLoss, self).__init__(**kwargs)             
        self.lamb_kl = self.add_weight(shape=(), 
                                       name="lamb_kl", 
                                       initializer="ones", 
                                       trainable=False)

    def call(self, inputs):
        mu, std  = inputs
        ### The one used by the tensorflow community
        kl_loss =  self.lamb_kl * 0.5 * tf.reduce_mean(
            tf.reduce_sum(tf.square(mu) + tf.square(std) - 2 * tf.log(std) - 1, 1
        ))

        ### Or you can use the tfp package
        '''
        var_dist = tfp.MultivariateNormalDiag(loc=mu, scale_diag=std)
        pri_dist = tfp.MultivariateNormalDiag(loc=K.zeros_like(mu), 
                                              scale_diag=K.ones_like(std))    
        kl_loss  = self.lamb_kl*K.mean(tfp.kl_divergence(var_dist, pri_dist))
        '''
        return kl_loss


class AddBernoulliLoss(layers.Layer):
    '''
        Add the KL divergence between the variational
        Bernoulli distribution and the prior to the loss
    '''
    def __init__(self,
                 prior_logits,
                 **kwargs):
        super(AddBernoulliLoss, self).__init__(**kwargs)
        self.prior_logits = prior_logits
        self.lamb_kl = self.add_weight(shape=(), 
                                       name="lamb_kl", 
                                       initializer="ones", 
                                       trainable=False)

    def call(self, inputs):
        if self.prior_logits is None:
            self.prior_logits = K.zeros_like(inputs)

        prob = tf.clip_by_value(K.sigmoid(inputs), 1e-10, 1-1e-10)
        prior_prob = tf.clip_by_value(K.sigmoid(self.prior_logits), 1e-10, 1-1e-10)
        log_prob, log_prior_prob = K.log(prob), K.log(prior_prob)
        log_one_minus_prob, log_one_minus_prior_prob = K.log(1-prob), K.log(1-prior_prob)
        kl_loss = self.lamb_kl * tf.reduce_mean(
            prob * log_prob + (1 - prob) * log_one_minus_prob - 
            prob * log_prior_prob - (1 - prob) * log_one_minus_prior_prob
        )
        ### Or you can use the tfp package
        '''
        var_dist = tfp.Bernoulli(logits=inputs)
        pri_dist = tfp.Bernoulli(logits=self.prior_logits)
        kl_loss = self.lamb_kl * K.mean(tfp.kl_divergence(var_dist, pri_dist))
        '''
        return kl_loss


class ReparameterizeGaussian(layers.Layer):
    '''
        Rearameterization trick for Gaussian
    '''
    def __init__(self, **kwargs):
        super(ReparameterizeGaussian, self).__init__(**kwargs)

    def call(self, stats):
        mu, std = stats
        dist = tfp.MultivariateNormalDiag(loc=mu, scale_diag=std)
        return dist.sample()


class ReparameterizeBernoulli(layers.Layer):
    '''
        Rearameterization trick for Bernoulli
    '''
    def __init__(self, **kwargs):
        super(ReparameterizeBernoulli, self).__init__(**kwargs)
        self.temp = self.add_weight(shape=(), 
                                    name="temp", 
                                    initializer="ones", 
                                    trainable=False)

    def call(self, logits):
        gumbel_dist = tfp.Gumbel(0, 1)
        gumbel_add = gumbel_dist.sample(K.shape(logits))
        gumbel_sub = gumbel_dist.sample(K.shape(logits))
        berno_samples = K.sigmoid((logits+gumbel_add-gumbel_sub)/self.temp)
        return berno_samples