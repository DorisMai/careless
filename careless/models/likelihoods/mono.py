from careless.models.likelihoods.base import Likelihood, BaseModel
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow_probability import util as tfu
from tensorflow_probability import bijectors as tfb
import numpy as np

class LocationScaleLikelihood(Likelihood):
    def get_loc_and_scale(self, inputs):
        loc   = self.get_intensities(inputs)
        scale = self.get_uncertainties(inputs)
        return tf.squeeze(loc), tf.squeeze(scale)

class NormalLikelihood(LocationScaleLikelihood):
    def call(self, inputs):
        return tfd.Normal(*self.get_loc_and_scale(inputs))

class LaplaceLikelihood(LocationScaleLikelihood):
    def call(self, inputs):
        loc, scale = self.get_loc_and_scale(inputs)
        return tfd.Laplace(loc, scale/np.sqrt(2.))

class StudentTLikelihood(LocationScaleLikelihood):
    def __init__(self, dof):
        """
        Parameters
        ----------
        dof : float
            Degrees of freedom of the student t likelihood.
        """
        super().__init__()
        self.dof = dof

    def call(self, inputs):
        return tfd.StudentT(self.dof, *self.get_loc_and_scale(inputs))

class XtalWeightedLikelihood(LocationScaleLikelihood):
    def  __init__(self, num_files, wckl_weight, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_files = num_files
        self.num_reflections = None
        self._inv_wc = tf.Variable(tf.ones(self.num_files - 1))
        self.wckl_weight = wckl_weight
        self.loc = None
        self.scale = None
        self.file_ids = None

    def distribution(self, loc, scale):
        raise NotImplementedError("Weighted likelihoods must implement self.distribution \
                                  which should have a log_prob method.")

    # @property
    # def norm_inv_wc(self):
    #     inv_wc = tf.concat([tf.constant([1.], dtype=tf.float32), self._inv_wc], axis=-1)
    #     invweight = tf.gather(inv_wc, self.file_ids)
    #     log_Z = tf.reduce_logsumexp(tf.math.log(invweight))
    #     return tf.exp(inv_wc - log_Z)

    def call(self, inputs):
        self.loc, self.scale = self.get_loc_and_scale(inputs)
        self.file_ids = self.get_file_id(inputs)
        self.num_reflections = self.file_ids.shape[0]
        return self
    
    def corrected_sigiobs(self, ipred):
        inv_wc = tf.concat([tf.constant([1.], dtype=tf.float32), self._inv_wc], axis=-1)
        invweight = tf.gather(inv_wc, self.file_ids)
        log_Z = tf.reduce_logsumexp(tf.math.log(invweight))
        norm_inv_wc = tf.exp(inv_wc - log_Z)
        invweight = tf.gather(norm_inv_wc, self.file_ids)

        # Compute wc KL divergence
        wc_prior = tfd.Categorical(probs=tf.ones(self.num_reflections) / tf.cast(self.num_reflections, tf.float32))
        wc_posterior = tfd.Categorical(probs=invweight)
        kl_div = wc_posterior.kl_divergence(wc_prior)
        wckl_loss = tf.reduce_mean(self.wckl_weight * kl_div)
        self.add_loss(wckl_loss)
        self.add_metric(wckl_loss, name='wckl_loss')

        invweight *= tf.cast(self.num_reflections, tf.float32)
        reshaped_invweight = tf.broadcast_to(tf.transpose(invweight), ipred.shape)
        ipred = tf.math.softplus(ipred)
        sigiobs = self.scale * reshaped_invweight
        return sigiobs
    
class NormalXtalWeightedLikelihood(XtalWeightedLikelihood):
    def log_prob(self, ipred):
        scale = self.corrected_sigiobs(ipred)
        return tfd.Normal(self.loc, scale).log_prob(ipred)

class StudentTXtalWeightedLikelihood(XtalWeightedLikelihood):
    def __init__(self, dof, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dof = dof

    def log_prob(self, ipred):
        scale = self.corrected_sigiobs(ipred)
        return tfd.StudentT(self.dof, self.loc, scale).log_prob(ipred)


class Ev11Likelihood(LocationScaleLikelihood):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Sdfac = tfu.TransformedVariable(1., tfb.Softplus())
        self.Sdadd = tfu.TransformedVariable(1., tfb.Softplus())
        self.SdB = tfu.TransformedVariable(1., tfb.Softplus())
        self.loc = None
        self.scale = None

    def call(self, inputs):
        self.loc, self.scale = self.get_loc_and_scale(inputs)
        return self

    def corrected_sigiobs(self, ipred):
        ipred = tf.math.softplus(ipred)
        sigiobs = self.Sdfac * tf.math.sqrt(
            tf.square(self.scale) + \
            self.SdB * ipred + \
            self.Sdadd * tf.square(ipred)
        )
        return sigiobs

class NormalEv11Likelihood(Ev11Likelihood):
    def log_prob(self, ipred):
        scale = self.corrected_sigiobs(ipred)
        return tfd.Normal(self.loc, scale).log_prob(ipred)

class StudentTEv11Likelihood(Ev11Likelihood):
    def __init__(self, dof, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dof = dof

    def log_prob(self, ipred):
        scale = self.corrected_sigiobs(ipred)
        return tfd.StudentT(self.dof, self.loc, scale).log_prob(ipred)

class NeuralLikelihood(Likelihood):
    def __init__(self, mlp_layers, mlp_width):
        super().__init__()

        layers = []
        for _ in range(mlp_layers):
            layer = tf.keras.layers.Dense(
                mlp_width,
                activation=tf.keras.layers.LeakyReLU(),
            )
            layers.append(layer)

        layer = tf.keras.layers.Dense(
            1,
            activation='softplus',
        )
        layers.append(layer)
        self.network = tf.keras.models.Sequential(layers)

    def base_dist(self, loc, scale):
        raise NotImplementedError("extensions of this class must implement a base_dist(loc, scale) method")

    def call(self, inputs):
        iobs = self.get_intensities(inputs)
        #metadata = self.get_metadata(inputs)
        sigiobs = self.get_uncertainties(inputs)
        delta = self.network(tf.concat((iobs, sigiobs), axis=-1))
        sigpred = sigiobs * delta / tf.reduce_mean(delta)
        return self.base_dist(
            tf.squeeze(iobs), 
            tf.squeeze(sigpred),
        )

class NeuralNormalLikelihood(NeuralLikelihood):
    def base_dist(self, loc, scale):
        return tfd.Normal(loc, scale)

# class WeightedLikelihoodDistribution(BaseModel):
#     def __init__(self, base_distribution, weights):
#         #super().__init__()
#         self.base_distribution = base_distribution
#         self.xtal_weights = weights

#     def log_prob(self, data):
#         reshaped_weights = tf.broadcast_to(tf.transpose(self.xtal_weights), data.shape)
#         return reshaped_weights * self.base_distribution.log_prob(data)

# class WeightedLikelihood(LocationScaleLikelihood):
#     def __init__(self, num_files):
#         super().__init__()
#         self.num_files = num_files
#         self.raw_wc = self.add_weight(shape=(self.num_files - 1,), initializer="zeros",
#                                       trainable=True, dtype=tf.float32, name='raw_wc')

#     @property
#     def norm_wc(self):
#         return tf.nn.softmax(tf.concat([tf.constant([0.], dtype=tf.float32), self.raw_wc], axis=0))

#     def distribution(self, loc, scale):
#         raise NotImplementedError("Weighted likelihoods must implement self.distribution \
#                                   which should have a log_prob method.")

#     def call(self, inputs):
#         loc, scale = self.get_loc_and_scale(inputs)
#         file_ids = self.get_file_id(inputs)
#         xtal_wc = tf.gather(self.norm_wc, file_ids)
#         xtal_wc = xtal_wc / tf.reduce_mean(xtal_wc)
#         base_dist = self.distribution(loc, scale)
#         likelihood = WeightedLikelihoodDistribution(base_dist, xtal_wc)
#         for i in range(self.num_files-1):
#             self.add_metric(self.raw_wc[i], name=f'raw_wc_{i}')
#             # self.add_metric(self.norm_wc[i], name=f'norm_wc_{i}')
#         #tf.print(self.norm_wc, summarize=-1, output_stream='file://./logs/norm_weights.log')
#         #tf.print(self.raw_wc, summarize=-1, output_stream='file://./logs/raw_weights.log')
#         return likelihood
    
# class NormalWeightedLikelihood(WeightedLikelihood):
#     def distribution(self, loc, scale):
#         return tfd.Normal(loc, scale)

# class StudentTWeightedLikelihood(WeightedLikelihood):
#     def __init__(self, dof, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.dof = dof

#     def distribution(self, loc, scale):
#         return tfd.StudentT(self.dof, loc, scale)
