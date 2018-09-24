"""GenerativeModel class for ... generative models"""

import numpy as np
import tensorflow as tf


class GenerativeModel(object):
    """Base class for generative models"""

    def __init__(
            self, dim_obs=None, dim_latent=None, post_z_samples=None, rng=123,
            dtype=tf.float32, **kwargs):
        """
        Set base class attributes

        Args:
            dim_obs (int): dimension of observation vector
            dim_latent (int): dimension of latent state
            post_z_samples (T x dim_latent tf Tensor): samples from the (appx)
                posterior of the latent states
            rng (int): rng seed for generating samples of observations
            dtype (tf.Dtype): data type for all model variables, placeholders

        """

        # set basic dims
        self.dim_latent = dim_latent
        self.dim_obs = dim_obs
        # tf Tensor that contains samples from the approximate posterior
        # (output of inference network)
        self.post_z_samples = post_z_samples
        # set rng seed for drawing samples from observation distribution
        self.rng = rng
        # # create list of output placeholders
        # self.obs_ph = None
        # use same data type throughout graph construction
        self.dtype = dtype

    def build_graph(self, *args, **kwargs):
        """Build tensorflow computation graph for generative model"""
        raise NotImplementedError

    def evaluate_log_density(self, y, z):
        """Evaluate log density of generative model"""
        raise NotImplementedError

    def get_params(self, sess):
        """Get parameters of generative model"""
        raise NotImplementedError

    def generate_samples(self, sess, num_samples=1):
        """Draw samples from model"""
        raise NotImplementedError


class LDS(GenerativeModel):
    """Linear dynamical system model with Gaussian observations"""

    def __init__(
            self, dim_obs=None, dim_latent=None, post_z_samples=None, rng=123,
            dtype=tf.float32, num_time_pts=None, gen_params=None):
        """
        Generative model is defined as
            z_t = A z_{t-1} + \eps
            y_t = C z_t + d + \eta

        Args:
            num_time_pts (int): number of time points per observation of the
                dynamical sequence
            gen_params (dict): dictionary of generative params for initializing
                model

        """

        super(LDS, self).__init__(
            dim_obs=dim_obs, dim_latent=dim_latent,
            post_z_samples=post_z_samples, rng=rng, dtype=dtype)
        self.num_time_pts = num_time_pts
        if gen_params is None:
            self.gen_params = {}
        else:
            self.gen_params = gen_params

    def build_graph(self, *args, **kwargs):
        """
        Build tensorflow computation graph for generative model

        Args:
            z_samples (num_time_pts x dim_latent tf.Tensor): samples of the
                latent states

        """

        z_samples = args[0]

        # construct data pipeline
        with tf.variable_scope('observations'):
            self.obs_ph = tf.placeholder(
                dtype=self.dtype,
                shape=[None, self.num_time_pts, self.dim_obs],
                name='obs_ph')

        with tf.variable_scope('model_params'):
            self.initialize_prior_vars()

        # initialize mapping from latent space to observations
        with tf.variable_scope('mapping'):
            self._initialize_mapping()
            self.y_pred = self._apply_mapping(z_samples)

        # define branch of graph for evaluating prior model
        with tf.variable_scope('generative_samples'):
            self._sample_yz()

    def initialize_prior_vars(self):
        """Initialize variables of prior"""

        tr_norm_initializer = tf.initializers.truncated_normal(
            mean=0.0, stddev=0.1, dtype=self.dtype)
        zeros_initializer = tf.initializers.zeros(dtype=self.dtype)

        # mean of initial latent state
        if 'z0_mean' in self.gen_params:
            self.z0_mean = tf.get_variable(
                'z0_mean',
                initializer=self.gen_params['z0_mean'],
                dtype=self.dtype)
        else:
            self.z0_mean = tf.get_variable(
                'z0_mean',
                shape=[1, self.dim_latent],
                initializer=zeros_initializer,
                dtype=self.dtype)

        # means of transition matrix
        if 'A' in self.gen_params:
            self.A = tf.get_variable(
                'A',
                initializer=self.gen_params['A'],
                dtype=self.dtype)
        else:
            self.A = tf.get_variable(
                'A',
                initializer=0.5 * np.eye(self.dim_latent,
                                         dtype=self.dtype.as_numpy_dtype),
                dtype=self.dtype)

        # square root of the innovation precision matrix
        if 'Q_sqrt' in self.gen_params:
            self.Q_sqrt = tf.get_variable(
                'Q_sqrt',
                initializer=self.gen_params['Q_sqrt'],
                dtype=self.dtype)
        else:
            # Q_sqrt = tf.get_variable(
            #     'Q_sqrt',
            #     shape=[self.dim_latent, self.dim_latent],
            #     initializer=tr_norm_initializer,
            #     dtype=self.dtype)
            self.Q_sqrt = tf.get_variable(
                'Q_sqrt',
                initializer=0.1 * np.eye(
                    self.dim_latent,
                    dtype=self.dtype.as_numpy_dtype),
                dtype=self.dtype)

        # square root of the initial innovation precision matrix
        if 'Q0_sqrt' in self.gen_params:
            self.Q0_sqrt = tf.get_variable(
                'Q0_sqrt',
                initializer=self.gen_params['Q0_sqrt'],
                dtype=self.dtype)
        else:
            # Q0_sqrt = tf.get_variable(
            #     'Q0_sqrt',
            #     shape=[self.dim_latent, self.dim_latent],
            #     initializer=tr_norm_initializer,
            #     dtype=self.dtype)
            self.Q0_sqrt = tf.get_variable(
                'Q0_sqrt',
                initializer=0.1 * np.eye(
                    self.dim_latent,
                    dtype=self.dtype.as_numpy_dtype),
                dtype=self.dtype)

        diag = 1e-6 * np.eye(self.dim_latent,
                             dtype=self.dtype.as_numpy_dtype)

        self.Q0 = tf.matmul(
            self.Q0_sqrt, self.Q0_sqrt, transpose_b=True, name='Q0') + diag
        self.Q = tf.matmul(
            self.Q_sqrt, self.Q_sqrt, transpose_b=True, name='Q') + diag

        self.Q0inv = tf.matrix_inverse(self.Q0, name='Q0inv')
        self.Qinv = tf.matrix_inverse(self.Q, name='Qinv')

    def _initialize_mapping(self):
        """Initialize mapping from latent space to observations"""

        # should eventually become user options
        tr_norm_initializer = tf.initializers.truncated_normal(
            mean=0.0, stddev=0.1, dtype=self.dtype)
        zeros_initializer = tf.initializers.zeros(dtype=self.dtype)
        activation = None
        use_bias = True
        kernel_initializer = tr_norm_initializer
        bias_initializer = zeros_initializer
        kernel_regularizer = None
        bias_regularizer = None
        num_layers = 1

        # square root of diagonal of observation covariance matrix
        # (assume diagonal)
        if 'Rsqrt' in self.gen_params:
            self.Rsqrt = tf.get_variable(
                'Rsqrt',
                initializer=self.gen_params['Rsqrt'],
                dtype=self.dtype)
        else:
            self.Rsqrt = tf.get_variable(
                'Rsqrt',
                shape=[1, self.dim_obs],
                initializer=tr_norm_initializer,
                dtype=self.dtype)
        self.R = tf.square(self.Rsqrt)
        self.Rinv = 1.0 / self.R

        # observation matrix
        # self.mapping = tf.layers.Dense(
        #     units=self.dim_obs,
        #     activation=None,
        #     use_bias=True,
        #     kernel_initializer=kernel_initializer,
        #     bias_initializer=bias_initializer,
        #     kernel_regularizer=kernel_regularizer,
        #     bias_regularizer=bias_regularizer,
        #     name='mapping')

        if 'C' in self.gen_params:
            self.C = tf.get_variable(
                'C',
                initializer=self.gen_params['C'],
                dtype=self.dtype)
        else:
            self.C = tf.get_variable(
                'C',
                shape=[self.dim_latent, self.dim_obs],
                initializer=tr_norm_initializer,
                dtype=self.dtype)

        # biases
        if 'd' in self.gen_params:
            self.d = tf.get_variable(
                'd',
                initializer=self.gen_params['d'],
                dtype=self.dtype)
        else:
            self.d = tf.get_variable(
                'd',
                shape=[1, self.dim_obs],
                initializer=tr_norm_initializer,
                dtype=self.dtype)

    def _apply_mapping(self, z_samples):
        """Apply model mapping from latent space to observations"""
        return tf.add(tf.tensordot(z_samples, self.C, axes=[[2], [0]]), self.d)
        # return self.mapping.apply(z_samples)

    def _sample_yz(self):
        """
        Define branch of tensorflow computation graph for sampling from the
        prior
        """

        self.num_samples_ph = tf.placeholder(
            dtype=tf.int32, shape=None, name='num_samples_ph')

        self.latent_rand_samples = tf.random_normal(
            shape=[self.num_samples_ph, self.num_time_pts, self.dim_latent],
            mean=0.0, stddev=1.0, dtype=self.dtype, name='latent_rand_samples')

        self.obs_rand_samples = tf.random_normal(
            shape=[self.num_samples_ph, self.num_time_pts, self.dim_obs],
            mean=0.0, stddev=1.0, dtype=self.dtype, name='obs_rand_samples')

        def lds_update(outputs, inputs):

            [z_val, y_val] = outputs
            [rand_z, rand_y] = inputs

            z_val = tf.matmul(z_val, tf.transpose(self.A)) \
                + tf.matmul(rand_z, tf.transpose(self.Q_sqrt))
            y_val = tf.squeeze(
                self._apply_mapping(tf.expand_dims(z_val, axis=1)), axis=1) \
                + tf.multiply(rand_y, self.Rsqrt)

            return [z_val, y_val]

        z0 = self.z0_mean \
            + tf.matmul(self.latent_rand_samples[:, 0, :],
                        tf.transpose(self.Q0_sqrt))
        y0 = tf.squeeze(
            self._apply_mapping(tf.expand_dims(z0, axis=1)), axis=1) \
            + tf.multiply(self.obs_rand_samples[:, 0, :], self.Rsqrt)

        # scan over time points, not samples
        rand_ph_shuff = tf.transpose(
            self.latent_rand_samples[:, 1:, :], perm=[1, 0, 2])
        obs_noise_ph_shuff = tf.transpose(
            self.obs_rand_samples[:, 1:, :], perm=[1, 0, 2])
        samples = tf.scan(
            fn=lds_update,
            elems=[rand_ph_shuff, obs_noise_ph_shuff],
            initializer=[z0, y0])

        z_samples_unshuff = tf.transpose(samples[0], perm=[1, 0, 2])
        y_samples_unshuff = tf.transpose(samples[1], perm=[1, 0, 2])

        self.z_samples_prior = tf.concat(
            [tf.expand_dims(z0, axis=1), z_samples_unshuff], axis=1)
        self.y_samples_prior = tf.concat(
            [tf.expand_dims(y0, axis=1), y_samples_unshuff], axis=1)

    def evaluate_log_density(self, y, z):
        """
        Evaluate log density for generative model, defined as
        p(y, z) = p(y | z) p(z)
        where
        p(z) = \prod_t p(z_t), z_t ~ N(A z_{t-1}, Q)
        p(y | z) = \prod_t p(y_t | z_t), y_t | z_t ~ N(C z_t + d, R)

        Args:
            y (num_samples x num_time_pts x dim_obs tf.Tensor)
            z (num_samples x num_time_pts x dim_latent tf.Tensor)

        Returns:
            float: log density over y and z, averaged over minibatch samples

        """

        # likelihood
        with tf.variable_scope('likelihood'):
            self.res_y = self.obs_ph - y
            self.log_density_y = -0.5 * (
                tf.reduce_sum(tf.reduce_mean(
                    tf.multiply(tf.square(self.res_y), self.Rinv), axis=0))
                + self.num_time_pts * tf.reduce_sum(tf.log(self.R))
                + self.num_time_pts * self.dim_obs * tf.log(2.0 * np.pi))

        # prior
        with tf.variable_scope('prior'):
            self.res_z0 = z[:, 0, :] - self.z0_mean
            self.res_z = z[:, 1:, :] \
                - tf.tensordot(z[:, :-1, :], tf.transpose(self.A),
                               axes=[[2], [0]])
            self.log_density_z = -0.5 * (
                tf.reduce_sum(tf.reduce_mean(tf.multiply(
                    tf.tensordot(self.res_z, self.Qinv, axes=[[2], [0]]),
                    self.res_z), axis=0))
                + tf.reduce_sum(tf.reduce_mean(tf.multiply(
                    tf.matmul(self.res_z0, self.Q0inv), self.res_z0), axis=0))
                + (self.num_time_pts-1) * tf.log(tf.matrix_determinant(self.Q))
                + tf.log(tf.matrix_determinant(self.Q0))
                + self.num_time_pts * self.dim_latent * tf.log(2.0 * np.pi))

        # tf.summary.scalar('mat_det_Q0', tf.matrix_determinant(self.Q0))
        # tf.summary.scalar('mat_det_Q', tf.matrix_determinant(self.Q))

        return self.log_density_y + self.log_density_z

    def generate_samples(self, sess, num_samples=1):
        """
        Generate samples from the model

        Args:
            sess (tf.Session object)
            num_samples (int, optional)

        Returns:
            num_time_pts x dim_obs x num_samples numpy array:
                sample observations y
            num_time_pts x dim_latent x num_samples numpy array:
                sample latent states z

        """

        [y, z] = sess.run(
            [self.y_samples_prior, self.z_samples_prior],
            feed_dict={self.num_samples_ph: num_samples})

        return y, z

    def get_params(self, sess):
        """Get parameters of generative model"""

        A, C, d, Rsqrt, z0_mean, Q, Q0 = sess.run(
            [self.A, self.C, self.d, self.Rsqrt,
             self.z0_mean, self.Q, self.Q0])

        param_dict = {
            'A': A, 'C': C, 'd': d, 'R': np.square(Rsqrt),
            'z0_mean': z0_mean, 'Q': Q, 'Q0': Q0}

        return param_dict


class LDSCoupled(LDS):
    """
    Linear dynamical system model with Gaussian observations; model parameters
    are coupled to parameters of approximate posterior SmoothingLDSCoupled
    through the use of the LDSCoupled Model class
    """

    def __init__(
            self, dim_obs=None, dim_latent=None, post_z_samples=None, rng=123,
            dtype=tf.float32, num_time_pts=None, gen_params=None):
        """
        Generative model is defined as
            z_t = A z_{t-1} + \eps
            y_t = C z_t + d + \eta

        Args:
            num_time_pts (int): number of time points per observation of the
                dynamical sequence
            gen_params (dict): dictionary of generative params for initializing
                model

        """

        super(LDSCoupled, self).__init__(
            dim_obs=dim_obs, dim_latent=dim_latent,
            post_z_samples=post_z_samples, rng=rng, dtype=dtype)
        self.num_time_pts = num_time_pts
        if gen_params is None:
            self.gen_params = {}
        else:
            self.gen_params = gen_params

    def build_graph(
            self, z_samples, z0_mean, A, Q_sqrt, Q, Qinv, Q0_sqrt, Q0, Q0inv):
        """
        Build tensorflow computation graph for generative model

        Args:
            z_samples (num_time_pts x dim_latent tf.Tensor): samples of the
                latent states
            z0_mean
            A
            Q_sqrt
            Q
            Qinv
            Q0_sqrt
            Q0
            Q0inv

        """

        # construct data pipeline
        with tf.variable_scope('observations'):
            self.obs_ph = tf.placeholder(
                dtype=self.dtype,
                shape=[None, self.num_time_pts, self.dim_obs],
                name='obs_ph')

        # make variables shared with inference model attributes
        self.z0_mean = z0_mean
        self.A = A
        self.Q0_sqrt = Q0_sqrt
        self.Q_sqrt = Q_sqrt
        self.Q0 = Q0
        self.Q = Q
        self.Q0inv = Q0inv
        self.Qinv = Qinv

        # initialize mapping from latent space to observations
        with tf.variable_scope('mapping'):
            self._initialize_mapping()
            self.y_pred = self._apply_mapping(z_samples)

        # define branch of graph for evaluating prior model
        with tf.variable_scope('generative_samples'):
            self._sample_yz()

    def initialize_prior_vars(self):
        """Initialize variables of prior"""

        tr_norm_initializer = tf.initializers.truncated_normal(
            mean=0.0, stddev=0.1, dtype=self.dtype)
        zeros_initializer = tf.initializers.zeros(dtype=self.dtype)

        # mean of initial latent state
        if 'z0_mean' in self.gen_params:
            z0_mean = tf.get_variable(
                'z0_mean',
                initializer=self.gen_params['z0_mean'],
                dtype=self.dtype)
        else:
            z0_mean = tf.get_variable(
                'z0_mean',
                shape=[1, self.dim_latent],
                initializer=zeros_initializer,
                dtype=self.dtype)

        # means of transition matrix
        if 'A' in self.gen_params:
            A = tf.get_variable(
                'A',
                initializer=self.gen_params['A'],
                dtype=self.dtype)
        else:
            A = tf.get_variable(
                'A',
                initializer=0.5 * np.eye(self.dim_latent,
                                         dtype=self.dtype.as_numpy_dtype()),
                dtype=self.dtype)

        # square root of the innovation precision matrix
        if 'Q_sqrt' in self.gen_params:
            Q_sqrt = tf.get_variable(
                'Q_sqrt',
                initializer=self.gen_params['Q_sqrt'],
                dtype=self.dtype)
        else:
            # Q_sqrt = tf.get_variable(
            #     'Q_sqrt',
            #     shape=[self.dim_latent, self.dim_latent],
            #     initializer=tr_norm_initializer,
            #     dtype=self.dtype)
            Q_sqrt = tf.get_variable(
                'Q_sqrt',
                initializer=np.eye(
                    self.dim_latent,
                    dtype=self.dtype.as_numpy_dtype()),
                dtype=self.dtype)

        # square root of the initial innovation precision matrix
        if 'Q0_sqrt' in self.gen_params:
            Q0_sqrt = tf.get_variable(
                'Q0_sqrt',
                initializer=self.gen_params['Q0_sqrt'],
                dtype=self.dtype)
        else:
            # Q0_sqrt = tf.get_variable(
            #     'Q0_sqrt',
            #     shape=[self.dim_latent, self.dim_latent],
            #     initializer=tr_norm_initializer,
            #     dtype=self.dtype)
            Q0_sqrt = tf.get_variable(
                'Q0_sqrt',
                initializer=np.eye(
                    self.dim_latent,
                    dtype=self.dtype.as_numpy_dtype()),
                dtype=self.dtype)

        Q0 = tf.matmul(Q0_sqrt, Q0_sqrt, transpose_b=True, name='Q0')
        Q = tf.matmul(Q_sqrt, Q_sqrt, transpose_b=True, name='Q')
        Q0inv = tf.matrix_inverse(Q0, name='Q0inv')
        Qinv = tf.matrix_inverse(Q, name='Qinv')

        return z0_mean, A, Q_sqrt, Q, Qinv, Q0_sqrt, Q0, Q0inv


class PLDS(LDS):
    pass


class PLDSCoupled(LDSCoupled):
    pass


class FLDS(LDS):
    pass


class FLDSCoupled(LDSCoupled):
    pass


class PFLDS(PLDS):
    pass


class PFLDSCoupled(PLDSCoupled):
    pass
