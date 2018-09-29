"""GenerativeModel class for ... generative models"""

import numpy as np
import tensorflow as tf


class GenerativeModel(object):
    """Base class for generative models"""

    def __init__(
            self, dim_obs=None, dim_latent=None, post_z_samples=None,
            **kwargs):
        """
        Set base class attributes

        Args:
            dim_obs (int): dimension of observation vector
            dim_latent (int): dimension of latent state
            post_z_samples (batch_size x num_mc_samples x num_time_pts x
                dim_latent tf.Tensor): samples from the (appx) posterior of the
                latent states

        """

        # set basic dims
        self.dim_latent = dim_latent
        self.dim_obs = dim_obs
        # tf.Tensor that contains samples from the approximate posterior
        # (output of inference network)
        self.post_z_samples = post_z_samples
        # use same data type throughout graph construction
        self.dtype = tf.float32

    def build_graph(self, *args, **kwargs):
        """Build tensorflow computation graph for generative model"""
        raise NotImplementedError

    def log_density(self, y, z):
        """Evaluate log density of generative model"""
        raise NotImplementedError

    def sample(self, sess, num_samples=1, seed=None):
        """Draw samples from model"""
        raise NotImplementedError


class LDS(GenerativeModel):
    """Linear dynamical system model with Gaussian observations"""

    def __init__(
            self, dim_obs=None, dim_latent=None, post_z_samples=None,
            num_time_pts=None, gen_params=None, noise_dist='gaussian'):
        """
        Generative model is defined as
            z_t = A z_{t-1} + \eps
            y_t = C z_t + d + \eta

        Args:
            num_time_pts (int): number of time points per observation of the
                dynamical sequence
            gen_params (dict): dictionary of generative params for initializing
                model
            noise_dist (str): 'gaussian' | 'poisson'

        """

        super().__init__(
            dim_obs=dim_obs, dim_latent=dim_latent,
            post_z_samples=post_z_samples)
        self.num_time_pts = num_time_pts
        if gen_params is None:
            self.gen_params = {}
        else:
            self.gen_params = gen_params

        # emissions matrix
        if 'C' in self.gen_params:
            kernel_initializer = tf.constant_initializer(
                self.gen_params['C'], dtype=self.dtype)
        else:
            kernel_initializer = tf.initializers.truncated_normal(
                mean=0.0, stddev=0.1, dtype=self.dtype)

        # biases
        if 'd' in self.gen_params:
            bias_initializer = tf.constant_initializer(
                self.gen_params['d'], dtype=self.dtype)
        else:
            bias_initializer = tf.initializers.zeros(dtype=self.dtype)

        # spiking nl
        self.noise_dist = noise_dist
        if noise_dist is 'gaussian':
            activation = None
        elif noise_dist is 'poisson':
            activation = tf.nn.softplus
        else:
            raise ValueError

        # list of dicts specifying (linear) nn to observations
        self.nn_params = [{
            'units': self.dim_obs,
            'activation': activation,
            'kernel_initializer': kernel_initializer,
            'bias_initializer': bias_initializer,
            'kernel_regularizer': None,
            'bias_regularizer': None,
            'name': 'mapping'}]

    def build_graph(self, *args, **kwargs):
        """
        Build tensorflow computation graph for generative model

        Args:
            z_samples (batch_size x num_mc_samples x num_time_pts x dim_latent
                tf.Tensor): samples of the latent states

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
            self._initialize_noise_dist_vars()
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

    def _initialize_noise_dist_vars(self):

        if self.noise_dist is 'gaussian':
            tr_norm_initializer = tf.initializers.truncated_normal(
                mean=0.0, stddev=0.1, dtype=self.dtype)
            zeros_initializer = tf.initializers.zeros(dtype=self.dtype)

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

    def _initialize_mapping(self):
        """Initialize mapping from latent space to observations"""
        self.layers = []
        for _, layer_params in enumerate(self.nn_params):
            self.layers.append(tf.layers.Dense(**layer_params))

    def _apply_mapping(self, z_samples):
        """Apply model mapping from latent space to observations"""

        net_input = z_samples
        for _, layer in enumerate(self.layers):
            net_input = layer.apply(net_input)

        return net_input

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

        if self.noise_dist is 'gaussian':

            self.obs_rand_samples = tf.random_normal(
                shape=[self.num_samples_ph, self.num_time_pts, self.dim_obs],
                mean=0.0, stddev=1.0, dtype=self.dtype,
                name='obs_rand_samples')

            def lds_update(outputs, inputs):

                [z_val, y_val] = outputs
                [rand_z, rand_y] = inputs

                z_val = tf.matmul(z_val, tf.transpose(self.A)) \
                    + tf.matmul(rand_z, tf.transpose(self.Q_sqrt))
                # expand dims to account for time and mc dims with mapping
                expanded_z = tf.expand_dims(
                    tf.expand_dims(z_val, axis=1), axis=1)
                y_val = tf.squeeze(
                    self._apply_mapping(expanded_z), axis=[1, 2]) \
                    + tf.multiply(rand_y, self.Rsqrt)

                return [z_val, y_val]

            # calculate samples for first time point
            z0 = self.z0_mean \
                + tf.matmul(self.latent_rand_samples[:, 0, :],
                            tf.transpose(self.Q0_sqrt))
            # expand dims to account for time and mc dims when applying mapping
            expanded_z0 = tf.expand_dims(tf.expand_dims(z0, axis=1), axis=1)
            y0 = tf.squeeze(self._apply_mapping(expanded_z0), axis=[1, 2]) \
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

        elif self.noise_dist is 'poisson':

            def lds_update(outputs, inputs):

                [z_val, y_val] = outputs
                rand_z = inputs

                z_val = tf.matmul(z_val, tf.transpose(self.A)) \
                        + tf.matmul(rand_z, tf.transpose(self.Q_sqrt))
                # expand dims to account for time and mc dims with mapping
                expanded_z = tf.expand_dims(tf.expand_dims(z_val, axis=1),
                                            axis=1)
                y_val = tf.squeeze(self._apply_mapping(expanded_z),
                                   axis=[1, 2])

                return [z_val, y_val]

            # calculate samples for first time point
            z0 = self.z0_mean \
                 + tf.matmul(self.latent_rand_samples[:, 0, :],
                             tf.transpose(self.Q0_sqrt))
            # expand dims to account for time and mc dims when applying mapping
            expanded_z0 = tf.expand_dims(tf.expand_dims(z0, axis=1), axis=1)
            y0 = tf.squeeze(self._apply_mapping(expanded_z0), axis=[1, 2])

            # scan over time points, not samples
            rand_ph_shuff = tf.transpose(
                self.latent_rand_samples[:, 1:, :], perm=[1, 0, 2])
            samples = tf.scan(
                fn=lds_update,
                elems=rand_ph_shuff,
                initializer=[z0, y0])

            z_samples_unshuff = tf.transpose(samples[0], perm=[1, 0, 2])
            y_samples_unshuff = tf.transpose(samples[1], perm=[1, 0, 2])

            self.z_samples_prior = tf.concat(
                [tf.expand_dims(z0, axis=1), z_samples_unshuff], axis=1)
            y_samples_prior = tf.concat(
                [tf.expand_dims(y0, axis=1), y_samples_unshuff], axis=1)
            self.y_samples_prior = tf.squeeze(tf.random_poisson(
                lam=y_samples_prior, shape=[1], dtype=self.dtype), axis=0)

    def log_density(self, y, z):
        """
        Evaluate log density for generative model, defined as
        p(y, z) = p(y | z) p(z)
        where
        p(z) = \prod_t p(z_t), z_t ~ N(A z_{t-1}, Q)
        p(y | z) = \prod_t p(y_t | z_t), y_t | z_t ~ N(C z_t + d, R)

        Args:
            y (batch_size x num_mc_samples x num_time_pts x dim_obs tf.Tensor)
            z (batch_size x num_mc_samples x num_time_pts x dim_latent
                tf.Tensor)

        Returns:
            float: log density over y and z, averaged over minibatch samples
                and monte carlo samples

        """

        # likelihood
        with tf.variable_scope('likelihood'):
            self.log_density_y = self._log_density_likelihood(y)

        # prior
        with tf.variable_scope('prior'):
            self.log_density_z = self._log_density_prior(z)

        return self.log_density_y + self.log_density_z

    def _log_density_likelihood(self, y):

        if self.noise_dist is 'gaussian':
            # expand observation dims over mc samples
            res_y = tf.expand_dims(self.obs_ph, axis=1) - y

            # average over batch and mc sample dimensions
            res_y_Rinv_res_y = tf.reduce_mean(
                tf.multiply(tf.square(res_y), self.Rinv), axis=[0, 1])

            # sum over time and observation dimensions
            test_like = tf.reduce_sum(res_y_Rinv_res_y)
            tf.summary.scalar('log_joint_like', -0.5 * test_like)

            # total term for likelihood
            log_density_y = -0.5 * (test_like
                 + self.num_time_pts * tf.reduce_sum(tf.log(self.R))
                 + self.num_time_pts * self.dim_obs * tf.log(2.0 * np.pi))

        elif self.noise_dist is 'poisson':

            # expand observation dims over mc samples
            obs_y = tf.expand_dims(self.obs_ph, axis=1)

            # average over batch and mc sample dimensions
            log_density_ya = tf.reduce_mean(
                tf.multiply(obs_y, tf.log(y)) - y - tf.lgamma(1 + obs_y),
                axis=[0, 1])

            # sum over time and observation dimensions
            log_density_y = tf.reduce_sum(log_density_ya)
            tf.summary.scalar('log_joint_like', log_density_y)

        else:
            raise ValueError

        return log_density_y

    def _log_density_prior(self, z):
        res_z0 = z[:, :, 0, :] - self.z0_mean
        res_z = z[:, :, 1:, :] - tf.tensordot(
            z[:, :, :-1, :], tf.transpose(self.A), axes=[[3], [0]])

        # average over batch and mc sample dimensions
        res_z_Qinv_res_z = tf.reduce_mean(tf.multiply(
            tf.tensordot(res_z, self.Qinv, axes=[[3], [0]]), res_z),
            axis=[0, 1])
        res_z0_Q0inv_res_z0 = tf.reduce_mean(tf.multiply(
            tf.tensordot(res_z0, self.Q0inv, axes=[[2], [0]]), res_z0),
            axis=[0, 1])

        # sum over time and latent dimensions
        test_prior = tf.reduce_sum(res_z_Qinv_res_z)
        test_prior0 = tf.reduce_sum(res_z0_Q0inv_res_z0)
        tf.summary.scalar('log_joint_prior', -0.5 * test_prior)
        tf.summary.scalar('log_joint_prior0', -0.5 * test_prior0)

        # total term for prior
        log_density_z = -0.5 * (test_prior + test_prior0
             + (self.num_time_pts - 1) * tf.log(tf.matrix_determinant(self.Q))
             + tf.log(tf.matrix_determinant(self.Q0))
             + self.num_time_pts * self.dim_latent * tf.log(2.0 * np.pi))

        return log_density_z

    def sample(self, sess, num_samples=1, seed=None):
        """
        Generate samples from the model

        Args:
            sess (tf.Session object)
            num_samples (int, optional)
            seed (int, optional)

        Returns:
            num_time_pts x dim_obs x num_samples numpy array:
                sample observations y
            num_time_pts x dim_latent x num_samples numpy array:
                sample latent states z

        """

        if seed is not None:
            tf.set_random_seed(seed)

        [y, z] = sess.run(
            [self.y_samples_prior, self.z_samples_prior],
            feed_dict={self.num_samples_ph: num_samples})

        return y, z

    def get_params(self, sess):
        """Get parameters of generative model"""

        if self.noise_dist is 'gaussian':
            A, layer_weights, Rsqrt, z0_mean, Q, Q0 = sess.run(
                [self.A, self.layers[0].weights, self.Rsqrt, self.z0_mean,
                 self.Q, self.Q0])

            param_dict = {
                'A': A, 'C': layer_weights[0], 'd': layer_weights[1],
                'R': np.square(Rsqrt), 'z0_mean': z0_mean, 'Q': Q, 'Q0': Q0}
        elif self.noise_dist is 'poisson':
            A, C, d, z0_mean, Q, Q0 = sess.run(
                [self.A, self.layers[0].weights, self.z0_mean, self.Q,
                 self.Q0])

            param_dict = {
                'A': A, 'C': C, 'd': d, 'z0_mean': z0_mean, 'Q': Q, 'Q0': Q0}
        else:
            raise ValueError

        return param_dict


class LDSCoupled(LDS):
    """
    Linear dynamical system model with Gaussian observations; model parameters
    are coupled to parameters of approximate posterior SmoothingLDSCoupled
    through the use of the LDSCoupled Model class
    """

    def __init__(
            self, dim_obs=None, dim_latent=None, post_z_samples=None,
            num_time_pts=None, gen_params=None, noise_dist='gaussian'):
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

        super().__init__(
            dim_obs=dim_obs, dim_latent=dim_latent, gen_params=gen_params,
            post_z_samples=post_z_samples, num_time_pts=num_time_pts,
            noise_dist=noise_dist)

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
            self._initialize_noise_dist_vars()
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


class FLDS(LDS):
    """
    Linear dynamical system model with neural network transformation to output
    and Gaussian observations
    """

    _layer_defaults = {
        'units': 30,
        'activation': 'tanh',
        'kernel_initializer': 'trunc_normal',
        'kernel_regularizer': None,
        'bias_initializer': 'zeros',
        'bias_regularizer': None}

    def __init__(
            self, dim_obs=None, dim_latent=None, post_z_samples=None,
            num_time_pts=None, gen_params=None, nn_params=None,
            noise_dist='gaussian'):
        """
        Generative model is defined as
            z_t = A z_{t-1} + \eps
            y_t = f(z_t) + \eta
        where f() is parameterized with a neural network

        Args:
            num_time_pts (int): number of time points per observation of the
                dynamical sequence
            gen_params (dict): dictionary of generative params for initializing
                model
            nn_params (list of dicts): each dict specifies parameters of a layer
                for building the neural network

        """

        super().__init__(
            dim_obs=dim_obs, dim_latent=dim_latent, num_time_pts=num_time_pts,
            post_z_samples=post_z_samples, gen_params=gen_params,
            noise_dist=noise_dist)

        if nn_params is None:
            nn_params = [{}]
        self._parse_nn_options(nn_params)

    def _parse_nn_options(self, nn_options):
        """Specify architecture of decoding network and set defaults"""

        self.nn_params = []
        for layer_num, layer_options in enumerate(nn_options):
            layer_params = dict(self._layer_defaults)
            layer_params['name'] = str('layer_%02i' % layer_num)
            # replace defaults with user-supplied options
            for key, value in layer_options.items():
                layer_params[key] = value
            # translate from strings to tf operations
            for key, value in layer_params.items():
                if value is not None:
                    # otherwise use default
                    if key is 'activation':
                        if value is 'identity':
                            value = None
                        elif value is 'relu':
                            value = tf.nn.relu
                        elif value is 'softmax':
                            value = tf.nn.softmax
                        elif value is 'softplus':
                            value = tf.nn.softplus
                        elif value is 'sigmoid':
                            value = tf.nn.sigmoid
                        elif value is 'tanh':
                            value = tf.nn.tanh
                        else:
                            raise ValueError(
                                '"%s" is not a valid string for specifying '
                                'the activation function' % value)
                    elif key is 'kernel_initializer' \
                            or key is 'bias_initializer':
                        if value is 'normal':
                            value = tf.initializers.random_normal(
                                mean=0.0, stddev=0.1, dtype=self.dtype)
                        elif value is 'trunc_normal':
                            value = tf.initializers.truncated_normal(
                                mean=0.0, stddev=0.1, dtype=self.dtype)
                        elif value is 'zeros':
                            value = tf.initializers.zeros(dtype=self.dtype)
                        else:
                            raise ValueError(
                                '"%s" is not a valid string for specifying '
                                'a variable initializer' % value)
                # reassign (possibly new) value to key
                layer_params[key] = value
            # make sure output is correct
            if layer_num == len(nn_options) - 1:
                layer_params['units'] = self.dim_obs
            self.nn_params.append(dict(layer_params))


class FLDSCoupled(LDSCoupled, FLDS):

    def __init__(
            self, dim_obs=None, dim_latent=None, post_z_samples=None,
            num_time_pts=None, gen_params=None, nn_params=None,
            noise_dist='gaussian'):
        """
        Generative model is defined as
            z_t = A z_{t-1} + \eps
            y_t = f(z_t) + \eta

        Args:
            num_time_pts (int): number of time points per observation of the
                dynamical sequence
            gen_params (dict): dictionary of generative params for initializing
                model
            nn_params

        """

        FLDS.__init__(
            self, dim_obs=dim_obs, dim_latent=dim_latent,
            gen_params=gen_params, num_time_pts=num_time_pts,
            post_z_samples=post_z_samples, nn_params=nn_params,
            noise_dist=noise_dist,)
