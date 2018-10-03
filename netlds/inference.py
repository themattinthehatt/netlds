"""InferenceNetwork class for approximate posteriors"""

import numpy as np
import tensorflow as tf
from netlds.network import Network
from netlds.chol_utils import blk_tridiag_chol, blk_chol_inv, \
    blk_chol_inv_multi


class InferenceNetwork(object):
    """Base class for inference networks"""

    # use same data type throughout graph construction
    dtype = tf.float32

    def __init__(
            self, dim_input=None, dim_latent=None, num_mc_samples=1):
        """
        Set base class attributes

        Args:
            dim_input (int): dimension of inputs that are transformed by a
                neural network to provide data-point specific distributional
                parameters of the approximate posterior
            dim_latent (int): dimension of latent state

        """

        # set basic dims
        self.dim_input = dim_input
        self.dim_latent = dim_latent
        self.num_mc_samples = num_mc_samples

    def build_graph(self, *args, **kwargs):
        """Build tensorflow computation graph for inference network"""
        raise NotImplementedError

    def entropy(self):
        """Entropy of approximate posterior"""
        raise NotImplementedError

    def sample(self, sess, observations, seed=None):
        """Draw samples from approximate posterior"""
        raise NotImplementedError


class SmoothingLDS(InferenceNetwork):
    """
    Approximate posterior is modeled as a Gaussian distribution with a
    structure mirroring that from a linear dynamical system
    """

    def __init__(
            self, dim_input=None, dim_latent=None, num_mc_samples=1,
            num_time_pts=None, nn_params=None):

        super().__init__(
            dim_input=dim_input, dim_latent=dim_latent,
            num_mc_samples=num_mc_samples)

        self.num_time_pts = num_time_pts
        self.nn_params = nn_params

        # initialize networks

        # default inference network
        if self.nn_params is None:
            layer_params = {
                'units': 30,
                'activation': 'tanh',
                'kernel_initializer': 'trunc_normal',
                'kernel_regularizer': None,
                'bias_initializer': 'zeros',
                'bias_regularizer': None}
            self.nn_params = [layer_params, layer_params]
        self.network = Network(
            output_dim=30, nn_params=self.nn_params)

        # inference network -> means
        layer_z_mean_params = [{
            'units': self.dim_latent,
            'activation': 'identity',
            'kernel_initializer': 'trunc_normal',
            'kernel_regularizer': None,
            'bias_initializer': 'zeros',
            'bias_regularizer': None,
            'name': 'z_means'}]
        self.layer_z_mean = Network(
            output_dim=self.dim_latent, nn_params=layer_z_mean_params)

        # inference network -> stddevs
        layer_z_var_params = [{
            'units': self.dim_latent * self.dim_latent,
            'activation': 'identity',
            'kernel_initializer': 'trunc_normal',
            'kernel_regularizer': None,
            'bias_initializer': 'zeros',
            'bias_regularizer': None,
            'name': 'z_stddev'}]
        self.layer_z_vars = Network(
            output_dim=self.dim_latent * self.dim_latent,
            nn_params=layer_z_var_params)

    def build_graph(self, param_dict):
        """Build tensorflow computation graph for inference network"""

        # set prior variables generated elsewhere
        self.z0_mean = param_dict['z0_mean']
        self.A = param_dict['A']
        self.Q0_sqrt = param_dict['Q0_sqrt']
        self.Q_sqrt = param_dict['Q_sqrt']
        self.Q0 = param_dict['Q0']
        self.Q = param_dict['Q']
        self.Q0_inv = param_dict['Q0_inv']
        self.Q_inv = param_dict['Q_inv']

        # construct data pipeline
        with tf.variable_scope('inference_input'):
            self._initialize_inference_input()

        with tf.variable_scope('inference_mlp'):
            self._build_inference_mlp()

        with tf.variable_scope('precision_matrix'):
            self._build_precision_matrix()

        with tf.variable_scope('posterior_mean'):
            self._build_posterior_mean()

        with tf.variable_scope('posterior_samples'):
            self._build_posterior_samples()

    def _initialize_inference_input(self):

        self.input_ph = tf.placeholder(
            dtype=self.dtype,
            shape=[None, self.num_time_pts, self.dim_input],
            name='obs_in_ph')
        self.samples_z = tf.random_normal(
            shape=[tf.shape(self.input_ph)[0],
                   self.num_time_pts, self.dim_latent, self.num_mc_samples],
            mean=0.0, stddev=1.0, dtype=self.dtype, name='samples_z')

    def _build_inference_mlp(self):

        self.network.build_graph()
        self.layer_z_mean.build_graph()
        self.layer_z_vars.build_graph()

        # compute layer outputs from inference network input
        self.hidden_act = self.network.apply_network(self.input_ph)

        # get data-dependent mean
        self.m_psi = self.layer_z_mean.apply_network(self.hidden_act)

        # get sqrt of inverse of data-dependent covariances
        r_psi_sqrt = self.layer_z_vars.apply_network(self.hidden_act)
        self.r_psi_sqrt = tf.reshape(
            r_psi_sqrt,
            [-1, self.num_time_pts, self.dim_latent, self.dim_latent])

    def _build_precision_matrix(self):
        # get inverse of data-dependent covariances
        self.c_psi_inv = tf.matmul(
            self.r_psi_sqrt,
            tf.transpose(self.r_psi_sqrt, perm=[0, 1, 3, 2]),
            name='precision_diag_data_dep')

        self.AQ0_invA_Q_inv = tf.matmul(
            tf.matmul(self.A, self.Q0_inv), self.A, transpose_b=True) \
            + self.Q_inv
        self.AQ_invA_Q_inv = tf.matmul(
            tf.matmul(self.A, self.Q_inv), self.A, transpose_b=True) \
            + self.Q_inv
        self.AQ0_inv = tf.matmul(-self.A, self.Q0_inv)
        self.AQ_inv = tf.matmul(-self.A, self.Q_inv)

        # put together components of precision matrix Sinv in tensor of
        # shape [batch_size, num_time_pts, dim_latent, dim_latent]
        Sinv_diag = tf.tile(
            tf.expand_dims(self.AQ_invA_Q_inv, 0),
            [self.num_time_pts - 2, 1, 1])
        Sinv_diag = tf.concat(
            [tf.expand_dims(self.Q0_inv, 0),
             tf.expand_dims(self.AQ0_invA_Q_inv, 0),
             Sinv_diag], axis=0, name='precision_diag_static')
        self.Sinv_diag = tf.add(Sinv_diag, self.c_psi_inv,
                                name='precision_diag')

        Sinv_ldiag = tf.tile(
            tf.expand_dims(self.AQ_inv, 0),
            [self.num_time_pts - 2, 1, 1], name='precision_lower_diag')
        Sinv_ldiag0 = tf.concat(
            [tf.expand_dims(self.AQ0_inv, 0), Sinv_ldiag], axis=0)

        # we now have Sinv (represented as diagonal and off-diagonal
        # blocks); to sample from the posterior we need the square root
        # of the inverse of Sinv; fortunately this is fast given the
        # tridiagonal block structure of Sinv. First we'll compute the
        # Cholesky decomposition of Sinv, then calculate the inverse using
        # that decomposition

        # get cholesky decomposition for each element in batch
        def scan_chol(_, inputs):
            """inputs refer to diagonal blocks, outputs the L/U matrices"""
            chol_decomp_Sinv = blk_tridiag_chol(inputs, Sinv_ldiag0)
            return chol_decomp_Sinv

        self.chol_decomp_Sinv = tf.scan(
            fn=scan_chol, elems=self.Sinv_diag,
            initializer=[Sinv_diag, Sinv_ldiag0],  # throwaway to get scan
            name='precision_chol_decomp')  # to behave

    def _build_posterior_mean(self):

        ia = tf.reduce_sum(
            tf.multiply(self.c_psi_inv,
                        tf.expand_dims(self.m_psi, axis=2)),
            axis=3)

        # ia now S x T x dim_latent

        # get posterior means for each element in batch
        def scan_chol_inv(_, inputs):
            """inputs refer to L/U matrices, outputs to means"""
            [chol_decomp_Sinv_0, chol_decomp_Sinv_1, ia] = inputs
            # mult by R
            ib = blk_chol_inv(
                chol_decomp_Sinv_0, chol_decomp_Sinv_1, ia,
                lower=True, transpose=False)
            post_z_means = blk_chol_inv(
                chol_decomp_Sinv_0, chol_decomp_Sinv_1, ib,
                lower=False, transpose=True)

            return post_z_means

        self.post_z_means = tf.scan(
            fn=scan_chol_inv,
            elems=[self.chol_decomp_Sinv[0], self.chol_decomp_Sinv[1], ia],
            initializer=ia[0])  # throwaway to get scan to behave

    def _build_posterior_samples(self):

        # get posterior sample(s) for each element in batch
        def scan_chol_half_inv(_, inputs):
            """
            inputs refer to L/U matrices and N(0, 1) samples, outputs to
            N(\mu, \sigma^2) samples
            """
            [chol_decomp_Sinv_0, chol_decomp_Sinv_1, samples] = inputs
            rands = blk_chol_inv_multi(
                chol_decomp_Sinv_0, chol_decomp_Sinv_1, samples,
                lower=False, transpose=True)

            return rands

        # note:
        #   B - batch_size; T - num_time_pts; D - dim_latent; S - mc samples
        #   self.chol_decomp_Sinv[0]: B x T x D x D
        #   self.chol_decomp_Sinv[1]: B x (T-1) x D x D
        #   self.samples_z: B x T x D x S
        rands = tf.scan(
            fn=scan_chol_half_inv,
            elems=[self.chol_decomp_Sinv[0], self.chol_decomp_Sinv[1],
                   self.samples_z],
            initializer=self.samples_z[0])  # throwaway for scan to behave

        # rands is currently
        # batch_size x num_time_pts x dim_latent x num_mc_samples
        # reshape to
        # batch_size x num_mc_samples x num_time_pts x dim_latent
        # to push through generative model layers
        rands = tf.transpose(rands, perm=[0, 3, 1, 2])

        # tf addition op will broadcast extra 'num_mc_samples' dims
        self.post_z_samples = tf.expand_dims(self.post_z_means, axis=1) + rands

    def entropy(self):
        """Entropy of approximate posterior"""

        # determinant of the covariance is the square of the determinant of the
        # cholesky factor; determinant of the cholesky factor is the product of
        # the diagonal elements of the block-diagonal

        # mean over batch dimension, sum over time dimension
        diags = tf.matrix_diag_part(self.chol_decomp_Sinv[0])
        ln_det = -2.0 * tf.reduce_sum(
            tf.reduce_mean(tf.log(diags), axis=0))

        entropy = ln_det / 2.0 + self.dim_latent * self.num_time_pts / 2.0 * (
                    1.0 + np.log(2.0 * np.pi))

        return entropy

    def sample(self, sess, observations, seed=None):
        """
        Draw samples from approximate posterior

        Args:
            sess (tf.Session object)
            observations (batch_size x num_time_pts x num_inputs numpy array)
            seed (int, optional)

        Returns:
            batch_size x num_mc_samples x num_time_pts x dim_latent numpy array

        """

        if seed is not None:
            tf.set_random_seed(seed)

        return sess.run(
            self.post_z_samples,
            feed_dict={self.input_ph: observations})

    def get_params(self, sess):
        """Get parameters of generative model"""

        A, z0_mean, Q_sqrt, Q, Q0_sqrt, Q0 = sess.run(
            [self.A, self.z0_mean, self.Q_sqrt, self.Q, self.Q0_sqrt, self.Q0])

        param_dict = {'A': A, 'z0_mean': z0_mean, 'Q': Q, 'Q0': Q0,
                      'Q_sqrt': Q_sqrt, 'Q0_sqrt': Q0_sqrt}

        return param_dict

    def get_posterior_means(self, sess, observations):
        """Get posterior means conditioned on observations"""

        feed_dict = {self.input_ph: observations}

        return sess.run(self.post_z_means, feed_dict=feed_dict)


class MeanFieldGaussian(InferenceNetwork):
    """
    Approximate posterior is modeled as a fully factorized Gaussian across time
    and latent dimensions, so that for
    x = [x_1, ..., x_T]
    and
    x_i = [x_1^i, ..., x_T^i]

    x ~ \prod_{t=1}^T \prod_{i=1}^dim_latent N( mu_t^i(y_t), sigma_t^i(y_t) )

    Each covariance sigma_t is a diagonal [dim_latent x dim_latent] covariance
    matrix.
    """

    def __init__(
            self, dim_input=None, dim_latent=None, num_mc_samples=1,
            num_time_pts=None, nn_params=None):

        super().__init__(
            dim_input=dim_input, dim_latent=dim_latent,
            num_mc_samples=num_mc_samples)

        self.num_time_pts = num_time_pts
        self.nn_params = nn_params

        # initialize networks

        # default inference network
        if self.nn_params is None:
            layer_params = {
                'units': 30,
                'activation': 'tanh',
                'kernel_initializer': 'trunc_normal',
                'kernel_regularizer': None,
                'bias_initializer': 'zeros',
                'bias_regularizer': None}
            self.nn_params = [layer_params, layer_params]
        self.network = Network(
            output_dim=30, nn_params=self.nn_params)

        # inference network -> means
        layer_z_mean_params = [{
            'units': self.dim_latent,
            'activation': 'identity',
            'kernel_initializer': 'trunc_normal',
            'kernel_regularizer': None,
            'bias_initializer': 'zeros',
            'bias_regularizer': None,
            'name': 'z_means'}]
        self.layer_z_mean = Network(
            output_dim=self.dim_latent, nn_params=layer_z_mean_params)

        # inference network -> log vars
        layer_z_var_params = [{
            'units': self.dim_latent,
            'activation': 'identity',
            'kernel_initializer': 'trunc_normal',
            'kernel_regularizer': None,
            'bias_initializer': 'zeros',
            'bias_regularizer': None,
            'name': 'z_log_vars'}]
        self.layer_z_log_vars = Network(
            output_dim=self.dim_latent, nn_params=layer_z_var_params)

    def build_graph(self, *args):
        """Build tensorflow computation graph for inference network"""

        # construct data pipeline
        with tf.variable_scope('inference_input'):
            self._initialize_inference_input()

        with tf.variable_scope('inference_mlp'):
            self._build_inference_mlp()

        with tf.variable_scope('posterior_samples'):
            self._build_posterior_samples()

    def _initialize_inference_input(self):

        self.input_ph = tf.placeholder(
            dtype=self.dtype,
            shape=[None, self.num_time_pts, self.dim_input],
            name='obs_in_ph')
        self.samples_z = tf.random_normal(
            shape=[tf.shape(self.input_ph)[0],
                   self.num_mc_samples, self.num_time_pts, self.dim_latent],
            mean=0.0, stddev=1.0, dtype=self.dtype, name='samples_z')

    def _build_inference_mlp(self):

        self.network.build_graph()
        self.layer_z_mean.build_graph()
        self.layer_z_log_vars.build_graph()

        # compute layer outputs from inference network input
        self.hidden_act = self.network.apply_network(self.input_ph)

        # get data-dependent mean
        self.post_z_means = self.layer_z_mean.apply_network(self.hidden_act)

        # get sqrt of inverse of data-dependent covariances
        self.post_z_log_vars = self.layer_z_log_vars.apply_network(
            self.hidden_act)

    def _build_posterior_samples(self):

        # keep log-vars in reasonable range
        #temp0 = 5.0 * tf.tanh(self.post_z_log_vars / 5.0)
        temp0 = self.post_z_log_vars
        temp = tf.exp(tf.expand_dims(temp0, axis=1))
        rands = tf.multiply(tf.sqrt(temp), self.samples_z)

        self.post_z_samples = tf.expand_dims(self.post_z_means, axis=1) + rands

    def entropy(self):
        """Entropy of approximate posterior"""

        ln_det = tf.reduce_sum(tf.reduce_mean(self.post_z_log_vars, axis=0))

        entropy = ln_det / 2.0 + self.dim_latent * self.num_time_pts / 2.0 * (
                    1.0 + np.log(2.0 * np.pi))

        return entropy

    def sample(self, sess, observations, seed=None):
        """
        Draw samples from approximate posterior

        Args:
            sess (tf.Session object)
            observations (batch_size x num_time_pts x num_inputs numpy array)
            seed (int, optional)

        Returns:
            batch_size x num_mc_samples x num_time_pts x dim_latent numpy array

        """

        if seed is not None:
            tf.set_random_seed(seed)

        return sess.run(
            self.post_z_samples,
            feed_dict={self.input_ph: observations})

    def get_posterior_means(self, sess, observations):
        """Get posterior means conditioned on observations"""

        feed_dict = {self.input_ph: observations}

        return sess.run(self.post_z_means, feed_dict=feed_dict)


class MeanFieldGaussianTemporal(InferenceNetwork):
    """
    Approximate posterior is modeled as a fully factorized Gaussian in time, so
    that for x = [x_1, ..., x_T]

    x ~ \prod_{t=1}^T N( mu_t(y_t), sigma_t(y_t) )

    Each covariance sigma_t is a full [dim_latent x dim_latent] covariance
    matrix.
    """

    def __init__(
            self, dim_input=None, dim_latent=None, num_mc_samples=1,
            num_time_pts=None, nn_params=None):

        super().__init__(
            dim_input=dim_input, dim_latent=dim_latent,
            num_mc_samples=num_mc_samples)

        self.num_time_pts = num_time_pts
        self.nn_params = nn_params

        # initialize networks

        # default inference network
        if self.nn_params is None:
            layer_params = {
                'units': 30,
                'activation': 'tanh',
                'kernel_initializer': 'trunc_normal',
                'kernel_regularizer': None,
                'bias_initializer': 'zeros',
                'bias_regularizer': None}
            self.nn_params = [layer_params, layer_params]
        self.network = Network(
            output_dim=30, nn_params=self.nn_params)

        # inference network -> means
        layer_z_mean_params = [{
            'units': self.dim_latent,
            'activation': 'identity',
            'kernel_initializer': 'trunc_normal',
            'kernel_regularizer': None,
            'bias_initializer': 'zeros',
            'bias_regularizer': None,
            'name': 'z_means'}]
        self.layer_z_mean = Network(
            output_dim=self.dim_latent, nn_params=layer_z_mean_params)

        # inference network -> log vars
        layer_z_var_params = [{
            'units': self.dim_latent * self.dim_latent,
            'activation': 'identity',
            'kernel_initializer': 'trunc_normal',
            'kernel_regularizer': None,
            'bias_initializer': 'zeros',
            'bias_regularizer': None,
            'name': 'z_stddev'}]
        self.layer_z_vars = Network(
            output_dim=self.dim_latent * self.dim_latent,
            nn_params=layer_z_var_params)

    def build_graph(self, *args):
        """Build tensorflow computation graph for inference network"""

        # construct data pipeline
        with tf.variable_scope('inference_input'):
            self._initialize_inference_input()

        with tf.variable_scope('inference_mlp'):
            self._build_inference_mlp()

        with tf.variable_scope('posterior_samples'):
            self._build_posterior_samples()

    def _initialize_inference_input(self):

        self.input_ph = tf.placeholder(
            dtype=self.dtype,
            shape=[None, self.num_time_pts, self.dim_input],
            name='obs_in_ph')
        self.samples_z = tf.random_normal(
            shape=[tf.shape(self.input_ph)[0],
                   self.num_time_pts, self.dim_latent, self.num_mc_samples],
            mean=0.0, stddev=1.0, dtype=self.dtype, name='samples_z')

    def _build_inference_mlp(self):

        self.network.build_graph()
        self.layer_z_mean.build_graph()
        self.layer_z_vars.build_graph()

        # compute layer outputs from inference network input
        self.hidden_act = self.network.apply_network(self.input_ph)

        # get data-dependent mean
        self.post_z_means = self.layer_z_mean.apply_network(self.hidden_act)

        # get sqrt of inverse of data-dependent covariances
        r_psi_sqrt = self.layer_z_vars.apply_network(self.hidden_act)
        self.r_psi_sqrt = tf.reshape(
            r_psi_sqrt,
            [-1, self.num_time_pts, self.dim_latent, self.dim_latent])

    def _build_posterior_samples(self):

        def sample_batch(outputs, inputs):
            # samples: num_time_pts x dim_latent x num_mc_samples
            # cov_chol: num_time_pts x dim_latent x dim_latent
            z_samples, cov_chol = inputs
            # scan over time dimension
            samples = tf.scan(
                fn=sample_time,
                elems=[z_samples, cov_chol],
                initializer=z_samples[0])  # throwaway for scan to behave
            return samples

        def sample_time(outputs, inputs):
            # samples: dim_latent x num_mc_samples
            # cov_chol: dim_latent x dim_latent
            z_samples, cov_chol = inputs
            # scan over time dimension
            samples = tf.matmul(cov_chol, z_samples)
            return samples

        # scan over batch dimension
        # returns num_batches x num_time_pts x dim_latent x num_mc_samples
        rands_shuff = tf.scan(
            fn=sample_batch,
            elems=[self.samples_z, self.r_psi_sqrt],
            initializer=self.samples_z[0])  # throwaway for scan to behave

        rands = tf.transpose(rands_shuff, perm=[0, 3, 1, 2])

        self.post_z_samples = tf.expand_dims(self.post_z_means, axis=1) + rands

    def entropy(self):
        """Entropy of approximate posterior"""

        # determinant of the covariance is the square of the determinant of the
        # cholesky factor; determinant of the cholesky factor is the product of
        # the diagonal elements of the block-diagonal
        def det_loop_batch(outputs, inputs):
            # inputs: num_time_dims x dim_latent x dim_latent
            # now scan over over time
            out = tf.scan(
                fn=det_loop_time, elems=inputs, initializer=0.0)
            return out

        def det_loop_time(outputs, inputs):
            # inputs is dim_latent x dim_latent matrix
            return tf.matrix_determinant(tf.matmul(
                inputs, inputs, transpose_b=True))

        # for each batch, scan over time, calculating det of time blocks; det
        # of full matrix is product of determinants over blocks
        self.dets = dets = tf.scan(
            fn=det_loop_batch, elems=self.r_psi_sqrt,
            initializer=self.r_psi_sqrt[0, :, 0, 0], infer_shape=False)
        # shape of initializer must match shape of returned element; in this
        # case a single scalar (the determinant) for each time point

        # mean over batch dimension, sum over time dimension
        ln_det = 2.0 * tf.reduce_sum(tf.reduce_mean(tf.log(dets), axis=0))

        entropy = ln_det / 2.0 + self.dim_latent * self.num_time_pts / 2.0 * (
                    1.0 + np.log(2.0 * np.pi))

        return entropy

    def sample(self, sess, observations, seed=None):
        """
        Draw samples from approximate posterior

        Args:
            sess (tf.Session object)
            observations (batch_size x num_time_pts x num_inputs numpy array)
            seed (int, optional)

        Returns:
            batch_size x num_mc_samples x num_time_pts x dim_latent numpy array

        """

        if seed is not None:
            tf.set_random_seed(seed)

        return sess.run(
            self.post_z_samples,
            feed_dict={self.input_ph: observations})

    def get_posterior_means(self, sess, observations):
        """Get posterior means conditioned on observations"""

        feed_dict = {self.input_ph: observations}

        return sess.run(self.post_z_means, feed_dict=feed_dict)
