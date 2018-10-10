"""GenerativeModel class for ... generative models"""

import numpy as np
import tensorflow as tf
from netlds.network import Network


class GenerativeModel(object):
    """Base class for generative models"""

    # use same data type throughout graph construction
    dtype = tf.float32

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

    def build_graph(self, *args, **kwargs):
        """Build tensorflow computation graph for generative model"""
        raise NotImplementedError

    def log_density(self, y, z):
        """Evaluate log density of generative model"""
        raise NotImplementedError

    def sample(self, sess, num_samples=1, seed=None):
        """Draw samples from model"""
        raise NotImplementedError


class NetFLDS(GenerativeModel):
    """
    Generative model is defined as

    z_t ~ N(A z_{t-1}, Q)
    E[y_t^i] = f(z_t^i)

    for each population i, where the z_t^i are non-overlapping subsets of z_t
    """

    def __init__(
            self, dim_obs=None, dim_latent=None, linear_predictors=None,
            num_time_pts=None, gen_params=None, noise_dist='gaussian',
            nn_params=None, post_z_samples=None):
        """
        Args:
            dim_obs (list): observation dimension for each population
            dim_latent (list): latent dimension for each population
            linear_predictors (dict):
                'dim_predictors' (list): dimension for each set of linear
                    predictors
                'predictor_indx' (list of lists): each element of the list
                    contains the indices of the predictors in the
                    `dim_predictors` list used by the corresponding population
                'predictor_params' (list of lists): each element contains
                    params for initializing the linear mapping of each pop/pred
                    combo; should match 'predictor_indx'
            num_time_pts (int): number of time points per observation of the
                dynamical sequence
            gen_params (dict): dictionary of generative params for initializing
                model
            noise_dist (str): 'gaussian' | 'poisson'
            nn_params (list): dictionaries for building each layer of the
                mapping from the latent space to observations; the same
                network architecture is used for each population
            post_z_samples (batch_size x num_mc_samples x num_time_pts x
                dim_latent tf.Tensor): samples from the (appx) posterior of the
                latent states

        """

        GenerativeModel.__init__(self, post_z_samples=post_z_samples)
        self.dim_obs = dim_obs
        self.dim_latent = dim_latent
        if linear_predictors is None:
            self.dim_predictors = None
            self.predictor_indx = None
        else:
            self.dim_predictors = linear_predictors['dim_predictors']
            predictor_indx = linear_predictors['predictor_indx']
            if 'predictor_params' in linear_predictors:
                predictor_params = linear_predictors['predictor_params']
            else:
                predictor_params = None

        self.num_time_pts = num_time_pts
        if gen_params is None:
            self.gen_params = {}
        else:
            self.gen_params = gen_params

        # spiking nl
        self.noise_dist = noise_dist
        if noise_dist is 'gaussian':
            activation = 'linear'
        elif noise_dist is 'poisson':
            activation = 'softplus'
        else:
            raise ValueError

        if nn_params is None:
            # use Network defaults
            nn_params = [{}]
        nn_params[-1]['activation'] = activation

        # networks mapping latent states to obs for each population
        self.networks = []
        for _, pop_dim in enumerate(dim_obs):
            self.networks.append(
                Network(output_dim=pop_dim, nn_params=nn_params))

        # networks mapping linear predictors to obs for each population
        # accessed as self.networks_linear[pop][pred]
        self.networks_linear = [
            [None for _ in range(len(self.dim_predictors))]
            for _ in range(len(dim_obs))]
        self.predictor_indx = [
            [None for _ in range(len(self.dim_predictors))]
            for _ in range(len(dim_obs))]
        # only initialize networks if we have linear predictors
        if self.dim_predictors is not None:
            linear_nn_params = [{'activation': 'linear'}]
            for pop, pop_dim in enumerate(self.dim_obs):
                # for pred, pred_dim in enumerate(self.dim_predictors):
                #     if any(pred_indx == pred
                #            for pred_indx in predictor_indx[pop]):
                #         self.networks_linear[pop][pred] = Network(
                #             output_dim=pop_dim, nn_params=linear_nn_params)
                #         self.predictor_indx[pop][pred] = pred
                for indx, pred_indx in enumerate(predictor_indx[pop]):
                    self.predictor_indx[pop][pred_indx] = pred_indx
                    if predictor_params is not None \
                            and predictor_params[pop][indx] is not None:
                        pred_params = predictor_params[pop][indx]
                    else:
                        pred_params = linear_nn_params
                    self.networks_linear[pop][pred_indx] = Network(
                        output_dim=pop_dim, nn_params=pred_params)

        # initialize lists for other relevant variables
        self.linear_predictors_phs = []
        self.y_pred = []
        self.y_pred_ls = []  # latent space
        self.y_pred_lp = []  # linear predictors
        self.y_samples_prior = []
        self.latent_indxs = []
        if noise_dist is 'gaussian':
            self.R_sqrt = []
            self.R = []
            self.R_inv = []

    def build_graph(self, z_samples, param_dict):
        """
        Build tensorflow computation graph for generative model

        Args:
            z_samples (batch_size x num_mc_samples x num_time_pts x dim_latent
                tf.Tensor): samples of the latent states
            param_dict (dict): output of NetLDS.initialize_prior_vars() method

        """

        # set prior variables generated elsewhere
        self.z0_mean = param_dict['z0_mean']
        self.A = param_dict['A']
        self.Q0_sqrt = param_dict['Q0_sqrt']
        self.Q_sqrt = param_dict['Q_sqrt']
        self.Q0 = param_dict['Q0']
        self.Q = param_dict['Q']
        self.Q0_inv = param_dict['Q0_inv']
        self.Q_inv = param_dict['Q_inv']

        # initialize placeholders for linear predictors
        with tf.variable_scope('linear_predictors'):
            if self.dim_predictors is not None:
                for pred, dim_pred in enumerate(self.dim_predictors):
                    self.linear_predictors_phs.append(
                        tf.placeholder(
                            dtype=self.dtype,
                            shape=[None, self.num_time_pts, dim_pred],
                            name='linear_pred_ph_%02i' % pred))

        # keep track of which latent states belong to each population
        indx_start = 0
        for pop, pop_dim_latent in enumerate(self.dim_latent):
            with tf.variable_scope(str('population_%02i' % pop)):
                # initialize mapping from latent space to observations
                with tf.variable_scope('latent_space_mapping'):
                    self.networks[pop].build_graph()
                    indx_end = indx_start + pop_dim_latent
                    self.latent_indxs.append(
                        np.arange(indx_start, indx_end+1, dtype=np.int32))
                    self.y_pred_ls.append(self.networks[pop].apply_network(
                        z_samples[:, :, :, indx_start:indx_end]))
                    indx_start = indx_end
                # initialize mapping from linear predictors to observations
                with tf.variable_scope('regressor_mapping'):
                    if self.dim_predictors is not None:
                        # append new list for this population
                        self.y_pred_lp.append([])
                        for pred, pred_dim in enumerate(self.dim_predictors):
                            if self.predictor_indx[pop][pred] is not None:
                                self.networks_linear[pop][pred].build_graph()
                                net_out = self.networks_linear[pop][pred].\
                                    apply_network(
                                        self.linear_predictors_phs[pred])
                                # expand network output to match dims of
                                # samples from latent space:
                                # batch x num_samps x num_time_pts x dim_latent
                                self.y_pred_lp[-1].append(tf.expand_dims(
                                    net_out, axis=1))
                            # else:
                            #     self.y_pred_lp[-1].append(0.0)

                # add contributions from latent space and linear predictors
                if self.dim_predictors is not None:
                    self.y_pred.append(tf.add(self.y_pred_ls[-1], tf.add_n(
                        self.y_pred_lp[-1])))
                else:
                    self.y_pred.append(self.y_pred_ls[-1])
                self._initialize_noise_dist_vars(pop)

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
                shape=[1, sum(self.dim_latent)],
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
                initializer=0.5 * np.eye(sum(self.dim_latent),
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
                    sum(self.dim_latent),
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
                    sum(self.dim_latent),
                    dtype=self.dtype.as_numpy_dtype()),
                dtype=self.dtype)

        # diag = tf.constant(1.0 * np.eye(
        #     sum(self.dim_latent), dtype=self.dtype.as_numpy_dtype),
        #     name='small_const')
        diag = tf.constant(1e-6 * np.eye(
            sum(self.dim_latent), dtype=self.dtype.as_numpy_dtype),
            name='small_const')

        Q0 = tf.matmul(Q0_sqrt, Q0_sqrt, transpose_b=True, name='Q0') + diag
        Q = tf.matmul(Q_sqrt, Q_sqrt, transpose_b=True, name='Q') + diag
        Q0_inv = tf.matrix_inverse(Q0, name='Q0_inv')
        Q_inv = tf.matrix_inverse(Q, name='Q_inv')

        param_dict = {
            'z0_mean': z0_mean, 'A': A,
            'Q_sqrt': Q_sqrt, 'Q': Q, 'Q_inv': Q_inv,
            'Q0_sqrt': Q0_sqrt, 'Q0': Q0, 'Q0_inv': Q0_inv}

        return param_dict

    def _initialize_noise_dist_vars(self, pop):

        if self.noise_dist is 'gaussian':
            tr_norm_initializer = tf.initializers.truncated_normal(
                mean=0.0, stddev=0.1, dtype=self.dtype)
            zeros_initializer = tf.initializers.zeros(dtype=self.dtype)

            # square root of diagonal of observation covariance matrix
            # (assume diagonal)
            if 'R_sqrt' in self.gen_params:
                self.R_sqrt.append(tf.get_variable(
                    'R_sqrt',
                    initializer=self.gen_params['R_sqrt'][pop],
                    dtype=self.dtype))
            else:
                self.R_sqrt.append(tf.get_variable(
                    'R_sqrt',
                    shape=[1, self.dim_obs[pop]],
                    initializer=tr_norm_initializer,
                    dtype=self.dtype))
            self.R.append(tf.square(self.R_sqrt[pop], name='R'))
            self.R_inv.append(tf.divide(1.0, self.R[pop] + 1e-6, name='R_inv'))

    def _sample_yz(self):
        """
        Define branch of tensorflow computation graph for sampling from the
        prior
        """

        self.num_samples_ph = tf.placeholder(
            dtype=tf.int32, shape=None, name='num_samples_ph')

        self._sample_z()
        self._sample_y()

    def _sample_z(self):

        self.latent_rand_samples = tf.random_normal(
            shape=[self.num_samples_ph,
                   self.num_time_pts,
                   sum(self.dim_latent)],
            mean=0.0, stddev=1.0, dtype=self.dtype, name='latent_rand_samples')

        # get random samples from latent space
        def lds_update(outputs, inputs):
            z_val = outputs
            rand_z = inputs
            z_val = tf.matmul(z_val, tf.transpose(self.A)) \
                + tf.matmul(rand_z, tf.transpose(self.Q_sqrt))
            return z_val

        # calculate samples for first time point
        z0_samples = self.z0_mean \
            + tf.matmul(self.latent_rand_samples[:, 0, :],
                        tf.transpose(self.Q0_sqrt))

        # scan over time points, not samples
        rand_ph_shuff = tf.transpose(
            self.latent_rand_samples[:, 1:, :], perm=[1, 0, 2])
        z_samples = tf.scan(
            fn=lds_update,
            elems=rand_ph_shuff,
            initializer=z0_samples)

        # concat across time (num_samples x num_time_pts x dim_latent)
        self.z_samples_prior = tf.concat(
            [tf.expand_dims(z0_samples, axis=1),
             tf.transpose(z_samples, perm=[1, 0, 2])], axis=1)

    def _sample_y(self):

        # expand dims to account for time and mc dims when applying mapping
        # now (1 x num_samples x num_time_pts x dim_latent)
        z_samples_ex = tf.expand_dims(self.z_samples_prior, axis=0)

        y_means_ls = []  # contribution from latent space
        y_means_lp = []  # contribution from linear predictors
        y_means = []
        for pop, pop_dim in enumerate(self.dim_obs):
            y_means_ls.append(tf.squeeze(self.networks[pop].apply_network(
                z_samples_ex[:, :, :,
                self.latent_indxs[pop][0]:
                self.latent_indxs[pop][-1]]),
                axis=0))
            if self.dim_predictors is not None:
                # append new list for this population
                y_means_lp.append([])
                for pred, pred_dim in enumerate(self.dim_predictors):
                    if self.predictor_indx[pop][pred] is not None:
                        net_out = self.networks_linear[pop][pred]. \
                            apply_network(self.linear_predictors_phs[pred])
                        y_means_lp[-1].append(net_out)
                    # else:
                    #     self.y_pred_lp[-1].append(0.0)
                y_means.append(
                    tf.add(y_means_ls[-1], tf.add_n(y_means_lp[-1])))
            else:
                y_means.append(y_means_ls[-1])

        # get random samples from observation space
        if self.noise_dist is 'gaussian':
            obs_rand_samples = []
            for pop, pop_dim in enumerate(self.dim_obs):
                obs_rand_samples.append(tf.random_normal(
                    shape=[self.num_samples_ph, self.num_time_pts, pop_dim],
                    mean=0.0, stddev=1.0, dtype=self.dtype,
                    name=str('obs_rand_samples_%02i' % pop)))
                self.y_samples_prior.append(y_means[pop] + tf.multiply(
                    obs_rand_samples[pop], self.R_sqrt[pop]))

        elif self.noise_dist is 'poisson':
            for pop, pop_dim in enumerate(self.dim_obs):
                self.y_samples_prior.append(tf.squeeze(tf.random_poisson(
                    lam=y_means[pop], shape=[1], dtype=self.dtype), axis=0))

    def log_density(self, y, z):
        """
        Evaluate log density for generative model, defined as
        p(y, z) = p(y | z) p(z)
        where
        p(z) = \prod_t p(z_t), z_t ~ N(A z_{t-1}, Q)
        p(y | z) = \prod_t p(y_t | z_t)

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

        log_density_y = []

        for pop, pop_dim in enumerate(self.dim_obs):
            with tf.variable_scope('population_%02i' % pop):

                if self.noise_dist is 'gaussian':
                    # expand observation dims over mc samples
                    res_y = tf.expand_dims(y[pop], axis=1) - self.y_pred[pop]

                    # average over batch and mc sample dimensions
                    res_y_R_inv_res_y = tf.reduce_mean(
                        tf.multiply(tf.square(res_y), self.R_inv[pop]),
                        axis=[0, 1])

                    # sum over time and observation dimensions
                    test_like = tf.reduce_sum(res_y_R_inv_res_y)
                    tf.summary.scalar('log_joint_like', -0.5 * test_like)

                    # total term for likelihood
                    log_density_y.append(-0.5 * (test_like
                         + self.num_time_pts * tf.reduce_sum(
                                tf.log(self.R[pop]))
                         + self.num_time_pts * pop_dim * tf.log(2.0 * np.pi)))

                elif self.noise_dist is 'poisson':
                    # expand observation dims over mc samples
                    obs_y = tf.expand_dims(y[pop], axis=1)

                    # average over batch and mc sample dimensions
                    log_density_ya = tf.reduce_mean(
                        tf.multiply(obs_y[pop], tf.log(self.y_pred[pop]))
                        - self.y_pred[pop]
                        - tf.lgamma(1 + obs_y[pop]),
                        axis=[0, 1])

                    # sum over time and observation dimensions
                    log_density_y.append(tf.reduce_sum(log_density_ya))
                    tf.summary.scalar('log_joint_like', log_density_y[-1])

                else:
                    raise ValueError

        return tf.add_n(log_density_y, name='log_joint_like_total')

    def _log_density_prior(self, z):
        self.res_z0 = res_z0 = z[:, :, 0, :] - self.z0_mean
        self.res_z = res_z = z[:, :, 1:, :] - tf.tensordot(
            z[:, :, :-1, :], tf.transpose(self.A), axes=[[3], [0]])

        # average over batch and mc sample dimensions
        res_z_Q_inv_res_z = tf.reduce_mean(tf.multiply(
            tf.tensordot(res_z, self.Q_inv, axes=[[3], [0]]), res_z),
            axis=[0, 1])
        res_z0_Q0_inv_res_z0 = tf.reduce_mean(tf.multiply(
            tf.tensordot(res_z0, self.Q0_inv, axes=[[2], [0]]), res_z0),
            axis=[0, 1])

        # sum over time and latent dimensions
        test_prior = tf.reduce_sum(res_z_Q_inv_res_z)
        test_prior0 = tf.reduce_sum(res_z0_Q0_inv_res_z0)
        tf.summary.scalar('log_joint_prior', -0.5 * test_prior)
        tf.summary.scalar('log_joint_prior0', -0.5 * test_prior0)

        # total term for prior
        log_density_z = -0.5 * (test_prior + test_prior0
            + (self.num_time_pts - 1) * tf.log(tf.matrix_determinant(self.Q))
            + tf.log(tf.matrix_determinant(self.Q0))
            + self.num_time_pts * sum(self.dim_latent) * tf.log(2.0 * np.pi))

        return log_density_z

    def sample(self, sess, num_samples=1, seed=None, linear_predictors=None):
        """
        Generate samples from the model

        Args:
            sess (tf.Session object)
            num_samples (int, optional)
            seed (int, optional)
            linear_predictors (list)

        Returns:
            num_samples x num_time_pts x dim_obs x numpy array:
                sample observations y
            num_samples x num_time_pts x dim_latent numpy array:
                sample latent states z

        """

        if seed is not None:
            tf.set_random_seed(seed)

        if self.dim_predictors is not None and linear_predictors is None:
            raise ValueError('must supply linear predictors for sampling')

        feed_dict = {self.num_samples_ph: num_samples}
        if self.dim_predictors is not None:
            for pred, pred_ph in enumerate(self.linear_predictors_phs):
                feed_dict[pred_ph] = linear_predictors[pred]

        [y, z] = sess.run(
            [self.y_samples_prior, self.z_samples_prior],
            feed_dict=feed_dict)

        return y, z

    def get_params(self, sess):
        """Get parameters of generative model"""

        if self.noise_dist is 'gaussian':
            A, R_sqrt, z0_mean, Q, Q0 = sess.run(
                [self.A, self.R_sqrt, self.z0_mean, self.Q, self.Q0])

            param_dict = {
                'A': A, 'R': np.square(R_sqrt), 'z0_mean': z0_mean,
                'Q': Q, 'Q0': Q0}

        elif self.noise_dist is 'poisson':
            A, z0_mean, Q, Q0 = sess.run(
                [self.A, self.z0_mean, self.Q, self.Q0])

            param_dict = {
                'A': A, 'z0_mean': z0_mean, 'Q': Q, 'Q0': Q0}
        else:
            raise ValueError

        return param_dict

    def get_linear_params(self, sess):
        """Get parameters of linear regressors"""

        param_dict = []
        for pop, pop_dim in enumerate(self.dim_obs):
            param_dict.append([])
            for pred, pred_dim in enumerate(self.dim_predictors):
                if self.predictor_indx[pop][pred] is not None:
                    layer_weights_ = sess.run(
                        self.networks_linear[pop][pred].layers[0].weights)
                else:
                    layer_weights_ = []
                param_dict[pop].append(layer_weights_)

        return param_dict


class NetLDS(NetFLDS):
    """
    Generative model is defined as

    z_t ~ N(A z_{t-1}, Q)
    y_t^i ~ N(C_i z_t^i + d_i, R_i)

    for each population i, where the z_t^i are non-overlapping subsets of z_t
    """

    def __init__(
            self, dim_obs=None, dim_latent=None, linear_predictors=None,
            num_time_pts=None, gen_params=None, noise_dist='gaussian',
            post_z_samples=None, **kwargs):
        """
        Args:
            dim_obs (list): observation dimension for each population
            dim_latent (list): latent dimension for each population
            linear_predictors (dict):
                'dim_predictors' (list): dimension for each set of linear
                    predictors
                'predictor_indx' (list of lists): each element of the list
                    contains the indices of the predictors in the
                    `dim_predictors` list used by the corresponding population
            num_time_pts (int): number of time points per observation of the
                dynamical sequence
            gen_params (dict): dictionary of generative params for initializing
                model
            noise_dist (str): 'gaussian' | 'poisson'
            post_z_samples (batch_size x num_mc_samples x num_time_pts x
                dim_latent tf.Tensor): samples from the (appx) posterior of the
                latent states

        """

        if gen_params is None:
            gen_params = {}

        # iterate through populations
        # NOTE: must set kernel/bias initializers outside of this constructor
        # for now since NetFLDS assumes nn_params is the same for each pop
        for pop, _ in enumerate(dim_obs):

            # emissions matrix
            if 'C' in gen_params:
                kernel_initializer = tf.constant_initializer(
                    gen_params['C'][pop], dtype=self.dtype)
            else:
                kernel_initializer = 'trunc_normal'

            # biases
            if 'd' in gen_params:
                bias_initializer = tf.constant_initializer(
                    gen_params['d'][pop], dtype=self.dtype)
            else:
                bias_initializer = 'zeros'

            # list of dicts specifying (linear) nn to observations
            nn_params = [{
                'units': dim_obs[pop],
                'activation': 'linear',
                'kernel_initializer': kernel_initializer,
                'bias_initializer': bias_initializer,
                'kernel_regularizer': None,
                'bias_regularizer': None}]

        super().__init__(
            dim_obs=dim_obs, dim_latent=dim_latent, nn_params=nn_params,
            linear_predictors=linear_predictors, noise_dist=noise_dist,
            post_z_samples=post_z_samples, num_time_pts=num_time_pts,
            gen_params=gen_params)

    def get_params(self, sess):
        """Get parameters of generative model"""

        param_dict = super().get_params(sess)

        param_dict['C'] = []
        param_dict['d'] = []
        for pop, pop_dim in enumerate(self.dim_obs):
            layer_weights = sess.run(self.networks[pop].layers[0].weights)
            param_dict['C'].append(layer_weights[0])
            param_dict['d'].append(layer_weights[1])

        return param_dict


class FLDS(NetFLDS):
    """
    Generative model is defined as
    z_t ~ N(A z_{t-1}, Q)
    E[y_t] ~ f(z_t)
    """

    def __init__(
            self, dim_obs=None, dim_latent=None, dim_predictors=None,
            num_time_pts=None, gen_params=None, noise_dist='gaussian',
            nn_params=None, post_z_samples=None, **kwargs):
        """
        Args:
            dim_obs (int): observation dimension
            dim_latent (int): latent dimension
            dim_predictors (list): dimension for each set of linear  predictors
            num_time_pts (int): number of time points per observation of the
                dynamical sequence
            gen_params (dict): dictionary of generative params for initializing
                model
            noise_dist (str): 'gaussian' | 'poisson'
            nn_params (list): dictionaries for building each layer of the
                mapping from the latent space to observations; the same
                network architecture is used for each population
            post_z_samples (batch_size x num_mc_samples x num_time_pts x
                dim_latent tf.Tensor): samples from the (appx) posterior of the
                latent states

        """

        if dim_predictors is not None:
            linear_predictors = {
                'dim_predictors': dim_predictors,
                'predictor_indx': [range(len(dim_predictors))]}
            if 'predictor_params' in kwargs:
                linear_predictors['predictor_params'] = \
                    [kwargs['predictor_params']]
        else:
            linear_predictors = None

        super().__init__(
            dim_obs=[dim_obs], dim_latent=[dim_latent],
            linear_predictors=linear_predictors,
            post_z_samples=post_z_samples, num_time_pts=num_time_pts,
            gen_params=gen_params, nn_params=nn_params, noise_dist=noise_dist)

    def sample(self, sess, num_samples=1, seed=None, linear_predictors=None):
        y, z = super().sample(sess, num_samples, seed, linear_predictors)
        return y[0], z


class LDS(NetFLDS):
    """
    Generative model is defined as
    z_t ~ N(A z_{t-1}, Q)
    y_t ~ N(C z_t + d, R)
    """

    def __init__(
            self, dim_obs=None, dim_latent=None, dim_predictors=None,
            num_time_pts=None, gen_params=None, noise_dist='gaussian',
            post_z_samples=None, **kwargs):
        """
        Args:
            dim_obs (int): observation dimension
            dim_latent (int): latent dimension
            dim_predictors (list): dimension for each set of linear predictors
            num_time_pts (int): number of time points per observation of the
                dynamical sequence
            gen_params (dict): dictionary of generative params for initializing
                model
            noise_dist (str): 'gaussian' | 'poisson'
            post_z_samples (batch_size x num_mc_samples x num_time_pts x
                dim_latent tf.Tensor): samples from the (appx) posterior of the
                latent states

        """

        if gen_params is None:
            gen_params = {}

        # emissions matrix
        if 'C' in gen_params:
            kernel_initializer = tf.constant_initializer(
                gen_params['C'], dtype=self.dtype)
        else:
            kernel_initializer = 'trunc_normal'

        # biases
        if 'd' in gen_params:
            bias_initializer = tf.constant_initializer(
                gen_params['d'], dtype=self.dtype)
        else:
            bias_initializer = 'zeros'

        # list of dicts specifying (linear) nn to observations
        nn_params = [{
            'units': dim_obs,
            'activation': 'linear',
            'kernel_initializer': kernel_initializer,
            'bias_initializer': bias_initializer,
            'kernel_regularizer': None,
            'bias_regularizer': None}]

        if dim_predictors is not None:
            linear_predictors = {
                'dim_predictors': dim_predictors,
                'predictor_indx': [range(len(dim_predictors))]}
            if 'predictor_params' in kwargs:
                linear_predictors['predictor_params'] = \
                    [kwargs['predictor_params']]
        else:
            linear_predictors = None

        super().__init__(
            dim_obs=[dim_obs], dim_latent=[dim_latent],
            linear_predictors=linear_predictors,
            post_z_samples=post_z_samples, num_time_pts=num_time_pts,
            gen_params=gen_params, nn_params=nn_params, noise_dist=noise_dist)

    def sample(self, sess, num_samples=1, seed=None, linear_predictors=None):
        y, z = super().sample(sess, num_samples, seed, linear_predictors)
        return y[0], z

    def get_params(self, sess):
        """Get parameters of generative model"""

        param_dict = super().get_params(sess)

        layer_weights = sess.run(self.networks[0].layers[0].weights)
        param_dict['C'] = layer_weights[0]
        param_dict['d'] = layer_weights[1]

        return param_dict
