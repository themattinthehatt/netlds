"""Model class for building models"""

import os
import numpy as np
import tensorflow as tf
from netlds.generative import *
from netlds.inference import *
from netlds.trainer import Trainer


class Model(object):
    """Base class for models"""

    def __init__(
            self, inf_network=None, inf_network_params=None, gen_model=None,
            gen_model_params=None, np_seed=0, tf_seed=0):
        """
        Constructor for full Model; combines an inference network with a
        generative model and provides training functions

        Args:
            inf_network (InferenceNetwork class)
            inf_network_params (dict)
            gen_model (GenerativeModel class)
            gen_model_params (dict)
            np_seed (int): for training minibatches
            tf_seed (int): for initializing tf.Variables (sampling functions
                have their own seed arguments)

        """

        # initialize inference network and generative models
        self.inf_net = inf_network(**inf_network_params)
        self.gen_net = gen_model(**gen_model_params)

        # initialize Trainer object
        self.trainer = Trainer()

        # location of generative model params if not part of Model
        self.checkpoint = None

        # set parameters for graph (constructed for each train)
        self.graph = None
        self.saver = None
        self.merge_summaries = None
        self.init = None
        self.sess_config = None
        self.np_seed = np_seed
        self.tf_seed = tf_seed

        # save constructor inputs for easy save/load
        constructor_inputs = {
            'inf_network': inf_network,
            'inf_network_params': inf_network_params,
            'gen_model': gen_model,
            'gen_model_params': gen_model_params,
            'np_seed': np_seed,
            'tf_seed': tf_seed}
        self.constructor_inputs = constructor_inputs

    def build_graph(self):
        """Build tensorflow computation graph for model"""
        raise NotImplementedError

    def _define_objective(self):
        """Objective function used to optimize model parameters"""
        self.objective = None
        raise NotImplementedError

    def train(self, **kwargs):
        """
        Train model

        See Trainer.train for input options
        """

        self.trainer.train(self, **kwargs)

    def sample(
            self, ztype='prior', num_samples=1, seed=None,
            checkpoint_file=None):
        """
        Generate samples from prior/posterior and model

        Args:
            ztype (str): distribution used for latent state samples
                'prior' | 'posterior'
            num_samples (int, optional)
            seed (int, optional): random seed for reproducibly generating
                random samples
            checkpoint_file (str, optional): checkpoint file specifying model
                from which to generate samples; if `None`, will then look for a
                checkpoint file created upon model initialization

        Returns:
            num_time_pts x dim_obs x num_samples numpy array: y
            num_time_pts x dim_latent x num_samples numpy array: z

        Raises:
            ValueError: for incorrect `ztype` values

        """

        # intialize session
        with tf.Session(graph=self.graph, config=self.sess_config) as sess:
            self.restore_model(sess, checkpoint_file=checkpoint_file)
            if ztype is 'prior':
                y, z = self.gen_net.sample(sess, num_samples, seed)
            elif ztype is 'posterior':
                # TODO: needs data input as well
                y = None
                z = self.gen_net.sample_z(sess, num_samples, seed)
            else:
                raise ValueError('Invalid string "%s" for ztype argument')

        return y, z

    def checkpoint_model(
            self, sess=None, checkpoint_file=None, save_filepath=False,
            print_filepath=False, opt_params=None):
        """
        Checkpoint model parameters in tf.Variables. The tensorflow graph will
        be constructed if necessary.

        Args:
            sess (tf.Session object, optional): current session object to run
                graph; if `None`, a session will be created
            checkpoint_file (str, optional): full path to output file; if
                `None`, the code will check for the `checkpoint_file` attribute
                of the model
            save_filepath (str, optional): save filepath as an attribute of the
                model
            print_filepath (bool, optional): print path of checkpoint file
            opt_params (dict): specify optimizer params if building graph for
                the first time

        Raises:
            ValueError: if no checkpoint_file is found

        """

        if checkpoint_file is None:
            if self.checkpoint is not None:
                checkpoint_file = self.checkpoint
            else:
                raise ValueError('Must specify checkpoint file')

        if not os.path.isdir(os.path.dirname(checkpoint_file)):
            os.makedirs(os.path.dirname(checkpoint_file))

        if sess is None:
            # assume we are saving an initial model
            # build tensorflow computation graph
            if self.graph is None:
                if opt_params is not None:
                    self.trainer.parse_optimizer_options(**opt_params)
                self.build_graph()

            # intialize session
            with tf.Session(graph=self.graph, config=self.sess_config) as sess:
                sess.run(self.init)
                # save graph
                tf.summary.FileWriter(checkpoint_file, graph=sess.graph)
                # checkpoint variables
                self.checkpoint_model(
                    sess=sess,
                    checkpoint_file=checkpoint_file,
                    save_filepath=False,
                    print_filepath=False)
                # save/print filepath in outer loop
                save_filepath = True
        else:
            self.saver.save(sess, checkpoint_file)

        if save_filepath:
            self.checkpoint = checkpoint_file

        if print_filepath:
            print('Model checkpointed to %s' % checkpoint_file)

    def restore_model(self, sess, checkpoint_file=None):
        """
        Restore previously checkpointed model parameters in tf.Variables

        Args:
            sess (tf.Session object): current session object to run graph
            checkpoint_file (str): full path to saved model

        Raises:
            ValueError: If `checkpoint_file` is not a valid filename

        """

        if checkpoint_file is None:
            if self.checkpoint is not None:
                checkpoint_file = self.checkpoint
            else:
                raise ValueError('Must specify checkpoint file')

        if not os.path.isfile(checkpoint_file + '.meta'):
            raise ValueError(
                str('"%s" is not a valid filename' % checkpoint_file))

        # restore saved variables into tf Variables
        self.saver.restore(sess, checkpoint_file)

    def save_model(self, save_file):
        """
        Save constructor inputs of model using pickle

        Args:
            save_file (str): full path to output file

        Example:
            model_0 = Model(...) # call constructor
            model_0.train(...)   # should checkpoint models here
            model_0.save_model('/path/to/file/model_0')

            model_1 = Model.load_model('/path/to/file/model_0')

            In order for model_1 to use the parameters learned during the call
            to model_0.train(), model_0 must checkpoint its parameters and
            store them in the `checkpoint` Model attribute; this attribute will
            be used by model_1 if not `None` for restoring model parameters
            (see the Model.train function for more info on how to specify when
            model parameters are checkpointed)

        """

        import pickle

        if not os.path.isdir(os.path.dirname(save_file)):
            os.makedirs(os.path.dirname(save_file))

        # grab constructor inputs (along with model class specification)
        constructor_inputs = dict(self.constructor_inputs)

        # trainer parameters (for rebuilding graph)
        constructor_inputs['learning_alg'] = self.trainer.learning_alg
        constructor_inputs['opt_params'] = self.trainer.opt_params

        # save checkpoint file as well
        if self.checkpoint is not None:
            constructor_inputs['checkpoint_file'] = self.checkpoint
        else:
            print('Warning: model has not been checkpointed; restoring this '
                  'model will result in random parameters')
            constructor_inputs['checkpoint_file'] = None

        with open(save_file, 'wb') as f:
            pickle.dump(constructor_inputs, f)

        print('Model pickled to %s' % save_file)

    @classmethod
    def load_model(cls, save_file):
        """
        Restore previously saved Model object

        Args:
            save_file (str): full path to saved model

        Raises:
            ValueError: If `save_file` is not a valid filename

        """

        import pickle

        if not os.path.isfile(save_file):
            raise ValueError(str('%s is not a valid filename' % save_file))

        with open(save_file, 'rb') as f:
            constructor_inputs = pickle.load(f)

        print('Model loaded from %s' % save_file)

        # extract model class to use as constructor
        model_class = constructor_inputs['model_class']
        del constructor_inputs['model_class']

        # extract trainer info for building graph
        learning_alg = constructor_inputs['learning_alg']
        del constructor_inputs['learning_alg']
        opt_params = constructor_inputs['opt_params']
        del constructor_inputs['opt_params']

        # tell model where to find checkpoint file for restoring parameters
        checkpoint_file = constructor_inputs['checkpoint_file']
        del constructor_inputs['checkpoint_file']
        if checkpoint_file is None:
            print('Warning: model has not been checkpointed; restoring this '
                  'model will result in random parameters')

        # initialize model
        model = model_class(**constructor_inputs)
        model.checkpoint = checkpoint_file

        # specify trainer params
        model.trainer.parse_optimizer_options(
            learning_alg=learning_alg, **opt_params)

        # build graph
        model.build_graph()

        return model


class DynamicalModel(Model):
    """
    Models with dynamical generative models; should be subclassed by a specific
    model that implements a `build_graph` method
    """

    def __init__(
            self, inf_network=None, inf_network_params=None, gen_model=None,
            gen_model_params=None, np_seed=0, tf_seed=0):
        """
        Constructor for full Model; combines an inference network with a
        generative model and provides training functions

        Args:
            inf_network (InferenceNetwork class)
            inf_network_params (dict)
            gen_model (GenerativeModel class)
            gen_model_params (dict)
            np_seed (int)
            tf_seed (int)

        """

        super(DynamicalModel, self).__init__(
            inf_network=inf_network, inf_network_params=inf_network_params,
            gen_model=gen_model, gen_model_params=gen_model_params,
            np_seed=np_seed, tf_seed=tf_seed)

        # to clean up training functions
        self.dim_obs = self.gen_net.dim_obs
        self.dim_latent = self.gen_net.dim_latent
        self.num_time_pts = self.gen_net.num_time_pts

    def build_graph(self):
        """Build tensorflow computation graph for model"""
        raise NotImplementedError

    def _define_objective(self):
        """
        Objective function used to optimize model parameters

        This function uses the log-joint/entropy formulation of the ELBO
        """

        # expected value of log joint distribution
        with tf.variable_scope('log_joint'):
            self.log_joint = self.gen_net.log_density(
                self.gen_net.y_pred, self.inf_net.post_z_samples)

        # entropy of approximate posterior
        with tf.variable_scope('entropy'):
            self.entropy = self.inf_net.entropy()

        # objective to minimize
        self.objective = -self.log_joint - self.entropy

        # save summaries
        # with tf.variable_scope('summaries'):
        tf.summary.scalar('log_joint', self.log_joint)
        tf.summary.scalar('entropy', self.entropy)
        tf.summary.scalar('elbo', -self.objective)

    def get_dynamics_params(self, checkpoint_file=None):
        """
        Get parameters of generative model

        Args:
            checkpoint_file (str, optional): location of checkpoint file
                specifying model from which to generate samples; if `None`,
                will then look for a checkpoint file created upon model
                initialization

        Returns:
            params (dict)

        """

        with tf.Session(graph=self.graph, config=self.sess_config) as sess:
            self.restore_model(sess, checkpoint_file=checkpoint_file)
            params = self.gen_net.get_params(sess)

        return params

    def get_posterior_means(self, observations=None, checkpoint_file=None):
        """
        Get posterior means from inference network

        Args:
            observations (num_samples x num_time_pts x dim_obs tf.Tensor):
                observations on which to condition the posterior means
            checkpoint_file (str, optional): location of checkpoint file
                specifying model from which to generate samples; if `None`,
                will then look for a checkpoint file created upon model
                initialization

        Returns:
            posterior_means (num_samples x num_time_pts x dim_latent tf.Tensor)

        """

        with tf.Session(graph=self.graph, config=self.sess_config) as sess:
            self.restore_model(sess, checkpoint_file=checkpoint_file)
            posterior_means = self.inf_net.get_posterior_means(
                sess, observations)

        return posterior_means

    def get_cost(self, observations=None, input_data=None, indxs=None,
                 checkpoint_file=None):
        """
        User function for retrieving cost

        Args:
            observations (num_samples x num_time_pts x dim_obs tf.Tensor):
                observations on which to condition the posterior means
            input_data (num_samples x num_time_pts x dim_input tf.Tensor,
                optional)
            indxs (list, optional): list of indices into observations and
                input_data
            checkpoint_file (str, optional): location of checkpoint file
                specifying model from which to generate samples; if `None`,
                will then look for a checkpoint file created upon model
                initialization

        Returns:
            float: value of objective function

        """

        if indxs is None:
            indxs = list(range(observations.shape[0]))

        with tf.Session(graph=self.graph, config=self.sess_config) as sess:
            self.restore_model(sess, checkpoint_file=checkpoint_file)
            cost = self.trainer._get_cost(
                sess=sess, observations=observations, input_data=input_data,
                indxs=indxs)

        return cost


class LDSCoupledModel(DynamicalModel):
    """LDS generative model, LDS approximate posterior, shared parameters"""

    _allowed_inf_networks = ['SmoothingLDSCoupled']
    _allowed_gen_models = [
        'LDSCoupled', 'PLDSCoupled', 'FLDSCoupled', 'PFLDSCoupled']

    def __init__(
            self, inf_network=None, inf_network_params=None, gen_model=None,
            gen_model_params=None, np_seed=0, tf_seed=0):
        """
        Constructor for full Model; see DynamicalModel for full arg
        documentation

        Args:
            inf_network (InferenceNetwork class):
                SmoothingLDSCoupled
            gen_model (GenerativeModel class)
                LDSCoupled | PLDSCoupled | FLDSCoupled | PFLDSCoupled

        """

        if inf_network.__name__ not in self._allowed_inf_networks:
            raise ValueError('%s is not a valid inference network for the '
                             'LDSCoupledModel class; must use '
                             'SmoothingLDSCoupled inference network instead.'
                             % inf_network.__name__)
        if gen_model.__name__ not in self._allowed_gen_models:
            raise ValueError('%s is not a valid generative model for the '
                             'LDSCoupledModel class; must use '
                             '*LDSCoupled generative model instead.'
                             % gen_model.__name__)

        super(LDSCoupledModel, self).__init__(
            inf_network=inf_network, inf_network_params=inf_network_params,
            gen_model=gen_model, gen_model_params=gen_model_params,
            np_seed=np_seed, tf_seed=tf_seed)

        self.constructor_inputs['model_class'] = LDSCoupledModel

    def build_graph(self):
        """Build tensorflow computation graph for model"""

        self.graph = tf.Graph()  # must be initialized before graph creation

        # build model graph
        with self.graph.as_default():

            # set random seed for this graph
            tf.set_random_seed(self.tf_seed)

            with tf.variable_scope('shared_vars'):
                z0_mean, A, Q_sqrt, Q, Qinv, Q0_sqrt, Q0, Q0inv = \
                    self.gen_net.initialize_prior_vars()

            with tf.variable_scope('inference_network'):
                self.inf_net.build_graph(
                    z0_mean, A, Q_sqrt, Q, Qinv, Q0_sqrt, Q0, Q0inv)

            with tf.variable_scope('generative_model'):
                self.gen_net.build_graph(
                    self.inf_net.post_z_samples,
                    z0_mean, A, Q_sqrt, Q, Qinv, Q0_sqrt, Q0, Q0inv)

            with tf.variable_scope('objective'):
                self._define_objective()

            with tf.variable_scope('optimizer'):
                self.trainer._define_optimizer_op(self)

            # add additional ops
            # for saving and restoring models (initialized after var creation)
            self.saver = tf.train.Saver()
            # collect all summaries into a single op
            self.merge_summaries = tf.summary.merge_all()
            # add variable initialization op to graph
            self.init = tf.global_variables_initializer()


class LDSModel(DynamicalModel):
    """LDS generative model, various options for approximate posterior"""

    _allowed_inf_networks = [
        'MeanFieldGaussian', 'MeanFieldGaussianTemporal', 'SmoothingLDS']
    _allowed_gen_models = ['LDS', 'PLDS', 'FLDS', 'PFLDS']

    def __init__(
            self, inf_network=None, inf_network_params=None, gen_model=None,
            gen_model_params=None, np_seed=0, tf_seed=0):
        """
        Constructor for full Model; see DynamicalModel for arg documentation

        Args:
            inf_network (InferenceNetwork class):
                MeanFieldGaussian | MeanFieldGaussianTemporal | SmoothingLDS
            gen_model (GenerativeModel class)
                LDS | PLDS | FLDS | PFLDS

        """

        if inf_network.__name__ not in self._allowed_inf_networks:
            raise ValueError('%s is not a valid inference network for the '
                             'LDSModel class.' % inf_network.__name__)
        if gen_model.__name__ not in self._allowed_gen_models:
            raise ValueError('%s is not a valid generative model for the '
                             'LDSCoupledModel class' % gen_model.__name__)

        super(LDSModel, self).__init__(
            inf_network=inf_network, inf_network_params=inf_network_params,
            gen_model=gen_model, gen_model_params=gen_model_params,
            np_seed=np_seed, tf_seed=tf_seed)

        self.constructor_inputs['model_class'] = LDSModel

    def build_graph(self, opt_params=None):
        """Build tensorflow computation graph for model"""

        self.graph = tf.Graph()  # must be initialized before graph creation

        # build model graph
        with self.graph.as_default():

            # set random seed for this graph
            tf.set_random_seed(self.tf_seed)

            with tf.variable_scope('inference_network'):
                self.inf_net.build_graph()

            with tf.variable_scope('generative_model'):
                self.gen_net.build_graph(self.inf_net.post_z_samples)

            with tf.variable_scope('objective'):
                self._define_objective()

            with tf.variable_scope('optimizer'):
                self.trainer._define_optimizer_op(self)

            # add additional ops
            # for saving and restoring models (initialized after var creation)
            self.saver = tf.train.Saver()
            # collect all summaries into a single op
            self.merge_summaries = tf.summary.merge_all()
            # add variable initialization op to graph
            self.init = tf.global_variables_initializer()
