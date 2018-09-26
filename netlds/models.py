"""Model class for building and training models"""

import os
import numpy as np
import tensorflow as tf
from netlds.generative import *
from netlds.inference import *
import time


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

    def build_graph(self, opt_params=None):
        """Build tensorflow computation graph for model"""
        raise NotImplementedError

    def _define_objective(self):
        """Objective function used to optimize model parameters"""
        self.objective = None
        raise NotImplementedError

    def _define_optimizer(self, opt_params):
        """Define one step of the optimization routine"""

        if opt_params['learning_alg'] is 'adam':
            self.train_step = tf.train.AdamOptimizer(
                learning_rate=opt_params['adam']['learning_rate'],
                beta1=opt_params['adam']['beta1'],
                beta2=opt_params['adam']['beta2'],
                epsilon=opt_params['adam']['epsilon']). \
                minimize(self.objective)
        elif opt_params['learning_alg'] is 'adagrad':
            self.train_step = tf.train.AdagradOptimizer(
                learning_rate=opt_params['adagrad']['learning_rate'],
                initial_accumulator_value=opt_params[
                    'adagrad']['initial_accumulator_value']). \
                minimize(self.objective)
        elif opt_params['learning_alg'] is 'adadelta':
            self.train_step = tf.train.AdadeltaOptimizer(
                learning_rate=opt_params['adadelta']['learning_rate'],
                rho=opt_params['adadelta']['rho'],
                epsilon=opt_params['adadelta']['epsilon']). \
                minimize(self.objective)

    def train(
            self, observations=None, input_data=None, train_indxs=None,
            test_indxs=None, opt_params=None, output_dir=None,
            checkpoint_file=None):
        """
        Model training function

        Args:
            observations (num_reps x num_time_pts x dim_obs numpy array)
            input_data (num_reps x num_time_pts x dim_input numpy array,
                optional): input to inference network; if using observations,
                leave as `None`.
            train_indxs (numpy array, optional): subset of data to use for
                training
            test_indxs (numpy array, optional): subset of data to use for
                testing; if available these are used when displaying updates,
                and are also the indices used for early stopping if enabled
            opt_params (dict, optional): optimizer-specific parameters; see
                Model.optimizer_defaults method for valid key-value pairs and
                corresponding default values.
            output_dir (str, optional): absolute path for saving checkpoint
                files and summary files; must be present if either
                `epochs_ckpt` or `epochs_summary` values in `opt_params` is not
                `None`. If `output_dir` is not `None`, regardless of checkpoint
                or summary settings, the graph will automatically be saved.
                Must be present if early_stopping is desired to restore the
                best fit, otherwise it will restore the model at break point.
            checkpoint_file (str, optional): restore parameters from a
                checkpoint file before beginning training; if 'self', the file
                stored in self.checkpoint is used

        Returns:
            int: number of total training epochs

        Raises:
            ValueError: If `epochs_ckpt` value in `opt_params` is not `None`
                and `output_dir` is `None`
            ValueError: If `epochs_summary` in `opt_params` is not `None` and
                `output_dir` is `None`
            ValueError: If `early_stop` > 0 and `test_indxs` is 'None'

        """

        # Check format of opt_params (and add some defaults)
        if opt_params is None:
            opt_params = {}
        opt_params = self.optimizer_defaults(opt_params)

        if train_indxs is None:
            train_indxs = np.arange(observations.shape[0])

        # Check values entered
        if opt_params['epochs_ckpt'] is not None and output_dir is None:
            raise ValueError(
                'output_dir must be specified to save model')
        if opt_params['epochs_summary'] is not None and output_dir is None:
            raise ValueError(
                'output_dir must be specified to save summaries')
        if opt_params['early_stop'] > 0 and test_indxs is None:
            raise ValueError(
                'test_indxs must be specified for early stopping')

        # for specifying device
        if opt_params['use_gpu']:
            self.sess_config = tf.ConfigProto(device_count={'GPU': 1})
        else:
            self.sess_config = tf.ConfigProto(device_count={'GPU': 0})

        # build tensorflow computation graph
        if self.graph is None:
            self.build_graph(opt_params=opt_params)

        # intialize session
        with tf.Session(graph=self.graph, config=self.sess_config) as sess:

            # handle output directories
            train_writer = None
            test_writer = None
            if output_dir is not None:

                # remake checkpoint directory
                if opt_params['epochs_ckpt'] is not None:
                    ckpts_dir = os.path.join(output_dir, 'ckpts')
                    if os.path.isdir(ckpts_dir):
                        tf.gfile.DeleteRecursively(ckpts_dir)
                    os.makedirs(ckpts_dir)

                # remake training summary directories
                summary_dir_train = os.path.join(
                    output_dir, 'summaries', 'train')
                if os.path.isdir(summary_dir_train):
                    tf.gfile.DeleteRecursively(summary_dir_train)
                os.makedirs(summary_dir_train)
                train_writer = tf.summary.FileWriter(
                    summary_dir_train, graph=sess.graph)

                # remake testing summary directories
                summary_dir_test = os.path.join(
                    output_dir, 'summaries', 'test')
                if test_indxs is not None:
                    if os.path.isdir(summary_dir_test):
                        tf.gfile.DeleteRecursively(summary_dir_test)
                    os.makedirs(summary_dir_test)
                    test_writer = tf.summary.FileWriter(
                        summary_dir_test, graph=sess.graph)

            # initialize all parameters
            sess.run(self.init)

            # restore params from a previous session
            if checkpoint_file is 'self':
                if self.checkpoint is not None:
                    self.restore_model(sess, checkpoint_file=self.checkpoint)
                else:
                    raise ValueError('self.checkpoint is `None` file')
            elif checkpoint_file is not None:
                self.restore_model(sess, checkpoint_file=checkpoint_file)

            # start/resume training
            epoch = self._train_loop(
                sess=sess,
                train_writer=train_writer,
                test_writer=test_writer,
                train_indxs=train_indxs,
                test_indxs=test_indxs,
                input_data=input_data,
                observations=observations,
                opt_params=opt_params,
                output_dir=output_dir)

        return epoch

    def _train_loop(
            self, sess=None, train_writer=None, test_writer=None,
            train_indxs=None, test_indxs=None, observations=None,
            input_data=None, opt_params=None, output_dir=None):
        """Training function for adam optimizer to clean up code in `train`"""

        # define useful quantities
        epochs_training = opt_params['epochs_training']
        epochs_display = opt_params['epochs_display']
        epochs_ckpt = opt_params['epochs_ckpt']
        epochs_summary = opt_params['epochs_summary']
        batch_size = opt_params['batch_size']

        if opt_params['run_diagnostics']:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
        else:
            run_options = None
            run_metadata = None

        if opt_params['early_stop'] > 0:
            prev_costs = np.multiply(np.ones(opt_params['early_stop']), np.nan)
            early_stop_params = {
                'prev_costs': prev_costs,
                'best_epoch': 0,
                'best_cost': np.inf,
                'chkpted': False,
                'output_dir': output_dir}
        else:
            early_stop_params = None

        num_batches = train_indxs.shape[0] // batch_size

        np.random.seed(self.np_seed)

        # start training loop
        for epoch in range(epochs_training):

            # shuffle data before each pass
            train_indxs_perm = np.random.permutation(train_indxs)

            # pass through dataset once
            start = time.time()
            for batch in range(num_batches):
                # get training indices for this batch
                batch_indxs = train_indxs_perm[
                    batch * batch_size:
                    (batch + 1) * batch_size]

                # one step of optimization routine
                feed_dict = self._get_feed_dict(
                    observations=observations,
                    input_data=input_data,
                    batch_indxs=batch_indxs)

                sess.run(self.train_step, feed_dict=feed_dict)
            epoch_time = time.time() - start

            # print training updates
            if epochs_display is not None and (
                    epoch % epochs_display == epochs_display - 1
                    or epoch == 0):
                self._train_print_updates(
                    sess, observations, input_data, train_indxs, test_indxs,
                    epoch, epoch_time)

            # save model checkpoints
            if epochs_ckpt is not None and (
                    epoch % epochs_ckpt == epochs_ckpt - 1
                    or epoch == 0):
                checkpoint_file = os.path.join(
                    output_dir, 'ckpts', str('epoch_%05g.ckpt' % epoch))
                self.checkpoint_model(sess, checkpoint_file=checkpoint_file,
                                      print_filepath=True)
                # store most recent checkpoint as model attribute
                self.checkpoint = checkpoint_file

            # save model summaries
            if epochs_summary is not None and (
                    epoch % epochs_summary == epochs_summary - 1
                    or epoch == 0):
                self._train_save_summaries(
                    sess, observations, input_data, train_indxs, test_indxs,
                    run_options, run_metadata, train_writer,
                    test_writer, epoch)

            # perform early stopping
            if opt_params['early_stop'] > 0:
                early_stop_params, stop_training = self._train_early_stop(
                    sess, observations, input_data, test_indxs,
                    early_stop_params, opt_params, epoch)
                if stop_training:
                    break

        # perform final checkpoint if not early stopping (handles case on own)
        if epochs_ckpt is np.inf and opt_params['early_stop'] == 0:
            checkpoint_file = os.path.join(
                output_dir, 'ckpts', str('epoch_%05g.ckpt' % epoch))
            self.checkpoint_model(sess, checkpoint_file=checkpoint_file,
                                  print_filepath=True)
            # store most recent checkpoint as model attribute
            self.checkpoint = checkpoint_file

        return epoch

    def _train_print_updates(
            self, sess, observations, input_data, train_indxs, test_indxs,
            epoch, epoch_time):

        # cost_train = self._get_cost(
        #     sess=sess,
        #     observations=observations,
        #     input_data=input_data,
        #     indxs=train_indxs)
        # cost_train /= len(train_indxs)
        cost_train = np.nan

        if test_indxs is not None:
            cost_test = self._get_cost(
                sess=sess,
                observations=observations,
                input_data=input_data,
                indxs=test_indxs)
            cost_test /= len(test_indxs)
        else:
            cost_test = np.nan

        # print testing info
        print('epoch %04d (%4.2f s):  avg train cost = %10.4f,  '
              'avg test cost = %10.4f' %
              (epoch, epoch_time, cost_train, cost_test))

        # print('epoch %04d (%4.2f s)' % (epoch, epoch_time))

    def _train_save_summaries(
            self, sess, observations, input_data, train_indxs, test_indxs,
            run_options, run_metadata, train_writer, test_writer, epoch):

        # evaluate summaries on all training indices
        feed_dict = self._get_feed_dict(
            observations=observations,
            input_data=input_data,
            batch_indxs=train_indxs)

        summary = sess.run(
            self.merge_summaries,
            feed_dict=feed_dict,
            options=run_options,
            run_metadata=run_metadata)
        if run_metadata is not None:
            # record compute time and memory usage of tf ops
            train_writer.add_run_metadata(
                run_metadata, 'epoch_%d' % epoch)
        train_writer.add_summary(summary, epoch)
        train_writer.flush()

        if test_writer is not None:

            # evaluate summaries on all test indices
            feed_dict = self._get_feed_dict(
                observations=observations,
                input_data=input_data,
                batch_indxs=test_indxs)

            summary = sess.run(
                self.merge_summaries,
                feed_dict=feed_dict,
                options=run_options,
                run_metadata=run_metadata)
            if run_metadata is not None:
                # record compute time and memory usage of tf ops
                test_writer.add_run_metadata(
                    run_metadata, 'epoch_%d' % epoch)
            test_writer.add_summary(summary, epoch)
            test_writer.flush()

    def _train_early_stop(
            self, sess, observations, input_data, test_indxs,
            early_stop_params, opt_params, epoch):

        # if you want to suppress that useless warning
        # with warnings.catch_warnings():
        #     warnings.simplefilter('ignore', category=RuntimeWarning)
        cost_test = self._get_cost(
            sess=sess,
            observations=observations,
            input_data=input_data,
            indxs=test_indxs)

        # unpack param dict
        prev_costs = early_stop_params['prev_costs']
        best_epoch = early_stop_params['best_epoch']
        best_cost = early_stop_params['best_cost']
        chkpted = early_stop_params['chkpted']
        output_dir = early_stop_params['output_dir']

        mean_before = np.nanmean(prev_costs)

        prev_costs = np.roll(prev_costs, 1)
        prev_costs[0] = cost_test
        mean_now = np.nanmean(prev_costs)

        delta = (mean_before - mean_now) / mean_before

        # to check and refine the condition on checkpointing best model
        # print(epoch, delta, 'delta condition:', delta < 1e-4)

        if cost_test < best_cost:
            # update best_cost and the epoch that it happened at
            best_cost = cost_test
            best_epoch = epoch
            # chkpt model if close to convergence
            if output_dir is not None:
                if delta < 1e-5:
                    checkpoint_file = os.path.join(
                        output_dir, 'best_model.ckpt')
                    self.checkpoint_model(sess, checkpoint_file)
                    self.checkpoint = checkpoint_file
                    chkpted = True

        stop_training = False
        if epoch > opt_params['early_stop'] and mean_now >= mean_before:
            # smoothed objective is starting to increase; exit training
            print('\n*** early stop criteria met...'
                  'stopping train now...')
            print('     ---> number of epochs used: %d,  '
                  'end cost: %04f' % (epoch, cost_test))
            print('     ---> best epoch: %d,  '
                  'best cost: %04f\n' % (best_epoch, best_cost))
            # restore saved variables into tf Variables
            if output_dir is not None and chkpted and best_epoch is not epoch \
                    and opt_params['early_stop_mode'] > 0:
                # restore checkpointed model from best epoch if not current
                self.restore_model(sess, checkpoint_file)
            elif output_dir is not None and not chkpted:
                # checkpoint model if it managed to slip by delta test
                checkpoint_file = os.path.join(output_dir, 'best_model.ckpt')
                self.checkpoint_model(sess, checkpoint_file)
                self.checkpoint = checkpoint_file
            stop_training = True

        # repack param dict
        early_stop_params['prev_costs'] = prev_costs
        early_stop_params['best_epoch'] = best_epoch
        early_stop_params['best_cost'] = best_cost
        early_stop_params['chkpted'] = chkpted

        return early_stop_params, stop_training

    def _get_cost(self, sess, observations, input_data, indxs):
        """Utility function to clean up code in `_train_adam` method"""
        raise NotImplementedError

    def _get_feed_dict(
            self, observations=None, input_data=None, batch_indxs=None):
        """Generates feed dict for training and other evaluation functions"""
        raise NotImplementedError

    def get_cost(self, observations, input_data):
        """User function for retrieving cost"""
        raise NotImplementedError

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
                if opt_params is None:
                    opt_params = {}
                opt_params = self.optimizer_defaults(opt_params)
                self.build_graph(opt_params=opt_params)

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
                    print_filepath=False,
                    opt_params=opt_params)
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
    def load_model(cls, save_file, opt_params=None):
        """
        Restore previously saved Model object

        Args:
            save_file (str): full path to saved model
            opt_params (dict): optimizer parameters for building graph

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

        # tell model where to find checkpoint file for restoring parameters
        checkpoint_file = constructor_inputs['checkpoint_file']
        del constructor_inputs['checkpoint_file']
        if checkpoint_file is None:
            print('Warning: model has not been checkpointed; restoring this '
                  'model will result in random parameters')

        # initialize model
        model = model_class(**constructor_inputs)
        model.checkpoint = checkpoint_file

        # build graph
        if model.graph is None:
            if opt_params is None:
                opt_params = {}
            opt_params = model.optimizer_defaults(opt_params)
            model.build_graph(opt_params=opt_params)

        return model

    @classmethod
    def optimizer_defaults(cls, opt_params):
        """
        Sets defaults for different optimizers

        Args:
            opt_params: dictionary with optimizer-specific parameters with the
                following keys:
            'learning_alg' (str, optional): 'adam' | 'adagrad' | 'adadelta'
                DEFAULT: 'adam'
            'use_gpu' (bool, optional): `True` to fit model on gpu.
                DEFAULT: True
            'batch_size' (int, optional): number of data points to
                use for each iteration of training.
                DEFAULT: 1
            'batch_size_test' (int, optional): number of data
                points to use for each iteration of finding test cost
                (useful if data is big)
                DEFAULT: None
            'epochs_training' (int, optional): max number of
                epochs.
                DEFAULT: 100
            'epochs_display' (int, optional): defines the number of epochs
                between updates to the console.
                DEFAULT: 0
            'epochs_ckpt' (int, optional): number of epochs between
                saving checkpoint files. If np.inf, will checkpoint final model
                before exiting training loop (and not using early stopping,
                which employs its own checkpointing scheme)
                DEFAULT: `None`
            'epochs_summary' (int, optional): number of epochs
                between saving network summary information.
                DEFAULT: `None`
            'early_stop_mode' (int, optional):
                0: don't chkpt, return the last model after loop break
                1: chkpt all models and choose the best one from the pool
                2: chkpt when training session is close to convergence
                DEFAULT: 0
            'early_stop' (int, optional): if greater than zero,
                training ends when the cost function evaluated on test_indxs is
                not lower than the maximum over that many previous checks.
                (Note that when early_stop > 0 and early_stop_mode = 1, early
                stopping will come in effect after epoch > early_stop pool
                size)
                DEFAULT: 0
            'run_diagnostics' (bool, optional): `True` to record
                compute time and memory usage of tensorflow ops during training
                and testing. `epochs_summary` must not be `None`.
                DEFAULT: `False`

            'adam' (dict, optional): dictionary of parameters for
                adam optimizer; see tf documentation for details
                'learning_rate' (float)
                    DEFAULT: 1e-3.
                'beta1' (float): 1st momentum term
                    DEFAULT: 0.9
                'beta2' (float): 2nd momentum term
                    DEFAULT: 0.999
                'epsilon' (float):
                    DEFAULT: 1e-8 (note normal Adam default is 1e-8)
            'adagrad' (dict, optional): dictionary of parameters for adagrad
                optimizer; see tf documentation for more details
                'learning_rate' (float)
                    DEFAULT: 1e-3
                'initial_accumulator_value' (float)
                    DEFAULT: 0.1
            'adadelta' (dict, optional): dictionary of parameters for adadelta
                optimizer; see tf documentation for more details
                'learning_rate' (float)
                    DEFAULT: 1e-3
                'rho' (float)
                    DEFAULT: 0.1
                'epsilon' (float)
                    DEFAULT: 1e-8

        """

        # Non-optimizer specific defaults
        if 'learning_alg' not in opt_params:
            opt_params['learning_alg'] = 'adam'
        if 'display' not in opt_params:
            opt_params['display'] = None
        if 'use_gpu' not in opt_params:
            opt_params['use_gpu'] = True
        if 'batch_size' not in opt_params:
            opt_params['batch_size'] = 1
        if 'batch_size_test' not in opt_params:
            opt_params['batch_size_test'] = None
        if 'epochs_training' not in opt_params:
            opt_params['epochs_training'] = 100
        if 'epochs_display' not in opt_params:
            opt_params['epochs_display'] = 0
        if 'epochs_ckpt' not in opt_params:
            opt_params['epochs_ckpt'] = None
        if 'epochs_summary' not in opt_params:
            opt_params['epochs_summary'] = None
        if 'early_stop_mode' not in opt_params:
            opt_params['early_stop_mode'] = 0
        if 'early_stop' not in opt_params:
            opt_params['early_stop'] = 0
        if 'run_diagnostics' not in opt_params:
            opt_params['run_diagnostics'] = False

        if opt_params['learning_alg'] is 'adam':
            if 'adam' not in opt_params:
                opt_params['adam'] = {}
            if 'learning_rate' not in opt_params['adam']:
                opt_params['adam']['learning_rate'] = 1e-3
            if 'beta1' not in opt_params['adam']:
                opt_params['adam']['beta1'] = 0.9
            if 'beta2' not in opt_params['adam']:
                opt_params['adam']['beta2'] = 0.999
            if 'epsilon' not in opt_params['adam']:
                opt_params['adam']['epsilon'] = 1e-8

        elif opt_params['learning_alg'] is 'adagrad':
            if 'adagrad' not in opt_params:
                opt_params['adagrad'] = {}
            if 'learning_rate' not in opt_params['adagrad']:
                opt_params['adagrad']['learning_rate'] = 1e-3
            if 'initial_accumulator_value' not in opt_params['adagrad']:
                opt_params['adagrad']['initial_accumulator_value'] = 0.9

        elif opt_params['learning_alg'] is 'adadelta':
            if 'adadelta' not in opt_params:
                opt_params['adadelta'] = {}
            if 'learning_rate' not in opt_params['adadelta']:
                opt_params['adadelta']['learning_rate'] = 1e-3
            if 'rho' not in opt_params['adadelta']:
                opt_params['adadelta']['rho'] = 0.95
            if 'epsilon' not in opt_params['adadelta']:
                opt_params['adadelta']['epsilon'] = 1e-8

        else:
            raise ValueError('"%s" learning alg not currently supported' %
                             opt_params['learning_alg'])

        return opt_params


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

    def build_graph(self, opt_params=None):
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

    def _get_cost(self, sess, observations, input_data, indxs):
        """Utility function to clean up code in training functions"""

        batch_size = 16  # fix for now
        num_batches = len(indxs) // batch_size

        # cycle through batches
        cost = 0.0
        for batch in range(num_batches):
            # get training indices for this batch
            batch_indxs = indxs[batch * batch_size: (batch + 1) * batch_size]
            feed_dict = self._get_feed_dict(
                observations=observations,
                input_data=input_data,
                batch_indxs=batch_indxs)
            cost += sess.run(self.objective, feed_dict=feed_dict) * len(
                batch_indxs)

        # last partial batch
        if num_batches * batch_size < len(indxs):
            batch_indxs = indxs[(batch + 1) * batch_size + 1: -1]
            feed_dict = self._get_feed_dict(
                observations=observations,
                input_data=input_data,
                batch_indxs=batch_indxs)
            cost += sess.run(self.objective, feed_dict=feed_dict) * len(
                batch_indxs)

        return cost / len(indxs)

    def _get_feed_dict(
            self, observations=None, input_data=None, batch_indxs=None):
        """Generates feed dict for training and other evaluation functions"""

        if batch_indxs is not None:
            feed_dict = {
                self.gen_net.obs_ph:
                    observations[batch_indxs, :, :],
                self.inf_net.input_ph:
                    observations[batch_indxs, :, :]}
        else:
            feed_dict = {
                self.gen_net.obs_ph: observations,
                self.inf_net.input_ph: observations}

        return feed_dict

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
            cost = self._get_cost(
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

    def build_graph(self, opt_params=None):
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
                self._define_optimizer(opt_params=opt_params)

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
                self._define_optimizer(opt_params=opt_params)

            # add additional ops
            # for saving and restoring models (initialized after var creation)
            self.saver = tf.train.Saver()
            # collect all summaries into a single op
            self.merge_summaries = tf.summary.merge_all()
            # add variable initialization op to graph
            self.init = tf.global_variables_initializer()
