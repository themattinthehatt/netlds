"""Trainer class for training models"""

import numpy as np
import tensorflow as tf
import os
import time
import copy


class Trainer(object):

    # use same data type throughout graph construction
    dtype = tf.float32

    _optimizer_ops = {
        'adam': tf.train.AdamOptimizer,
        'adagrad': tf.train.AdagradOptimizer,
        'adadelta': tf.train.AdadeltaOptimizer}
    _data_types = ['train', 'test', 'validation']

    def __init__(self):
        """
        Set optimizer defaults

        See Trainer.parse_optimizer_options for descriptions
        """

        # optimizer info
        self.learning_alg = 'adam'
        self.optimizer = self._optimizer_ops[self.learning_alg]
        self.opt_params = {}
        for key, _ in self._optimizer_ops.items():
            self.opt_params[key] = self._set_optimizer_defaults(key)

        # training info
        self.epochs_training = 100
        self.batch_size = 1
        self.early_stop_mode = 0  # to get rid of
        self.early_stop = 0
        self.use_gpu = True

        # logging info
        self.epochs_display = None
        self.epochs_ckpt = None
        self.checkpoints_dir = None
        self.epochs_summary = None
        self.summaries_dir = None
        self.writers = {'train': None, 'test': None, 'validation': None}
        self.run_diagnostics = False

    def _build_optimizer(self, model):
        """Define one step of the optimization routine"""

        model.train_step = \
            self.optimizer(**self.opt_params[self.learning_alg]).minimize(
                model.objective)

    def _build_data_pipeline(
            self, num_time_pts, dim_obs, dim_input, dim_predictors):

        # one placeholder for all data
        with tf.variable_scope('observations_input'):
            self.y_true_ph = tf.placeholder(
                dtype=self.dtype,
                shape=[None, num_time_pts, sum(dim_obs)],
                name='output_ph')
        with tf.variable_scope('inference_input'):
            self.input_ph = tf.placeholder(
                dtype=self.dtype,
                shape=[None, num_time_pts, dim_input],
                name='input_ph')
        with tf.variable_scope('linear_predictors'):
            if dim_predictors is not None:
                self.linear_predictors_phs = []
                for pred, dim_pred in enumerate(dim_predictors):
                    self.linear_predictors_phs.append(
                        tf.placeholder(
                            dtype=self.dtype,
                            shape=[None, num_time_pts, dim_pred],
                            name='linear_pred_ph_%02i' % pred))

            else:
                self.linear_predictors_phs = None

        return self.y_true_ph, self.input_ph, self.linear_predictors_phs

    def train(
            self, model=None, data=None, indxs=None, opt_params=None,
            output_dir=None, checkpoint_file=None):
        """
        Model training function

        Args:
            model (Model object): model to train
            data (dict)
                'observations' (num_reps x num_time_pts x dim_obs numpy array)
                'inf_input' (num_reps x num_time_pts x dim_input numpy array,
                    optional): input to inference network; if using
                    observations as input, leave as `None`.
                'linear_predictors' (list): each entry is a
                    num_reps x num_time_pts x dim_lin_pred numpy array
            indxs (dict, optional): numpy arrays of indices
                'train', 'test', 'validation'; 'test' indices are used for
                early stopping if enabled
            opt_params (dict, optional): optimizer-specific parameters; see
                Model.parse_optimizer_options method for valid key-value pairs
            output_dir (str, optional): absolute path for saving checkpoint
                files and summary files; must be present if either
                `epochs_ckpt` or `epochs_summary` attribute is not `None`.
                If `output_dir` is not `None`, regardless of checkpoint
                or summary settings, the graph will automatically be saved.
                Must be present if early_stopping is desired to restore the
                best fit, otherwise it will restore the model at break point.
            checkpoint_file (str, optional): restore parameters from a
                checkpoint file before beginning training; if 'self', the file
                stored in model.checkpoint is used

        Raises:
            ValueError: If data dict does not contain observations
            ValueError: If data dict does not contain inference network input
            ValueError: If `epochs_ckpt` value is not `None` and `output_dir`
                is `None`
            ValueError: If `epochs_summary` is not `None` and `output_dir` is
                `None`
            ValueError: If `early_stop` > 0 and `test_indxs` is 'None'

        """

        # Check format of opt_params (and add some defaults)
        if opt_params is not None:
            self.parse_optimizer_options(**opt_params)

        if indxs is None:
            indxs = {
                'train': np.arange(data['observations'].shape[0]),
                'test': None, 'validation': None}
        if 'test' not in indxs:
            indxs['test'] = None
        if 'validation' not in indxs:
            indxs['validation'] = None

        # Check user-supplied data
        if 'observations' not in data:
            raise ValueError('must supply observation data')
        if 'inf_input' not in data:
            data['inf_input'] = data['observations']
        if 'linear_predictors' not in data:
            data['linear_predictors'] = []

        # Check values entered
        if self.epochs_ckpt is not None and output_dir is None:
            raise ValueError(
                'output_dir must be specified to save model')
        if self.epochs_summary is not None and output_dir is None:
            raise ValueError(
                'output_dir must be specified to save summaries')
        if self.early_stop > 0 and indxs['test'] is None:
            raise ValueError(
                'test indices must be specified for early stopping')

        # for specifying device
        if self.use_gpu:
            model.sess_config = tf.ConfigProto(device_count={'GPU': 1})
        else:
            model.sess_config = tf.ConfigProto(device_count={'GPU': 0})

        # build tensorflow computation graph
        if model.graph is None:
            model.build_graph()

        # intialize session
        with tf.Session(graph=model.graph, config=model.sess_config) as sess:

            # handle output directories
            if output_dir is not None:

                # checkpoint directory
                if self.epochs_ckpt is not None:
                    self.checkpoints_dir = \
                        os.path.join(output_dir, 'checkpoints')
                    if os.path.isdir(self.checkpoints_dir):
                        tf.gfile.DeleteRecursively(self.checkpoints_dir)
                    os.makedirs(self.checkpoints_dir)

                # summary directories
                if self.epochs_summary is not None:
                    self.summaries_dir = \
                        os.path.join(output_dir, 'summaries')
                    if os.path.isdir(self.summaries_dir):
                        tf.gfile.DeleteRecursively(self.summaries_dir)
                    os.makedirs(self.summaries_dir)

                    # remake summary directories
                    for _, data_type in enumerate(self._data_types):
                        if indxs[data_type] is not None:
                            summary_dir = os.path.join(
                                self.summaries_dir, data_type)
                            os.makedirs(summary_dir)
                            self.writers[data_type] = tf.summary.FileWriter(
                                summary_dir, graph=sess.graph)

            # initialize all parameters
            sess.run(model.init)

            # restore params from a previous session
            if checkpoint_file is 'self':
                if model.checkpoint is not None:
                    model.restore_model(sess, checkpoint_file=model.checkpoint)
                else:
                    raise ValueError('self.checkpoint is `None` file')
            elif checkpoint_file is not None:
                model.restore_model(sess, checkpoint_file=checkpoint_file)

            # start/resume training
            costs_train, costs_test = self._train_loop(
                model=model, data=data, sess=sess, indxs=indxs)

            # perform final checkpoint if not early stopping (handles on own)
            if self.epochs_ckpt is np.inf and self.early_stop == 0:
                checkpoint_file = os.path.join(
                    self.checkpoints_dir, str('epoch_%05g.ckpt' % self.epoch))
                model.checkpoint_model(
                    sess, checkpoint_file=checkpoint_file, print_filepath=True)
                # store most recent checkpoint as model attribute
                model.checkpoint = checkpoint_file

        return costs_train, costs_test

    def _train_loop(self, model, data=None, sess=None, indxs=None):
        """Training function for adam optimizer to clean up code in `train`"""

        if self.run_diagnostics:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
        else:
            run_options = None
            run_metadata = None

        if self.early_stop > 0:
            self.early_stop_params = {
                'prev_costs': np.multiply(np.ones(self.early_stop), np.nan),
                'best_epoch': 0,
                'best_cost': np.inf,
                'chkpted': False,
                'stop_training': False}

        # save initial model checkpoint
        if self.epochs_ckpt:
            checkpoint_file = os.path.join(self.checkpoints_dir, 'init.ckpt')
            model.checkpoint_model(
                sess, checkpoint_file=checkpoint_file, print_filepath=True)
            # store most recent checkpoint as model attribute
            model.checkpoint = checkpoint_file

        # store costs throughout training
        costs_train = []
        costs_test = []

        num_batches = indxs['train'].shape[0] // self.batch_size

        np.random.seed(model.np_seed)

        # start training loop
        self.epoch = np.nan
        for epoch in range(self.epochs_training):

            self.epoch = epoch

            # shuffle data before each pass
            train_indxs_perm = np.random.permutation(indxs['train'])

            # pass through dataset once
            start = time.time()
            for batch in range(num_batches):
                # get training indices for this batch
                batch_indxs = train_indxs_perm[
                    batch * self.batch_size:
                    (batch + 1) * self.batch_size]

                # one step of optimization routine
                feed_dict = self._get_feed_dict(
                    data=data, batch_indxs=batch_indxs)

                sess.run(model.train_step, feed_dict=feed_dict)
            epoch_time = time.time() - start

            # print training updates
            if self.epochs_display is not None and (
                    epoch % self.epochs_display == self.epochs_display - 1
                    or epoch == 0):
                cost_train, cost_test = self._train_print_updates(
                    sess, model, data, indxs, epoch_time)
                costs_train.append(cost_train)
                costs_test.append(cost_test)

            # save model checkpoints
            if self.epochs_ckpt is not None and (
                    epoch % self.epochs_ckpt == self.epochs_ckpt - 1):
                checkpoint_file = os.path.join(
                    self.checkpoints_dir, str('epoch_%05g.ckpt' % epoch))
                model.checkpoint_model(
                    sess, checkpoint_file=checkpoint_file, print_filepath=True)
                # store most recent checkpoint as model attribute
                model.checkpoint = checkpoint_file

            # save model summaries
            if self.epochs_summary is not None and (
                    epoch % self.epochs_summary == self.epochs_summary - 1
                    or epoch == 0):
                self._train_save_summaries(
                    sess, model, data, indxs, run_options, run_metadata)

            # perform early stopping
            if self.early_stop > 0:
                self._train_early_stop(sess, model, data, indxs)
                if self.early_stop_params['stop_training']:
                    break

        return costs_train, costs_test

    def _train_print_updates(
            self, sess, model, data, indxs, epoch_time):

        cost_train = self._get_cost(
            sess=sess, model=model, data=data, indxs=indxs['train'])
        cost_train /= len(indxs['train'])

        if indxs['test'] is not None:
            cost_test = self._get_cost(
                sess=sess, model=model, data=data, indxs=indxs['test'])
            cost_test /= len(indxs['test'])
        else:
            cost_test = np.nan

        # print testing info
        print('epoch %04d (%4.2f s):  avg train cost = %10.4f,  '
              'avg test cost = %10.4f' %
              (self.epoch, epoch_time, cost_train, cost_test))

        return cost_train, cost_test

    def _train_save_summaries(
            self, sess, model, data, indxs, run_options, run_metadata):

        # evaluate summaries on all indices
        for _, data_type in enumerate(self._data_types):

            if self.writers[data_type] is not None:

                feed_dict = self._get_feed_dict(
                    data=data, batch_indxs=indxs[data_type])

                summary = sess.run(
                    model.merge_summaries,
                    feed_dict=feed_dict,
                    options=run_options,
                    run_metadata=run_metadata)
                if run_metadata is not None:
                    # record compute time and memory usage of tf ops
                    self.writers[data_type].add_run_metadata(
                        run_metadata, 'epoch_%d' % self.epoch)
                self.writers[data_type].add_summary(summary, self.epoch)
                self.writers[data_type].flush()

    def _train_early_stop(self, model, sess, data, indxs):

        # if you want to suppress that useless warning
        # with warnings.catch_warnings():
        #     warnings.simplefilter('ignore', category=RuntimeWarning)
        cost_test = self._get_cost(
            sess=sess, model=model, data=data, indxs=indxs['test'])

        # unpack param dict
        prev_costs = self.early_stop_params['prev_costs']
        best_epoch = self.early_stop_params['best_epoch']
        best_cost = self.early_stop_params['best_cost']
        chkpted = self.early_stop_params['chkpted']

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
            best_epoch = self.epoch
            # chkpt model if close to convergence
            if self.checkpoints_dir is not None:
                if delta < 1e-5:
                    checkpoint_file = os.path.join(
                        self.checkpoints_dir, 'best_model.ckpt')
                    model.checkpoint_model(sess, checkpoint_file)
                    model.checkpoint = checkpoint_file
                    chkpted = True

        if self.epoch > self.early_stop and mean_now >= mean_before:
            # smoothed objective is starting to increase; exit training
            print('\n*** early stop criteria met...'
                  'stopping train now...')
            print('     ---> number of epochs used: %d,  '
                  'end cost: %04f' % (self.epoch, cost_test))
            print('     ---> best epoch: %d,  '
                  'best cost: %04f\n' % (best_epoch, best_cost))
            # restore saved variables into tf Variables
            if self.checkpoints_dir is not None and chkpted \
                    and best_epoch is not self.epoch \
                    and self.early_stop_mode > 0:
                # restore checkpointed model from best epoch if not current
                model.restore_model(sess, checkpoint_file)
            elif self.checkpoints_dir is not None and not chkpted:
                # checkpoint model if it managed to slip by delta test
                checkpoint_file = os.path.join(
                    self.checkpoints_dir, 'best_model.ckpt')
                model.checkpoint_model(sess, checkpoint_file)
                model.checkpoint = checkpoint_file
                self.early_stop_params['stop_training'] = True

        # repack param dict
        self.early_stop_params['prev_costs'] = prev_costs
        self.early_stop_params['best_epoch'] = best_epoch
        self.early_stop_params['best_cost'] = best_cost
        self.early_stop_params['chkpted'] = chkpted

    def _get_cost(self, sess, model, data, indxs):
        """Utility function to clean up code in training functions"""

        batch_size = 16  # fix for now
        num_batches = len(indxs) // batch_size

        # cycle through batches
        cost = 0.0
        for batch in range(num_batches):
            # get training indices for this batch
            batch_indxs = indxs[batch * batch_size: (batch + 1) * batch_size]
            feed_dict = self._get_feed_dict(data=data, batch_indxs=batch_indxs)
            cost += sess.run(model.objective, feed_dict=feed_dict) * len(
                batch_indxs)

        # last partial batch
        if num_batches * batch_size < len(indxs):
            batch_indxs = indxs[(batch + 1) * batch_size + 1: -1]
            feed_dict = self._get_feed_dict(data=data, batch_indxs=batch_indxs)
            cost += sess.run(model.objective, feed_dict=feed_dict) * len(
                batch_indxs)

        return cost / len(indxs)

    def _get_feed_dict(self, data=None, batch_indxs=None):
        """Generates feed dict for training and other evaluation functions"""

        if batch_indxs is not None:
            feed_dict = {
                self.y_true_ph: data['observations'][batch_indxs, :, :],
                self.input_ph: data['inf_input'][batch_indxs, :, :]}
            for indx_, data_ in enumerate(data['linear_predictors']):
                feed_dict[self.linear_predictors_phs[indx_]] = \
                    data_[batch_indxs, :, :]
        else:
            feed_dict = {
                self.y_true_ph: data['observations'],
                self.input_ph: data['input_data']}
            for indx_, data_ in enumerate(data['linear_predictors']):
                feed_dict[self.linear_predictors_phs[indx_]] = data_

        return feed_dict

    @classmethod
    def _set_optimizer_defaults(cls, learning_alg):

        if learning_alg is 'adam':
            opt_params = {
                'learning_rate': 1e-2,
                'beta1': 0.9,
                'beta2': 0.999,
                'epsilon': 1e-8}
        elif learning_alg is 'adagrad':
            opt_params = {
                'learning_rate': 1e-2,
                'initial_accumulator_value': 0.9}
        elif learning_alg is 'adadelta':
            opt_params = {
                'learning_rate': 1e-3,
                'rho': 0.95,
                'epsilon': 1e-8}

        return opt_params

    def parse_optimizer_options(self, **kwargs):
        """
        Sets defaults for different optimizers

        Args:
            learning_alg (str): 'adam' | 'adagrad' | 'adadelta'
            use_gpu (bool): `True` to fit model on gpu.
            batch_size (int): number of data points to use for each iteration
                of training.
            epochs_training (int): max number of epochs.
            epochs_display (int, optional): defines the number of epochs
                between updates to the console.
            epochs_ckpt (int): number of epochs between saving checkpoint
                files. If np.inf, will checkpoint final model before exiting
                training loop (and not using early stopping, which employs its
                own checkpointing scheme)
            epochs_summary (int): number of epochs between saving network
                summary information.
            early_stop_mode (int):
                0: don't chkpt, return the last model after loop break
                1: chkpt all models and choose the best one from the pool
                2: chkpt when training session is close to convergence
            early_stop (int): if greater than zero, training ends when the cost
                function evaluated on test_indxs is not lower than the maximum
                over that many previous checks. (Note that when early_stop > 0
                and early_stop_mode = 1, early stopping will come in effect
                after epoch > early_stop pool size)
            run_diagnostics (bool): `True` to record compute time and memory
                usage of tensorflow ops during training and testing.
                `epochs_summary` must not be `None`.
            adam (dict): dictionary of parameters for adam optimizer; see tf
                documentation for details
                'learning_rate' (float)
                'beta1' (float): 1st momentum term
                'beta2' (float): 2nd momentum term
                'epsilon' (float):
            adagrad (dict): dictionary of parameters for adagrad optimizer; see
                tf documentation for more details
                'learning_rate' (float)
                'initial_accumulator_value' (float)
            adadelta (dict): dictionary of parameters for adadelta optimizer;
                see tf documentation for more details
                'learning_rate' (float)
                'rho' (float)
                'epsilon' (float)

        """

        # iterate through key-value pairs
        for key, value in kwargs.items():
            if isinstance(value, dict):
                # optimizer-specific defaults
                param_dict = copy.copy(self.opt_params[key])
                for keyd, valued in value.items():
                    param_dict[keyd] = valued
                self.opt_params[key] = param_dict
            else:
                # non-optimizer specific defaults; lazy type checking for now
                if isinstance(value, type(getattr(self, key))) or \
                        getattr(self, key) is None:
                    setattr(self, key, value)
                # update optimizer
                if key is 'learning_alg':
                    self.optimizer = self._optimizer_ops[self.learning_alg]
