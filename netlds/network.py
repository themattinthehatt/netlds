"""Basic feed-forward neural network class"""

import tensorflow as tf


class Network(object):

    # use same data type throughout graph construction
    dtype = tf.float32

    # defaults for densely connected layers
    _layer_defaults = {
        'units': 30,
        'activation': 'tanh',
        'kernel_initializer': 'trunc_normal',
        'kernel_regularizer': None,
        'bias_initializer': 'zeros',
        'bias_regularizer': None}

    def __init__(self, output_dim, nn_params=None):

        if nn_params is None:
            # use all defaults with a single layer
            nn_params = [{}]

        self.output_dim = output_dim
        self._parse_nn_options(nn_params)
        self.layers = []

    def _parse_nn_options(self, nn_options):
        """Specify architecture of decoding network and set defaults"""

        self.nn_params = []
        for layer_num, layer_options in enumerate(nn_options):

            # start with _layer_defaults
            layer_params = dict(self._layer_defaults)
            # update default name
            layer_params['name'] = str('layer_%02i' % layer_num)

            # replace defaults with user-supplied options
            for key, value in layer_options.items():
                layer_params[key] = value

            # translate from strings to tf operations
            for key, value in layer_params.items():
                if value is not None:
                    # otherwise use default
                    if key is 'activation':
                        if value is 'exponential':
                            value = tf.exp
                        elif value is 'identity':
                            value = None
                        elif value is 'linear':
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
                        elif isinstance(value, str):
                            # allow actual initializers to be specified
                            raise ValueError(
                                '"%s" is not a valid string for specifying '
                                'a variable initializer' % value)
                # reassign (possibly new) value to key
                layer_params[key] = value

            # make sure output is correct size
            if layer_num == len(nn_options) - 1:
                layer_params['units'] = self.output_dim
            self.nn_params.append(dict(layer_params))

    def build_graph(self):

        for _, layer_params in enumerate(self.nn_params):
            self.layers.append(tf.layers.Dense(**layer_params))

    def apply_network(self, network_input):

        for _, layer in enumerate(self.layers):
            network_input = layer.apply(network_input)

        return network_input
