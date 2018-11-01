import tensorflow as tf

from garage.misc.overrides import overrides
from garage.tf.core.networks import cnn, q_func
from garage.tf.q_functions import QFunction


class CNNQFunction(QFunction):
    """
    This class implements a q value network to predict q based on the input
    state and action. It uses an CNN to fit the function of Q(s, a).
    """

    def __init__(self,
                 env_spec,
                 filter_dims,
                 num_filters,
                 stride=1,
                 padding="SAME",
                 max_pooling=False,
                 pool_shape=(2, 2),
                 hidden_nonlinearity=tf.nn.relu,
                 output_nonlinearity=None):
        """
        Initialize class with multiple attributes.

        Args:
            env_spec: environment specification
            filter_dims: Dimension of the filters.
            num_filters: Number of filters.
            stride: The stride of the sliding window.
            name: Variable scope of the cnn.
            padding: The type of padding algorithm to use, from "SAME", "VALID".
            max_pooling: Boolean for using max pooling layer or not.
            pool_shape: Dimension of the pooling layer(s).
            hidden_nonlinearity: Activation function for
                        intermediate dense layer(s).
            output_nonlinearity: Activation function for
                        output dense layer.

        Return:
            The tf.Tensor of CNNQFunction.
        """

        super(CNNQFunction, self).__init__()

        self._action_dim = env_spec.action_space.n
        self._filter_dims = filter_dims
        self._hidden_nonlinearity = hidden_nonlinearity
        self._output_nonlinearity = output_nonlinearity
        self._num_filters = num_filters
        self._stride = stride
        self._padding = padding
        self._max_pooling = max_pooling
        self._pool_shape = pool_shape

    @overrides
    def _build_net(self, name, input, dueling=False, layer_norm=False):
        network = cnn(
            input_var=input,
            output_dim=512,
            filter_dims=self._filter_dims,
            hidden_nonlinearity=self._hidden_nonlinearity,
            output_nonlinearity=self._hidden_nonlinearity,
            num_filters=self._num_filters,
            stride=self._stride,
            padding=self._padding,
            max_pooling=self._max_pooling,
            pool_shape=self._pool_shape,
            name=name)

        return q_func(
            input_network=network,
            output_dim=self._action_dim,
            hidden_sizes=[256],
            name=name,
            hidden_nonlinearity=self._hidden_nonlinearity,
            output_nonlinearity=self._output_nonlinearity,
            layer_normalization=layer_norm,
            dueling=dueling)
