import numpy as np
import tensorflow as tf

from garage.tf.core.networks import cnn
from tests.fixtures import TfGraphTestCase


class TestCNN(TfGraphTestCase):
    def setUp(self):
        super(TestCNN, self).setUp()
        self.obs_input = np.ones((2, 5, 4, 3))
        input_shape = self.obs_input.shape[1:]  # height, width, channel
        self.hidden_nonlinearity = tf.nn.relu

        self._input_ph = tf.placeholder(
            tf.float32, shape=(None, ) + input_shape, name="input")

        self._output_shape = 2

        # We build a default cnn
        with tf.variable_scope("CNN"):
            self.cnn = cnn(
                input_var=self._input_ph,
                output_dim=self._output_shape,
                filter_dims=(3, 3, 3),
                num_filters=(32, 64, 128),
                stride=1,
                name="cnn1",
                hidden_nonlinearity=self.hidden_nonlinearity)

        self.sess.run(tf.global_variables_initializer())

    def test_shape(self):
        result = self.sess.run(
            self.cnn, feed_dict={self._input_ph: self.obs_input})
        assert result.shape[1] == self._output_shape
