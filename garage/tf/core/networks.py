"""
Different networks with tensorflow as the only dependency.

The module contains MLP and CNN implementation.
It aims to replace existing implementation of network classes
(garage.tf.core.network), which is under development.
"""
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.gen_array_ops import broadcast_to


def mlp(input_var,
        output_dim,
        hidden_sizes,
        name,
        hidden_nonlinearity=tf.nn.relu,
        hidden_w_init=tf.contrib.layers.xavier_initializer(),
        hidden_b_init=tf.zeros_initializer(),
        output_nonlinearity=None,
        output_w_init=tf.contrib.layers.xavier_initializer(),
        output_b_init=tf.zeros_initializer(),
        layer_normalization=False):
    """
    MLP function.

    Args:
        input_var: Input tf.Tensor to the MLP.
        output_dim: Dimension of the network output.
        hidden_sizes: Output dimension of dense layer(s).
        name: variable scope of the mlp.
        hidden_nonlinearity: Activation function for
                    intermediate dense layer(s).
        hidden_w_init: Initializer function for the weight
                    of intermediate dense layer(s).
        hidden_b_init: Initializer function for the bias
                    of intermediate dense layer(s).
        output_nonlinearity: Activation function for
                    output dense layer.
        output_w_init: Initializer function for the weight
                    of output dense layer(s).
        output_b_init: Initializer function for the bias
                    of output dense layer(s).
        layer_normalization: Bool for using layer normalization or not.

    Return:
        The output tf.Tensor of the MLP
    """
    with tf.variable_scope(name):
        l_hid = input_var
        for idx, hidden_size in enumerate(hidden_sizes):
            l_hid = tf.layers.dense(
                inputs=l_hid,
                units=hidden_size,
                activation=hidden_nonlinearity,
                kernel_initializer=hidden_w_init,
                bias_initializer=hidden_b_init,
                name="hidden_{}".format(idx))
            if layer_normalization:
                l_hid = tf.contrib.layers.layer_norm(l_hid)

        l_out = tf.layers.dense(
            inputs=l_hid,
            units=output_dim,
            activation=output_nonlinearity,
            kernel_initializer=output_w_init,
            bias_initializer=output_b_init,
            name="output")
    return l_out


def parameter(input_var,
              length,
              initializer=tf.zeros_initializer(),
              dtype=tf.float32,
              trainable=True,
              name="parameter"):
    """
    Paramter function that creates  variables that could be
    broadcast to a certain shape to match with input var.

    Args:
        input_var: Input tf.Tensor.
        length: Integer dimension of the variables.
        initializer: Initializer of the variables.
        dtype: Data type of the variables.
        trainable: Whether these variables are trainable.
        name: variable scope of the variables.

    Return:
        A tensor of broadcasted variables
    """
    with tf.variable_scope(name):
        p = tf.get_variable(
            "parameter",
            shape=(length, ),
            dtype=dtype,
            initializer=initializer,
            trainable=trainable)

        ndim = input_var.get_shape().ndims
        broadcast_shape = tf.concat(
            axis=0, values=[tf.shape(input_var)[:ndim - 1], [length]])
        p_broadcast = broadcast_to(p, shape=broadcast_shape)
        return p_broadcast


def cnn(input_var,
        output_dim,
        filter_dims,
        num_filters,
        stride,
        name,
        padding="SAME",
        max_pooling=False,
        pool_shape=(2, 2),
        hidden_nonlinearity=tf.nn.relu,
        hidden_w_init=tf.contrib.layers.xavier_initializer(),
        hidden_b_init=tf.zeros_initializer(),
        output_nonlinearity=None,
        output_w_init=tf.contrib.layers.xavier_initializer(),
        output_b_init=tf.zeros_initializer()):
    """
    CNN function.

    Args:
        input_var: Input tf.Tensor to the CNN.
        output_dim: Dimension of the network output.
        filter_dims: Dimension of the filters.
        num_filters: Number of filters.
        stride: The stride of the sliding window.
        name: Variable scope of the cnn.
        padding: The type of padding algorithm to use, from "SAME", "VALID".
        max_pooling: Boolean for using max pooling layer or not.
        pool_shape: Dimension of the pooling layer(s).
        hidden_nonlinearity: Activation function for
                    intermediate dense layer(s).
        hidden_w_init: Initializer function for the weight
                    of intermediate dense layer(s).
        hidden_b_init: Initializer function for the bias
                    of intermediate dense layer(s).
        output_nonlinearity: Activation function for
                    output dense layer.
        output_w_init: Initializer function for the weight
                    of output dense layer(s).
        output_b_init: Initializer function for the bias
                    of output dense layer(s).

    Return:
        The output tf.Tensor of the CNN
    """

    def conv(input_var, name, filter_size, num_filter, strides,
             padding="SAME"):

        # based on 'NHWC' data format
        # [batch, height, width, channel]

        # channel from input
        input_shape = input_var.get_shape()[-1].value
        # [filter_height, filter_width, in_channels, out_channels]
        w_shape = [filter_size, filter_size, input_shape, num_filter]
        b_shape = [1, 1, 1, num_filter]

        with tf.variable_scope(name):
            w = tf.get_variable('w', w_shape, initializer=hidden_w_init)
            b = tf.get_variable('b', b_shape, initializer=hidden_b_init)

            return tf.nn.conv2d(
                input_var, w, strides=strides, padding=padding) + b

    # based on 'NHWC' data format
    # [batch, height, width, channel]

    strides = [1, stride, stride, 1]
    pool_shape = [1, pool_shape[0], pool_shape[1], 1]

    with tf.variable_scope(name):
        h = input_var
        for index, (filter_dim, num_filter) in enumerate(
                zip(filter_dims, num_filters)):
            h = hidden_nonlinearity(
                conv(h, 'h{}'.format(index), filter_dim, num_filter, strides,
                     padding))
            if max_pooling:
                h = tf.nn.max_pool(
                    h, ksize=pool_shape, strides=strides, padding=padding)
        # convert conv to dense
        # height * width * channel
        num_h = np.prod([v.value for v in h.get_shape()[1:]])
        h = tf.reshape(h, [-1, num_h])
        h = tf.layers.dense(
            inputs=h,
            units=output_dim,
            activation=output_nonlinearity,
            kernel_initializer=output_w_init,
            bias_initializer=output_b_init,
            name="output")

        return h


def q_func(input_network,
           output_dim,
           hidden_sizes,
           name,
           hidden_nonlinearity=tf.nn.relu,
           hidden_w_init=tf.contrib.layers.xavier_initializer(),
           hidden_b_init=tf.zeros_initializer(),
           output_nonlinearity=None,
           output_w_init=tf.contrib.layers.xavier_initializer(),
           output_b_init=tf.zeros_initializer(),
           layer_normalization=False,
           dueling=False):
    """
    Q-Function.
    Useful for building q-function with another network as input, e.g. CNN.

    Args:
        input_network: Input tf.Tensor to the Q-Function.
        output_dim: Dimension of the network output.
        hidden_sizes: Output dimension of dense layer(s).
        name: variable scope of the Q-Function.
        hidden_nonlinearity: Activation function for
                    intermediate dense layer(s).
        hidden_w_init: Initializer function for the weight
                    of intermediate dense layer(s).
        hidden_b_init: Initializer function for the bias
                    of intermediate dense layer(s).
        output_nonlinearity: Activation function for
                    output dense layer.
        output_w_init: Initializer function for the weight
                    of output dense layer(s).
        output_b_init: Initializer function for the bias
                    of output dense layer(s).
        layer_normalization: Bool for using layer normalization or not.
        dueling: Boolean for using dueling network or not.

    Return:
        The output tf.Tensor of the Q-Function.
    """
    with tf.variable_scope(name):
        with tf.variable_scope("action_value"):
            l_hid = input_network
            for idx, hidden_size in enumerate(hidden_sizes):
                l_hid = tf.layers.dense(
                    inputs=l_hid,
                    units=hidden_size,
                    activation=hidden_nonlinearity,
                    kernel_initializer=hidden_w_init,
                    bias_initializer=hidden_b_init,
                    name="action_value")
                if layer_normalization:
                    l_hid = tf.contrib.layers.layer_norm(l_hid)
            action_out = tf.layers.dense(
                inputs=l_hid,
                units=output_dim,
                activation=output_nonlinearity,
                kernel_initializer=output_w_init,
                bias_initializer=output_b_init,
                name="output_action_value")

        if dueling:
            with tf.variable_scope("state_value"):
                l_hid = input_network
                for idx, hidden_size in enumerate(hidden_sizes):
                    l_hid = tf.layers.dense(
                        inputs=l_hid,
                        units=hidden_size,
                        activation=hidden_nonlinearity,
                        kernel_initializer=hidden_w_init,
                        bias_initializer=hidden_b_init,
                        name="state_value")
                    if layer_normalization:
                        l_hid = tf.contrib.layers.layer_norm(l_hid)
                state_out = tf.layers.dense(
                    inputs=l_hid,
                    units=output_dim,
                    activation=output_nonlinearity,
                    kernel_initializer=output_w_init,
                    bias_initializer=output_b_init,
                    name="output_state_value")
            action_out_mean = tf.reduce_mean(action_out, 1)
            # calculate the advantage of performing certain action over other action
            # in a particular state
            action_out_advantage = action_out - tf.expand_dims(
                action_out_mean, 1)
            q_out = state_out + action_out_advantage
        else:
            q_out = action_out

    return q_out
