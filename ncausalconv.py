import tensorflow as tf
import math


class NonCausalConv1D(tf.layers.Conv1D):
    def __init__(self, filters,
                kernel_size,
                strides=1,
                dilation_rate=1,
                activation=None,
                use_bias=True,
                kernel_initializer=None,
                bias_initializer=tf.zeros_initializer(),
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                bias_constraint=None,
                trainable=True,
                name=None,
                **kwargs):
        super(NonCausalConv1D, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='valid',
            data_format='channels_last',
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            trainable=trainable,
            name=name,
            **kwargs
        )

    def call(self, inputs):
        padding = math.ceil((self.kernel_size[0]-1) * self.dilation_rate[0] / 2)

        inputs = tf.pad(inputs, tf.constant([(0, 0), (1, 1), (0, 0)]) * padding)
        return super(NonCausalConv1D, self).call(inputs)

class TemporalBlock(tf.layers.Layer):
    def __init__(self, n_outputs, kernel_size, strides, dilation_rate, dropout=0.2,
                 trainable=True, name=None, dtype=None, activity_regularizer=None, **kwargs):
        super(TemporalBlock, self).__init__(
            trainable=trainable, dtype=dtype,
            activity_regularizer=activity_regularizer,
            name=name, **kwargs
        )
        self.dropout = dropout
        self.dialation_rate = dilation_rate
        self.strides = strides
        self.n_outputs = n_outputs
        self.conv1 = NonCausalConv1D(
            n_outputs, kernel_size, strides=strides,
            dilation_rate=dilation_rate, activation=tf.nn.relu,
            name='conv1'
        )
        self.conv2 = NonCausalConv1D(
            n_outputs, kernel_size, strides=strides,
            dilation_rate=dilation_rate, activation=tf.nn.relu,
            name='conv2'
        )
        self.down_sample = None

    def build(self, input_shape):
        channel_dim = 2
        self.dropout1 = tf.layers.Dropout(self.dropout, [tf.constant(1), tf.constant(1), tf.constant(self.n_outputs)])
        self.dropout2 = tf.layers.Dropout(self.dropout, [tf.constant(1), tf.constant(1), tf.constant(self.n_outputs)])

        if input_shape[channel_dim] != self.n_outputs:
            self.down_sample = tf.layers.Dense(self.n_outputs, activation=None)
        self.built = True

    def call(self, inputs, training=True):
        x = tf.contrib.layers.instance_norm(self.conv1(inputs))
        x = self.dropout1(x, training=training)
        x = tf.contrib.layers.instance_norm(self.conv2(x))
        x = self.dropout2(x, training=training)
        if self.down_sample is not None:
            inputs = self.down_sample(inputs)
        return tf.nn.leaky_relu(x + inputs)
