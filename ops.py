import math
import numpy as np

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import ops
import tensorflow.contrib.layers as tflayers

from utils import *
from ncausalconv import *

def batch_norm(input, is_training=True, name="batch_norm"):
    x = tflayers.batch_norm(inputs=input,
                            scale=True,
                            is_training=is_training,
                            trainable=True,
                            reuse=None)
    return x

def instance_norm(input, name="instance_norm", is_training=True):
    with tf.variable_scope(name):
        depth = input.get_shape()[2]
        scale = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input, axes=[1], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input-mean)*inv
        return scale*normalized + offset


def conv2d(input_, output_dim, ks=4, s=2, stddev=0.02, padding='SAME', name="conv2d", activation_fn=None):
    with tf.variable_scope(name):
        return slim.conv2d(input_, output_dim, ks, s, padding=padding, activation_fn=activation_fn,
                           weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                           biases_initializer=None)


def deconv2d(input_, output_dim, ks=4, s=2, stddev=0.02, name="deconv2d"):
    with tf.variable_scope(name):
        input_ = tf.image.resize_images(images=input_,
                                        size=tf.shape(input_)[1:3] * s,
                                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)  # That is optional
        return conv2d(input_=input_, output_dim=output_dim, ks=ks, s=1, padding='SAME')


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)


def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [input_.get_shape()[-1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
                               initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias

def TempBlock(input_, outchannels_, layer_index=0, ks=2, s=1, dilation_rate_=1, dropout_=0.2,
              name='TemBlock', is_training=True):
    with tf.variable_scope(name):
        tb = TemporalBlock(outchannels_, ks, s, dilation_rate_,
                           dropout_, name="tblock_{}".format(layer_index))

        return tb(input_, training=is_training)
