# -*- coding: utf-8 -*-

"""
Tensorflow MNIST Classifier

"""

import tensorflow as tf



def init_weights(input_size, output_size):
    """
    Init Weights for Layers
    :param input_size:
    :param output_size:
    :return:
    """
    return tf.Variable(tf.random_normal([input_size, output_size], stddev=0.01))


def init_conv_filter(filter_size, filter_nums, in_channels):
    """
    Init Convolution Kernel
    :param filter_size:
    :param filter_nums:
    :param in_channels:
    :return:
    """
    return tf.Variable(tf.random_normal([filter_size, filter_size, in_channels, filter_nums], stddev=0.01))


def init_bias(dim):

    return tf.Variable(tf.random_normal([dim], stddev=0.01))


def leaky_relu(x, alpha=0.3):

    return tf.maximum(x*alpha, x)


def conv2d(name, x, conv_filter, b, stride=1, act=leaky_relu):
    x = tf.nn.conv2d(
        input=x,
        filter=conv_filter,
        strides=[1, stride, stride, 1],
        padding='SAME',
        name=name
    )
    x = tf.nn.bias_add(x, b)
    x = act(x)
    return x



