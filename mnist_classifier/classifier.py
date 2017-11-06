# -*- coding: utf-8 -*-

"""
Tensorflow MNIST Classifier

"""

import tensorflow as tf

image_size = 28


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
        name=name)
    x = tf.nn.bias_add(x, b)
    x = act(x)
    return x


def maxpool(x, pool_size, stride, name):

    return tf.nn.max_pool(
        value=x,
        ksize=[1, pool_size, pool_size, 1],
        strides=[1, stride, stride, 1],
        padding='SAME',
        name=name)


def batch_norm(x):
    batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], keep_dims=True)
    scale = tf.Variable(tf.ones([x.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([x.get_shape()[-1]]))
    epsilon = 1e-3
    return tf.nn.batch_normalization(
        x,
        mean=batch_mean,
        variance=batch_var,
        scale=scale,
        variance_epsilon=epsilon,
        offset=beta
    )


def model(x):

    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, image_size, image_size, 1])

    with tf.name_scope('conv_block_1'):
        conv_filter1 = init_conv_filter(3, 32, 1)
        bias1 = init_bias(32)
        x1 = conv2d('conv_1', x_image, conv_filter1, bias1)
        conv_filter2 = init_conv_filter(3, 64, 32)
        bias2 = init_bias(64)
        x2 = conv2d('conv_2', x1, conv_filter2, bias2)
        x3 = maxpool(x2, 2, 2, name='pool1')

    with tf.name_scope('conv_block_2'):

        conv_filter3 = init_conv_filter(3, 128, 64)
        conv_filter4 = init_conv_filter(3, 256, 128)
        bias3 = init_bias(128)
        bias4 = init_bias(256)

        x4 = conv2d('conv_3', x3, conv_filter3, bias3)
        x5 = conv2d('conv_4', x4, conv_filter4, bias4)
        x6 = maxpool(x5, 2, 2, name='pool2')

    with tf.name_scope('conv_block_3'):

        conv_filter5 = init_conv_filter(3, 256, 256)
        bias5 = init_bias(256)
        x7 = conv2d('conv_5', x6, conv_filter5, bias5)
        x8 = maxpool(x7, 2, 2, name='pool3')

    with tf.name_scope('fc'):

        resize_height = 5

        x = tf.reshape(x8, [-1, 256*resize_height*resize_height])
        weights1 = init_weights(256*resize_height*resize_height, 256)
        bias6 = init_bias(256)
        weights2 = init_weights(256, 10)
        bias7 = init_bias(10)

        x = leaky_relu(tf.matmul(x, weights1) + bias6, 0.5)
        x = tf.nn.dropout(x, keep_prob=0.7)
        x = tf.matmul(x, weights2) + bias7

    return x









