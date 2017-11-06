# -*- coding: utf-8 -*-

"""
Tensorflow MNIST Classifier

"""

import tensorflow as tf
from classifier import model
import mnist_data
import collections


train_data_size = 60000
test_data_size = 10000

Config = collections.namedtuple('Config', 'lr, batch_size, epoches, data_dir')


def train(config):

    mnist = mnist_data.read_data_sets(config.data_dir, one_hot=True)

    x = tf.placeholder(tf.float32, [None, 784])
    label = tf.placeholder(tf.float32, [None, 10])

    pred_y = model(x)

    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=pred_y)

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(config.lr).minimize(cross_entropy)

    with tf.name_scope('accuracy'):

        correct_prediction = tf.equal(tf.argmax(pred_y, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        for i in xrange(config.epoches):
            loss = 0
            for j in xrange(train_data_size/config.batch_size):

                batch = mnist.train.next_batch(config.batch_size)
                train_step.run(feed_dict={x: batch[0], label: batch[1]})
                sess += sess.run(cross_entropy, feed_dict={x: batch[0], label: batch[1]})
                if j % 100:
                    train_accuracy = accuracy.eval(feed_dict={x: batch[0], label: batch[1]})
                    print('[step %d] training accuracy: %g loss: %s' % (i, train_accuracy, loss/(j+1)))

            train_accuracy = accuracy.eval(feed_dict={x: mnist.test.images, label: mnist.test.labels})
            print("[Epoch: %s] Test Accuracy: %s" % (i + 1, train_accuracy))


if __name__ == '__main__':

    config = Config(
        lr=0.001,
        epoches=20,
        data_dir='',
        batch_size=32
    )

    train(config)


