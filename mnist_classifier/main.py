# -*- coding: utf-8 -*-

"""
Tensorflow MNIST Classifier

"""

import tensorflow as tf
from classifier import model
import mnist_data
import collections
import argparse

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
        cross_entropy = tf.reduce_mean(cross_entropy)
        tf.summary.scalar('loss', cross_entropy)

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(config.lr).minimize(cross_entropy)

    with tf.name_scope('accuracy'):

        correct_prediction = tf.equal(tf.argmax(pred_y, 1), tf.argmax(label, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)
        tf.summary.scalar('accuracy', accuracy)

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        # merged = tf.summary.merge_all()
        # writer = tf.summary.FileWriter("logs/", sess.graph)
        sess.run(tf.global_variables_initializer())

        for i in xrange(config.epoches):
            loss = 0
            for j in xrange(train_data_size/config.batch_size):

                batch = mnist.train.next_batch(config.batch_size)
                train_step.run(feed_dict={x: batch[0], label: batch[1]})
                loss += sess.run(cross_entropy, feed_dict={x: batch[0], label: batch[1]})#sess.run(tf.reduce_mean(sess.run(cross_entropy, feed_dict={x: batch[0], label: batch[1]})))
                if j % 5 == 0:
                    train_accuracy = accuracy.eval(feed_dict={x: batch[0], label: batch[1]})
                    print('[Epoch: %d Sample: %d] training accuracy: %g loss: %s' % (i, (j+1)*config.batch_size, train_accuracy, loss/(j+1)))

            train_accuracy = accuracy.eval(feed_dict={x: mnist.test.images, label: mnist.test.labels})
            print("[Epoch: %s] Test Accuracy: %s" % (i + 1, train_accuracy))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data/', help='MNIST Data Directory')
    parser.add_argument('--epoches', type=int, default=20, help='Training epoches')

    a = parser.parse_args()

    config = Config(
        lr=0.001,
        epoches=a.epoches,
        data_dir=a.data_dir,
        batch_size=32
    )

    train(config)


