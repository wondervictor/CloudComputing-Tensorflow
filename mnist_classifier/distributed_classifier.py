# -*- coding: utf-8 -*-

"""
Distributed MNIST Classifier

"""

import tensorflow as tf
import collections
import mnist_data
from classifier import model

Config = collections.namedtuple('Config', 'lr, batch_size, epoches, data_dir')


num_workers = 3
num_ps = 2
train_data_size = 60000


def distributed_model(config):

    tf.app.flags.DEFINE_string("job_name", "", "Enter 'ps' or 'worker' ")
    tf.app.flags.DEFINE_integer("task_index", 0, "Index of the task within the jobs")
    tf.app.flags.DEFINE_bool("async", True, "Async or Sync Train")
    tf.app.flags.DEFINE_string("ps_hosts", "", "Comma-separated list of hostname:port pairs")
    tf.app.flags.DEFINE_string("worker_hosts", "", "Comma-separated list of hostname:port pairs")
    tf.app.flags.DEFINE_string("data_dir", "./data/", "Data directory")
    FLAGS = tf.app.flags.FLAGS

    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")

    mnist = mnist_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    cluster = tf.train.ClusterSpec({
        "worker": worker_hosts,
        "ps": ps_hosts
    })

    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

    if FLAGS.job_name == 'ps':
        server.join()
    else:
        with tf.device(tf.train.replica_device_setter(
            worker_device='/job:worker/task:%d' % FLAGS.task_index,
            cluster=cluster
        )):
            global_step = tf.Variable(0, name="global_step", trainable=False)
            x = tf.placeholder(tf.float32, [None, 784])
            label = tf.placeholder(tf.float32, [None, 10])

            pred_y = model(x)

            with tf.name_scope('loss'):
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=pred_y)
                cross_entropy = tf.reduce_mean(cross_entropy)
                tf.summary.scalar('loss', cross_entropy)

            with tf.name_scope('adam_optimizer'):
                optimizer = tf.train.AdamOptimizer(config.lr)

            with tf.name_scope('accuracy'):
                correct_prediction = tf.equal(tf.argmax(pred_y, 1), tf.argmax(label, 1))
                correct_prediction = tf.cast(correct_prediction, tf.float32)
                accuracy = tf.reduce_mean(correct_prediction)
                tf.summary.scalar('accuracy', accuracy)

            with tf.name_scope('grads_and_vars'):

                grads_and_vars = optimizer.compute_gradients(cross_entropy)

            if FLAGS.async:

                # 异步模式
                train_op = optimizer.apply_gradients(grads_and_vars)

            else:

                rep_op = tf.train.SyncReplicasOptimizer(
                    optimizer,
                    replicas_to_aggregate=len(worker_hosts),
                    total_num_replicas=len(worker_hosts),
                    use_locking=True
                )

                train_op = rep_op.apply_gradients(grads_and_vars, global_step=global_step)
                init_token_op = rep_op.get_init_tokens_op()
                chief_queue_runner = rep_op.get_chief_queue_runner()

            init_op = tf.global_variables_initializer()
            saver = tf.train.Saver()
            summary_op = tf.summary.merge_all()

            sv = tf.train.Supervisor(
                is_chief=(FLAGS.task_index == 0),
                logdir='./summary_log/',
                init_op=init_op,
                summary_op=None,
                saver=saver,
                global_step=global_step,
                save_model_secs=60
            )

            summary_writer = tf.summary.FileWriter('./summary_log/')

            with sv.prepare_or_wait_for_session(server.target) as sess:

                if FLAGS.task_index == 0 and not FLAGS.async:

                    sv.start_queue_runners(sess, [chief_queue_runner])
                    sess.run(init_token_op)

                for i in xrange(config.epoches):
                    loss = 0
                    for j in xrange(train_data_size / config.batch_size):

                        batch = mnist.train.next_batch(config.batch_size)
                        sess.run(train_op, feed_dict={x: batch[0], label: batch[1]})
                        loss += sess.run(cross_entropy, feed_dict={x: batch[0], label: batch[1]})
                        if j % 5 == 0:
                            train_accuracy = accuracy.eval(session=sess, feed_dict={x: batch[0], label: batch[1]})
                            print('[Epoch: %d Sample: %d] training accuracy: %g loss: %s' % (
                            i, (j + 1) * config.batch_size, train_accuracy, loss / (j + 1)))
                    summary = sess.run(summary_op)
                    summary_writer.add_summary(summary, global_step)
                    train_accuracy = accuracy.eval(session=sess,
                                                   feed_dict={x: mnist.test.images, label: mnist.test.labels})
                    print("[Epoch: %s] Test Accuracy: %s" % (i + 1, train_accuracy))
                sv.stop()


if __name__ == '__main__':

    config = Config(
        lr=0.001,
        batch_size=16,
        epoches=10,
        data_dir=''
    )

    distributed_model(config)






