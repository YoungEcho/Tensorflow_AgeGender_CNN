# -*- coding: utf-8 -*-

'''
__author__ = 'youngtong'

'This is a python file for Age and Gender CNN net'
'''

import glob
import tensorflow as tf  # 0.12
from tensorflow.contrib.layers import *
from tensorflow.contrib.slim.python.slim.nets.inception_v3 import inception_v3_base
import numpy as np
from random import shuffle
import os
from AgeGender import DataProc

# Input Images Size
IMAGE_HEIGHT = 227
IMAGE_WIDTH = 227

'''
Traing &&& Lables Data
'''
data_set = DataProc.parse_data()
shuffle(data_set)

batch_size = 128
num_batch = len(data_set) // batch_size
NumEpoch = 400

lables_size = DataProc.lables_size

def resize_image(file_name):    
    jpg_data = tf.placeholder(dtype=tf.string)          ## string type
    decode_jpg = tf.image.decode_jpeg(jpg_data, channels=3)
    resize = tf.image.resize_images(decode_jpg, [IMAGE_HEIGHT, IMAGE_WIDTH])
    resize = tf.cast(resize, tf.uint8) / 255            ### """Casts a tensor to a new type.

    with tf.gfile.FastGFile(file_name, 'r') as f:
        image_data = f.read()
    with tf.Session() as sess:
        image = sess.run(resize, feed_dict={jpg_data: image_data})
    return image

pointer = 0

'''
# data_set[0]--imgsPath, data_set[1]--lables
'''
def get_next_batch(data_set, batch_size=128):
    global pointer          ### global var -- remeber the batch_size for Training
    batch_x = []
    batch_y = []
    for i in range(batch_size):
        batch_x.append(resize_image(data_set[pointer][0]))
        batch_y.append(data_set[pointer][1])
        pointer += 1
    return batch_x, batch_y

'''
nlabels---labels numbers, images----input imgs data
'''
def conv_net(nlabels, images, pkeep=1.0):
    weights_regularizer = tf.contrib.layers.l2_regularizer(0.0005)
    with tf.variable_scope("conv_net", "conv_net", [images]) as scope:
        with tf.contrib.slim.arg_scope([convolution2d, fully_connected], weights_regularizer=weights_regularizer,
                                       biases_initializer=tf.constant_initializer(1.),
                                       weights_initializer=tf.random_normal_initializer(stddev=0.005), trainable=True):
            with tf.contrib.slim.arg_scope([convolution2d],
                                           weights_initializer=tf.random_normal_initializer(stddev=0.01)):
                conv1 = convolution2d(images, 96, [7, 7], [4, 4], padding='VALID',
                                      biases_initializer=tf.constant_initializer(0.), scope='conv1')
                pool1 = max_pool2d(conv1, 3, 2, padding='VALID', scope='pool1')
                norm1 = tf.nn.local_response_normalization(pool1, 5, alpha=0.0001, beta=0.75, name='norm1')     ### LRN

                conv2 = convolution2d(norm1, 256, [5, 5], [1, 1], padding='SAME', scope='conv2')
                pool2 = max_pool2d(conv2, 3, 2, padding='VALID', scope='pool2')
                norm2 = tf.nn.local_response_normalization(pool2, 5, alpha=0.0001, beta=0.75, name='norm2')

                conv3 = convolution2d(norm2, 384, [3, 3], [1, 1], biases_initializer=tf.constant_initializer(0.),
                                      padding='SAME', scope='conv3')
                pool3 = max_pool2d(conv3, 3, 2, padding='VALID', scope='pool3')

                flat = tf.reshape(pool3, [-1, 384 * 6 * 6], name='reshape')
                full1 = fully_connected(flat, 512, scope='full1')
                drop1 = tf.nn.dropout(full1, pkeep, name='drop1')

                full2 = fully_connected(drop1, 512, scope='full2')
                drop2 = tf.nn.dropout(full2, pkeep, name='drop2')
    with tf.variable_scope('output') as scope:
        weights = tf.Variable(tf.random_normal([512, nlabels], mean=0.0, stddev=0.01), name='weights')    
        biases = tf.Variable(tf.constant(0.0, shape=[nlabels], dtype=tf.float32), name='biases')
        output = tf.add(tf.matmul(drop2, weights), biases, name=scope.name)
    return output


def training():
    X = tf.placeholder(dtype=tf.float32, shape=[batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3])
    Y = tf.placeholder(dtype=tf.int32, shape=[batch_size])

    logits = conv_net(lables_size, X)

    def optimizer(eta, loss_fn):
        global_step = tf.Variable(0, trainable=False)
        optz = lambda lr: tf.train.MomentumOptimizer(lr, 0.9)
        lr_decay_fn = lambda lr, global_step: tf.train.exponential_decay(lr, global_step, 100, 0.97, staircase=True)
        return tf.contrib.layers.optimize_loss(loss_fn, global_step, eta, optz, clip_gradients=4.,
                                               learning_rate_decay_fn=lr_decay_fn)

    def loss(logits, labels):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = cross_entropy_mean + 0.01 * sum(regularization_losses)
        loss_averages = tf.train.ExponentialMovingAverage(0.9)
        loss_averages_op = loss_averages.apply([cross_entropy_mean] + [total_loss])
        with tf.control_dependencies([loss_averages_op]):
            total_loss = tf.identity(total_loss)
        return total_loss
        # loss

    total_loss = loss(logits, Y)
    # optimizer
    train_op = optimizer(0.001, total_loss)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()


    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        summary_writer = tf.train.SummaryWriter('/tmp/log/', sess.graph)

        global pointer
        for epoch in range(NumEpoch):
        # while True:
            pointer = 0
            for batch in range(num_batch):
                batch_x, batch_y = get_next_batch(data_set, batch_size)
                _, loss_value = sess.run([train_op, total_loss], feed_dict={X: batch_x, Y: batch_y})
                print(epoch, batch, loss_value)
                # print("epoch %d, batch %d, training accuracy %g" % (epoch, batch, 1 - loss_value))
            
            summary_str = sess.run(summary_op)
            summary_writer.add_summary(summary_str, global_step=epoch)

            saver.save(sess, 'age.module' if DataProc.AGE == True else 'sex.module')


if __name__ == '__main__':
    training()

