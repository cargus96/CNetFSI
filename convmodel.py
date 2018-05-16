# -*- coding: utf-8 -*-

# Sample code to use string producer.
import numpy as np
import tensorflow as tf


def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


num_classes = 4
batch_size = 4


# --------------------------------------------------
#
#       DATA SOURCE
#
# --------------------------------------------------

def dataSource(paths, batch_size):
    min_after_dequeue = 10
    capacity = min_after_dequeue + 3 * batch_size

    example_batch_list = []
    label_batch_list = []

    for i, p in enumerate(paths):
        filename = tf.train.match_filenames_once(p)
        filename_queue = tf.train.string_input_producer(filename, shuffle=False)
        reader = tf.WholeFileReader()
        _, file_image = reader.read(filename_queue)
        if i == 0:
            image, label = tf.image.decode_jpeg(file_image), [0., 0., 1.]  # [one_hot(float(i), num_classes)]
        if i == 1:
            image, label = tf.image.decode_jpeg(file_image), [0., 1., 0.]  # [one_hot(float(i), num_classes)]
        if i == 2:
            image, label = tf.image.decode_jpeg(file_image), [1., 0., 0.]  # [one_hot(float(i), num_classes)]
        image = tf.image.resize_image_with_crop_or_pad(image, 80, 140)
        image = tf.reshape(image, [80, 140, 1])
        image = tf.to_float(image) / 256. - 0.5
        example_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=capacity,
                                                            min_after_dequeue=min_after_dequeue)
        example_batch_list.append(example_batch)
        label_batch_list.append(label_batch)

    example_batch = tf.concat(values=example_batch_list, axis=0)
    label_batch = tf.concat(values=label_batch_list, axis=0)
    return example_batch, label_batch


# --------------------------------------------------
#
#       MODEL
#
# --------------------------------------------------

def myModel(X, reuse=False):
    with tf.variable_scope('ConvNet', reuse=reuse):
        o1 = tf.layers.conv2d(inputs=X, filters=32, kernel_size=3, activation=tf.nn.relu)
        o2 = tf.layers.max_pooling2d(inputs=o1, pool_size=2, strides=2)
        o3 = tf.layers.conv2d(inputs=o2, filters=64, kernel_size=3, activation=tf.nn.relu)
        o4 = tf.layers.max_pooling2d(inputs=o3, pool_size=2, strides=2)
        h = tf.layers.dense(inputs=tf.reshape(o4, [batch_size * 3, 18 * 33 * 64]), units=1024, activation=tf.nn.relu)
        y = tf.layers.dense(inputs=h, units=3, activation=tf.nn.softmax)
    return y


example_batch_train, label_batch_train = dataSource(
    ['Signos/Dataset/0/*.JPG', 'Signos/Dataset/1/*.JPG', 'Signos/Dataset/2/*.JPG'],
    batch_size=batch_size)
example_batch_valid, label_batch_valid = dataSource(
    ['Signos/Dataset/0_Validacion/*.JPG', 'Signos/Dataset/1_Validacion/*.JPG', 'Signos/Dataset/2_Validacion/*.JPG'],
    batch_size=batch_size)

example_batch_test, label_batch_test = dataSource(
    ['Signos/Dataset/0_Test/*.JPG', 'Signos/Dataset/1_Test/*.JPG', 'Signos/Dataset/2_Test/*.JPG'],
    batch_size=batch_size)

example_batch_train_predicted = myModel(example_batch_train, reuse=False)
example_batch_valid_predicted = myModel(example_batch_valid, reuse=True)
example_batch_test_predicted = myModel(example_batch_test, reuse=True)

cost = tf.reduce_sum(tf.square(example_batch_train_predicted - label_batch_train))
cost_valid = tf.reduce_sum(tf.square(example_batch_valid_predicted - label_batch_valid))
cost_test = tf.reduce_sum(tf.square(example_batch_test_predicted - label_batch_test))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# --------------------------------------------------
#
#       TRAINING
#
# --------------------------------------------------

# Add ops to save and restore all the variables.

saver = tf.train.Saver()

with tf.Session() as sess:
    file_writer = tf.summary.FileWriter('./logs', sess.graph)

    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    for _ in range(680):
        sess.run(optimizer)
        if _ % 20 == 0:
            print("Iter:", _, "---------------------------------------------")
            print(sess.run(label_batch_valid))
            print(sess.run(example_batch_valid_predicted))
            print("Error:", sess.run(cost_valid))

    result = sess.run(example_batch_test_predicted)
    test = sess.run(label_batch_test)
    accuracy = 0
    for b, r in zip(result, test):
        if np.argmax(b) == np.argmax(r):
            accuracy += 1
    prec = accuracy / float(len(test))
    print("Accuracy was: ", prec)
    save_path = saver.save(sess, "./tmp/model.ckpt")
    print("Model saved in file: %s" % save_path)
    coord.request_stop()
    coord.join(threads)
