import tensorflow as tf
import numpy as np
from cleverhans.model import Model
from cleverhans.attacks import FastGradientMethod
import os

import math

import sgld_tf

def make_graph(optimize_method, learning_rate=0.01):
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))
    model = sgld_tf.CNN(n_filters=32, n_dense=1024)
    preds = model.get_probs(x)
    logits = model.get_logits(x)
    loss = tf.losses.softmax_cross_entropy(y, logits)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    optimizer = sgld_tf.SGLD(learning_rate = 0.01, lrdecay=1)
    if optimize_method == "kfac":
        optimizer = tf.contrib.kfac.optimizer.KfacOptimizer(
            learning_rate=learning_rate,
            damping=0.001,
            cov_ema_decay=0.95,
            layer_collection=model.layer_collection)
    elif optimize_method == "tfkfac":
        optimizer = sgld_tf.KfacOptimizer(
            learning_rate=learning_rate,
            damping=0.001,
            cov_ema_decay=0.95,
            layer_collection=model.layer_collection)
    elif optimize_method == "sgd":
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=tf.train.exponential_decay(
                learning_rate,
                tf.train.get_global_step(),
                1000, 1))
    elif optimize_method == "sgld":
        optimizer = sgld_tf.SGLD(
            lrdecay=1,
            decay_interval=1000,
            learning_rate=learning_rate)
    elif optimize_method == "psgld":
        optimizer = sgld_tf.pSGLD(
            learning_rate=learning_rate,
            lrdecay=1,
            decay_interval=1000
        )
    elif optimize_method == "ksgld":
        optimizer = sgld_tf.kSGLDOpt(
            lrdecay=1,
            decay_interval=1000,
            learning_rate=learning_rate,
            decay=0.9,
            layer_collection = model.layer_collection)

    train_step = optimizer.minimize(loss, global_step=global_step)
    saver = tf.train.Saver()
    sess = tf.Session()
    fgsm = FastGradientMethod(model, sess=sess)
    adv_x = fgsm.generate(x, eps=.3)
    adv_preds = model.get_probs(adv_x)

    return x, y, preds, logits, loss, train_step, saver, sess, adv_preds

def bayes_evaluate(files, root, sess, batch_size, X_test, Y_test, x, y, preds,
                   dunno = False):
    preds_acc = []
    saver = tf.train.Saver()

    for model_file in files:
        saver.restore(sess, root + model_file)
        acc, pre, lab = sgld_tf.evaluate(sess, batch_size, X_test, Y_test, x, y, preds)
        preds_acc.append(pre)

    final_preds = np.zeros_like(preds_acc[0])
    for pred in preds_acc:
        final_preds += pred

    if dunno:
        dunno_filter = np.max(final_preds, axis=1)/len(files) > dunno
        print('dunno percent ', np.mean(dunno_filter))
        final_acc = np.mean(np.argmax(final_preds, axis=1)[dunno_filter] == np.argmax(lab, axis=1)[dunno_filter])

        return final_acc

    final_acc = np.mean(np.argmax(final_preds, axis=1) == np.argmax(lab, axis=1))

    return final_acc


def bayes_uncertainty(files, root, sess, batch_size, X_test, Y_test, x, y, preds,
                   dunno = False):
    preds_acc = []
    saver = tf.train.Saver()


    for model_file in files:
        saver.restore(sess, root + model_file)
        acc, pre, lab = sgld_tf.evaluate(sess, batch_size, X_test, Y_test, x, y, preds)
        preds_acc.append(pre)

    final_preds = np.zeros_like(preds_acc[0])
    for pred in preds_acc:
        final_preds += pred

    if dunno:
        dunno_filter = np.max(final_preds, axis=1)/len(files) > dunno
        print('dunno percent ', np.mean(dunno_filter))
        final_acc = np.mean(np.argmax(final_preds, axis=1)[dunno_filter] == np.argmax(lab, axis=1)[dunno_filter])

        return final_acc


    return final_preds/len(files)

def evaluate(sess, batch_size, X_test, Y_test, x, y, preds):
    nb_batches_eval = int(math.ceil(float(len(X_test)) / batch_size))
    correct_preds = tf.equal(tf.argmax(y, axis=-1), tf.argmax(preds, axis=-1))
    accuracy = 0.0
    preds_acc = []
    labels_acc = []

    with sess.as_default():
        for batch in range(nb_batches_eval):
            start = batch*batch_size
            end = min((batch+1)*batch_size, len(X_test))
            feed_dict = {x: X_test[start:end],
                            y: Y_test[start:end]}
            curr_corr_preds = correct_preds.eval(feed_dict=feed_dict)
            preds_acc.append(preds.eval(feed_dict=feed_dict))
            labels_acc.append(y.eval(feed_dict=feed_dict))
            accuracy += curr_corr_preds.sum()
    return accuracy/len(X_test), np.vstack(preds_acc), np.vstack(labels_acc)

def train(n_epochs, sess, batch_size, X_train, Y_train, X_test, Y_test, x, y, preds, train_step, loss, save_root):
    if not os.path.exists(save_root): os.makedirs(save_root)
    saver = tf.train.Saver(max_to_keep=100000)
    with sess.as_default():
        tf.global_variables_initializer().run()
        i = 0
        for epoch in range(n_epochs):
            print('epoch: ', epoch)
            nb_batches = int(math.ceil(float(len(X_train)) / batch_size))
            for batch in range(nb_batches):
                i=i+1
                start = batch*batch_size
                end = min((batch+1)*batch_size, len(X_train))

                feed_dict = {x: X_train[start:end],
                            y: Y_train[start:end]}
                train_step.run(feed_dict=feed_dict)
                if i%20 == 0:
                    saver.save(sess, save_root + 'model.ckpt-' + str(i))

            acc, _, _ = sgld_tf.evaluate(sess, batch_size, X_test, Y_test, x, y, preds)
            print("eval acc", acc)




class CNN(Model):
    def __init__(self, n_filters, n_dense):
        self.layer_names = ['reshape',
                           'conv1',
                           'pool1',
                           'conv2',
                           'pool2',
                           'pool2_flat',
                           'dense',
                           'logits',
                           'probs']

        self.n_filters = n_filters
        self.kernel_size = [5, 5]
        self.pool_size = [2, 2]
        self.n_dense = n_dense


        self.conv1 = tf.layers.Conv2D(filters=self.n_filters,
                              kernel_size=self.kernel_size,
                              padding="same")
        self.conv2 = tf.layers.Conv2D(filters=2*self.n_filters,
                              kernel_size=self.kernel_size,
                              padding="same")
        self.dense1 = tf.layers.Dense(units=self.n_dense)
        self.dense2 = tf.layers.Dense(units=10)

    def fprop(self, x):
        reshaped_input = tf.reshape(x, [-1, 28, 28, 1])
        conv1_pre = self.conv1(reshaped_input)
        conv1_out = tf.nn.relu(conv1_pre)

        pool1 = tf.layers.max_pooling2d(inputs=conv1_out, pool_size=[2, 2], strides=2)

        conv2_pre = self.conv2(pool1)
        conv2_out = tf.nn.relu(conv2_pre)

        pool2 = tf.layers.max_pooling2d(inputs=conv2_out, pool_size=[2, 2], strides=2)
        pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

        dense_pre = self.dense1(pool2_flat)
        dense_out = tf.nn.relu(dense_pre)

        logits = self.dense2(dense_out)
        probs = tf.nn.softmax(logits)

        self.layer_collection = tf.contrib.kfac.layer_collection.LayerCollection()
        self.layer_collection.register_conv2d((self.conv1.kernel, self.conv1.bias), (1, 1, 1, 1), "SAME", reshaped_input, conv1_pre)
        self.layer_collection.register_conv2d((self.conv2.kernel, self.conv2.bias), (1, 1, 1, 1), "SAME", pool1, conv2_pre)
        self.layer_collection.register_fully_connected((self.dense1.kernel, self.dense1.bias), pool2_flat, dense_pre)
        self.layer_collection.register_fully_connected((self.dense2.kernel, self.dense2.bias), dense_out, logits)
        self.layer_collection.register_categorical_predictive_distribution(logits, name="logits")


        return {'reshape': reshaped_input,
               'conv1': conv1_out,
               'pool1': pool1,
               'conv2': conv2_out,
               'pool2': pool2,
               'pool2_flat': pool2_flat,
               'dense': dense_out,
               'logits': logits,
               'probs': probs}

