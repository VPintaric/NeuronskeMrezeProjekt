import numpy as np
import tensorflow as tf

import pdb

class NeuralNet:
    def __init__(self, config, param_delta=0.5, print_every=100):
        self.print_every = print_every

        self.X = tf.placeholder(tf.float64, [None, config[0]])
        self.Y_ = tf.placeholder(tf.float64, [None, config[len(config) - 1]])

        self.W = []
        self.b = []
        for i in range(1, len(config)):
            self.W.append(tf.Variable(initial_value=np.random.randn(config[i], config[i - 1]), expected_shape=[config[i], config[i - 1]], name="w" + str(i)))
            self.b.append(tf.Variable(initial_value=np.random.randn(config[i]), expected_shape=[config[i]], name="b" + str(i)))

        self.Y = self.X
        for i in range(len(self.W) - 1):
            self.Y = tf.nn.relu(tf.matmul(self.Y, self.W[i], transpose_b=True) + self.b[i])
        self.Y = tf.matmul(self.Y, self.W[len(self.W) - 1], transpose_b=True) + self.b[len(self.W) - 1]
        # do not use softmax! find something better (squeeze output in range 0-255?)

        self.loss = tf.reduce_sum(tf.square(self.Y_ - self.Y)) / (tf.size(self.X) / tf.constant(config[0]))

        self.trainer = tf.train.AdamOptimizer()    
        self.train_step = self.trainer.minimize(self.loss)

        self.session = tf.Session()

    def train(self, X, Y_, param_niter=1000):
        self.session.run(tf.global_variables_initializer())

        for i in range(param_niter):
            loss, _, w, b = self.session.run([self.loss, self.train_step, self.W, self.b], feed_dict = {self.X: X, self.Y_: Y_})
            if(i % self.print_every == 0):
                print("ITERATION " + str(i) + ", loss = " + str(loss))

    def eval(self, X):
        Y = self.session.run(self.Y, feed_dict = {self.X: X})
        return Y

    def count_params(self):
        trainables = tf.trainable_variables()

        nr_components = 0

        for trainable in trainables:
            print(trainable)
            shape = trainable.shape
            if len(shape) > 1:
                nr_components += shape[0].value * shape[1].value
            else:
                nr_components += shape[0].value

        print("Total trainable components = " + str(nr_components))