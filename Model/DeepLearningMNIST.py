import numpy as np
import tensorflow as tf
import string
import DataAccess.DataAccess as data_access


class DeepLearningMNIST:
    def __init__(self):
        self.last_batch_index = 0
        self.dal = data_access.DataAccess()
        self.train = self.convert_data(self.dal.train_data)
        self.test = self.convert_data(self.dal.test_data)

    def convert_data(self, data):
        x = list()
        y = list()

        for z in data:
            x.append(self.convert_word_to_vec(z.x))
            y.append(self.convert_label_to_vec(z.y))

        return {
            "x": np.array(x),
            "y": np.array(y)
        }

    def convert_word_to_vec(self, word):
        vec = np.zeros(26)
        for c in word.lower():
            i = string.ascii_lowercase.index(c)
            vec[i] += 1

        return vec

    def convert_label_to_vec(self, label):
        vec = np.zeros(5000)
        i = self.dal.dictionary.index(label)
        vec[i] = 1

        return vec

    def get_next_batch(self, batch_size):
        start_index = self.last_batch_index
        end_index = start_index + batch_size
        batch_x = self.train["x"][start_index:end_index]
        batch_y = self.train["y"][start_index:end_index]

        self.last_batch_index += batch_size

        return batch_x, batch_y

    def train_model(self):
        x = tf.placeholder(tf.float32, [None, 26])
        W = tf.Variable(tf.zeros([26, 5000]))
        b = tf.Variable(tf.zeros([5000]))
        y = tf.matmul(x, W) + b

        y_ = tf.placeholder(tf.float32, [None, 5000])

        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

        sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        # Train
        for _ in range(40):
            batch_xs, batch_ys = self.get_next_batch(1000)
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

        # Test trained model
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print(sess.run(accuracy, feed_dict={x: self.test["x"],
                                            y_: self.test["y"]}))
