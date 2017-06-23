import numpy as np
import tensorflow as tf
import string
import DataAccess.DataAccess as data_access
from tensorflow.contrib import slim


class DeepLearningMNIST:
    def __init__(self):
        print("Start normalizing data")
        self.last_batch_index = 0
        self.dal = data_access.DataAccess()
        self.train = self.convert_data(self.dal.train_data)
        self.test = self.convert_data(self.dal.test_data)
        print("End normalizing data")

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
        vec = np.zeros(26 * 26)
        # for c in word.lower():
        #     i = string.ascii_lowercase.index(c)
        #     vec[i] += 1
        lowered_word = word.lower()
        for i in range(len(lowered_word) - 1):
            first_letter = lowered_word[i]
            second_letter = lowered_word[i + 1]

            ascii_first = string.ascii_lowercase.index(first_letter)
            ascii_second = string.ascii_lowercase.index(second_letter)

            vec_index = 26 * ascii_first + ascii_second
            vec[vec_index] += 1

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
        print("Start model network")
        x = tf.placeholder(tf.float32, [None, 26 * 26])
        fc1 = slim.fully_connected(x, 500,
                                   weights_initializer=tf.contrib.layers.xavier_initializer(),
                                   activation_fn=tf.nn.relu)
        y = slim.fully_connected(fc1, 5000,
                                 weights_initializer=tf.contrib.layers.xavier_initializer(),
                                 activation_fn=tf.nn.relu)

        y_ = tf.placeholder(tf.float32, [None, 5000])

        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

        sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()

        print("Start Training...")
        for b in range(41):
            print("Batch number " + str(b))
            batch_xs, batch_ys = self.get_next_batch(1000)
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

        print("Start testing...")
        success = 0
        test_size = len(self.test["x"])
        for i in range(test_size):
            prediction = sess.run(y, feed_dict={x: self.test["x"][i].reshape(1,26 * 26),
                                                y_: self.test["y"][i].reshape(1,5000)})
            prediction = list(prediction[0])
            prediction_clone = list(prediction)
            prediction_clone.sort()
            first_index = prediction.index(prediction_clone[-1])
            second_index = prediction.index(prediction_clone[-2])
            third_index = prediction.index(prediction_clone[-3])

            true_label_index = np.argmax(self.test["y"][i])
            if true_label_index == first_index or true_label_index == second_index or true_label_index == third_index:
                success += 1

        print("Success: {0} Accuracy is: {1}".format(str(success), str(success / test_size)))
