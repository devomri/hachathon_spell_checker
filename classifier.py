import pickle
import editdistance
from heapq import nsmallest


class Classifier(object):
    def __init__(self):
        with open("./data_50000.pickle", "rb") as handle:
            data = pickle.load(handle)
        with open("./dictionary_5000.pickle", "rb") as handle:
            self.dictionary = pickle.load(handle)
        # dictionary.sort()
        self.total_dictionary = [[i[1], i[1]] for i in self.dictionary] + data[:40000]

    def predict(self, word_with_typo):
        distance_array = [editdistance.eval(word_with_typo, val[1]) for val in self.total_dictionary]
        a = [(k, i) for i, k in enumerate(distance_array)]
        b = [val[1] for val in self.dictionary]
        return [b.index(self.total_dictionary[i[1]][0]) for i in nsmallest(3, a)]

    def classify(self, X):
        """
        Recieves a list of m corrupted words, and predicts 3 most likely corrections.
        :param X: A list of length m containing the words (strings)
        :return: y_hat - a matrix of size mx3. The i'th row has the prediction for the
                 i'th test sample, containing word indices of the correction candidates.
                 Word indices are specified in the file dictionary_5000.pickle
        """
        y_hat = [self.predict(m) for m in X]
        return y_hat