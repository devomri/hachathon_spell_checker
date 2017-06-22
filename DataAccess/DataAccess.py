import pickle

class Z:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class DataAccess:
    def __init__(self):
        with open("./DataAccess/Data/data_50000.pickle", "rb") as handle:
            data = pickle.load(handle)
            self.train_data = self.get_train_data(data)
            self.test_data = self.get_test_data(data)
            self.validation_data = self.get_validation_data(data)

        with open("./DataAccess/Data/dictionary_5000.pickle", "rb") as handle:
            data = pickle.load(handle)
            self.dictionary = self.get_dictionary(data)

    def get_train_data(self, data):
        a = list()
        for word in data[:40000]:
            a.append(Z(word[0], word[1]))
        return a

    def get_test_data(self, data):
        b = list()
        for word in data[40000:49000]:
            b.append(Z(word[0], word[1]))
        return b

    def get_validation_data(self, data):
        c = list()
        for word in data[49000:]:
            c.append(Z(word[0], word[1]))
        return c

    def get_dictionary(self, data):
        return [tup[1] for tup in data]
