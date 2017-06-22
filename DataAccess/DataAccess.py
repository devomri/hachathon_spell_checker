import pickle

class Z:
	def __init__(self, x, y):
		self.x  = x
		self.y = y



class DataAccess:

    def __init__(self):
    	dataContent = pickle.load(open("./Data/data_50000.pickle", "rb"))
        close("./Data/data_50000.pickle")
        self.train_data = self.get_train_data(list(dataContent))
        self.test_data = self.get_test_data(list(dataContent))
        self.validation_data = self.get_validation_data(list(dataContent))

    def get_train_data(self, data):
        a = list()
        for word in data[:40000]:
        	a.append(new Z(word[0], word[1]))
        return a


    def get_train_data(self, data):
        b = list()
        for word in data[40000:49000]:
        	b.append(new Z(word[0], word[1]))
        return b

    def get_train_data(self, data):
        c = list()
        for word in data[49000:]:
        	c.append(new Z(word[0], word[1]))
        return c