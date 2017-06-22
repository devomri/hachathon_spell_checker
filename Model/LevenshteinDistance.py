import pickle
from heapq import nsmallest
import editdistance

class LevenshteinDistance:
    def __init__(self, matching_idx):
        self.matching_idx = matching_idx

    def match_best_candidates(sample , dictionary, matching_index):
        candidates = nsmallest(matching_index, ((k,i) for i,k in enumerate([editdistance.eval(sample, val[1]) for val in dictionary])))
        return [i[1] for i in candidates]

    def calc_LS(self, data, dictionary):
        matching_idx = self.matching_idx
        num_of_success = [(x_y[0] in [dictionary[index][1] for index in LevenshteinDistance.match_best_candidates(x_y[1], dictionary, matching_idx)]) for x_y in data].count(True)
        return num_of_success/len(data)


if __name__ == '__main__':
    data = pickle.load(open("../data_50000.pickle", "rb"))
    dictionary = pickle.load(open("../dictionary_5000.pickle", "rb"))
    print(LevenshteinDistance(3).calc_LS(data, dictionary))
