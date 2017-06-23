import pickle
from heapq import nsmallest
import editdistance

class LevenshteinDistanceOptimated:
    def __init__(self, matching_idx):
        self.matching_idx = matching_idx

    @staticmethod
    def match_best_candidates(sample , dictionary, matching_index):
        a = [(k, i) for i, k in enumerate([editdistance.eval(sample, val[1]) for val in dictionary])]
        # candidates=[]
        # for i in a:
        #     if[dictionary[i[1]][0] not in candidates]:
        #         candidates.append(i)
        return [i[1] for i in nsmallest(matching_index, a)]


    def calc_LS(self, data, dictionary):
        matching_idx = self.matching_idx
        num_of_success = [(x_y[0] in [dictionary[index][0] for index in LevenshteinDistanceOptimated.match_best_candidates(x_y[1], dictionary, matching_idx)]) for x_y in data].count(True)
        return num_of_success/len(data)


if __name__ == '__main__':
    data = pickle.load(open("../data_50000.pickle", "rb"))
    dictionary = pickle.load(open("../dictionary_5000.pickle", "rb"))
    dictionary.sort()
    new_dictionary = [[i[1],i[1]] for i in dictionary]+ data[:40000]
    # print(new_dictionary)
    print(LevenshteinDistanceOptimated(3).calc_LS(data[40000:45000], new_dictionary))