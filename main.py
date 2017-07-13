# import classifier
# import pickle
#
# def main():
#     with open("./data_50000.pickle", "rb") as handle:
#         data = pickle.load(handle)
#     with open("./dictionary_5000.pickle", "rb") as handle:
#         dictionary_raw = pickle.load(handle)
#     test_data = data[45000:]
#     dictionary = [tup[1] for tup in dictionary_raw]
#     x = [tup[1] for tup in test_data]
#     y_word = [tup[0] for tup in test_data]
#     y = [dictionary.index(word) for word in y_word]
#
#     c = classifier.Classifier()
#
#     mat = c.classify(x)
#
#     success = 0
#     for i in range(len(mat)):
#         if y[i] in mat[i]:
#             success += 1
#
#     print("Success number: {0} Success percentage: {1}".format(success, success/len(mat)))
#
#
# if __name__ == '__main__':
#     main()

from classifier import Classifier
import pickle
import numpy as np
import time


def main():
    # create test data and labels
    with open('dictionary_5000.pickle', 'rb') as f:
        words_list = pickle.load(f)
    words_list = map(lambda t: (t[1], t[0]), words_list)
    words_dict = dict(words_list)
    with open('data_50000.pickle', 'rb') as f:
        testData = pickle.load(f)
        testData = testData[-5000:]
    Y, X = map(list, zip(*testData))
    Y_numbers = np.array([])
    for i in range(0, len(Y)):
        Y_numbers = np.append(Y_numbers, [words_dict[Y[i]]], axis=0)

    # get the Classifier labels for the test data
    theClassifier = Classifier()
    start = time.time()
    Y_hat = theClassifier.classify(X)
    end = time.time()
    print("time in seconds: %.2f" %(end - start))
    #print(Y_hat)

    # compare and get final score
    success = np.zeros(len(Y), dtype=bool)
    for x in range (0, 3):
        curr = Y_hat[:,x]
        curr =  curr == Y_numbers
        success = np.logical_or(success, curr)
    success = np.sum(success.astype(int)).astype(float)/len(success)
    score = success*100
    print("the score: %.3f" %score)


if __name__ == "__main__":
    main()
