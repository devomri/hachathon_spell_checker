import classifier
import pickle

def main():
    with open("./data_50000.pickle", "rb") as handle:
        data = pickle.load(handle)
    with open("./dictionary_5000.pickle", "rb") as handle:
        dictionary_raw = pickle.load(handle)
    test_data = data[45000:]
    dictionary = [tup[1] for tup in dictionary_raw]
    x = [tup[1] for tup in test_data]
    y_word = [tup[0] for tup in test_data]
    y = [dictionary.index(word) for word in y_word]

    c = classifier.Classifier()

    mat = c.classify(x)

    success = 0
    for i in range(len(mat)):
        if y[i] in mat[i]:
            success += 1

    print("Success number: {0} Success percentage: {1}".format(success, success/len(mat)))


if __name__ == '__main__':
    main()
