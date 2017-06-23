import classifier
import pickle

def main():
    with open("./data_50000.pickle", "rb") as handle:
        data = pickle.load(handle)
    test_data = data[49000:]
    x = [tup[1] for tup in test_data]
    y = [tup[0] for tup in test_data]

    c = classifier.Classifier()

    mat = c.classify(x)

    success = 0
    for i in range(len(mat)):
        if y[i] in mat[i]:
            success += 1

    print("Success rate: " + str(success/len(mat)))


if __name__ == '__main__':
    main()
