import numpy as np
import collections
import Model.DeepLearningMNIST as dl


def main():
    a = dl.DeepLearningMNIST()
    a.train_model()

    print("Hello World")


if __name__ == '__main__':
    main()
