import matplotlib.pyplot as plt
import numpy as np

from svm_iu import SVMClassifier

np.random.seed(seed=30)


def TestLinearData(total_size=200, dim=2, k=1, b=0):
    allData = np.random.uniform(0, 1, total_size * dim)
    allData = allData.reshape((total_size, dim)).tolist()
    classes = []
    for d in allData:
        if d[1] >= k * d[0] + b:
            classes.append(1)
        else:
            classes.append(-1)

    classifier = SVMClassifier(C=3.0)
    DEBUG_CASE = 5
    for i in range(len(classes)):
        print("add {}".format(i))
        if i == DEBUG_CASE:
            print("here!")
        classifier.IncrementPair(allData[i], classes[i])

    correct_cnt = 0
    for i in range(len(classes)):
        if classifier(allData[i]) * classes[i] <= 0:
            print(i)
        correct_cnt += 1 if classifier(allData[i]) * classes[i] > 0 else 0
    print(correct_cnt)

    return


def TestDependingTimeOnSize(final_size=128):

    return


if __name__ == "__main__":
    TestLinearData()
