import numpy as np
import matplotlib.pyplot as plt
from iu_svm import SVMClassifier
import time


def absmin(x):
    i = np.argmin(x)
    sign = 1 if x[i] >= 0 else 0
    newArr = [abs(x_i) * sign for x_i in x]
    return np.min(newArr)


def VizalizeData(x_set, y_set, title, xlabel="x", ylabel="y"):
    indexes1 = np.argwhere(y_set == -1)
    indexes2 = np.argwhere(y_set == 1)

    x_subset1 = np.squeeze(x_set[indexes1])
    x_subset2 = np.squeeze(x_set[indexes2])

    plt.figure()
    plt.scatter(x_subset1[:, 0], x_subset1[:, 1], c="r", marker="o")
    plt.scatter(x_subset2[:, 0], x_subset2[:, 1], c="b", marker="*")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    return


def MakeTest(
    classifier: SVMClassifier,
    x_set,
    y_set,
    plotTimes,
    plotResults,
    printAccuracy,
    naming="linear",
):
    times = []

    for i in range(len(x_set)):
        start = time.time()
        classifier.IncrementPair(x_set[i], y_set[i])
        times.append(time.time() - start)

    if printAccuracy:
        print(
            "S: {}, E: {}, R: {}, accuracy: {}".format(
                len(classifier.S),
                len(classifier.E),
                len(classifier.O),
                1 - (len(classifier.E) / x_set.shape[0]),
            )
        )

    if plotResults:
        x_low, x_high = np.min(x_set[:, 0]), np.max(x_set[:, 0])
        y_low, y_high = np.min(x_set[:, 1]), np.max(x_set[:, 1])

        x = np.arange(x_low, x_high, 0.05)
        y = np.arange(y_low, y_high, 0.05)
        xgrid, ygrid = np.meshgrid(x, y)

        z = np.zeros(xgrid.shape)
        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                z[i, j] = classifier(np.array([x[j], y[i]]))

        indexes1 = np.argwhere(y_set == -1)
        indexes2 = np.argwhere(y_set == 1)

        # x_subset1 = np.squeeze(x_set[indexes1])
        # x_subset2 = np.squeeze(x_set[indexes2])

        x_subset1S = np.squeeze(
            x_set[np.array(list(set(indexes1.flatten().tolist()) & set(classifier.S)))]
        )
        x_subset1R = np.squeeze(
            x_set[np.array(list(set(indexes1.flatten().tolist()) & set(classifier.O)))]
        )

        x_subset2S = np.squeeze(
            x_set[np.array(list(set(indexes2.flatten().tolist()) & set(classifier.S)))]
        )
        x_subset2R = np.squeeze(
            x_set[np.array(list(set(indexes2.flatten().tolist()) & set(classifier.O)))]
        )

        if len(x_subset1S.shape) == 1:
            x_subset1S = np.array([x_subset1S])

        if len(x_subset1R.shape) == 1:
            x_subset1R = np.array([x_subset1R])

        if len(x_subset2S.shape) == 1:
            x_subset2S = np.array([x_subset2S])

        if len(x_subset2R.shape) == 1:
            x_subset2R = np.array([x_subset2R])

        plt.figure()
        plt.scatter(
            x_subset1R[:, 0], x_subset1R[:, 1], c="#dd2222", marker="*", alpha=0.5
        )
        plt.scatter(
            x_subset2R[:, 0], x_subset2R[:, 1], c="#2222dd", marker="*", alpha=0.5
        )

        if len(list(set(indexes1.flatten().tolist()) & set(classifier.E))) != 0:
            x_subset1E = np.squeeze(
                x_set[
                    np.array(list(set(indexes1.flatten().tolist()) & set(classifier.E)))
                ]
            )

            if len(x_subset1E.shape) == 1:
                x_subset1E = np.array([x_subset1E])

            plt.scatter(
                x_subset1E[:, 0], x_subset1E[:, 1], c="#aa5555", marker="x", alpha=0.3
            )

        if len(list(set(indexes1.flatten().tolist()) & set(classifier.E))) != 0:
            x_subset2E = np.squeeze(
                x_set[
                    np.array(list(set(indexes2.flatten().tolist()) & set(classifier.E)))
                ]
            )

            if len(x_subset2E.shape) == 1:
                x_subset2E = np.array([x_subset2E])

            plt.scatter(
                x_subset2E[:, 0], x_subset2E[:, 1], c="#5555aa", marker="x", alpha=0.3
            )

        plt.contour(xgrid, ygrid, z, levels=[-1, 0, 1])
        plt.scatter(x_subset2S[:, 0], x_subset2S[:, 1], c="#0000ff", marker="o", s=60)
        plt.scatter(x_subset1S[:, 0], x_subset1S[:, 1], c="#ff0000", marker="o", s=60)
        plt.title("Результат классификации, C={}".format(classifier.C))
        plt.xlim([x_low, x_high])
        plt.xlim([y_low, y_high])
        plt.xlabel("x")
        plt.ylabel("y")
        plt.savefig(naming + "_res_c={}.png".format(classifier.C))
        pass

    if plotTimes:
        plt.figure()
        plt.plot(times)
        plt.title("Время добавления от количества элементов, C={}".format(classifier.C))
        plt.xlabel("Число элементов")
        plt.ylabel("Время (сек)")
        plt.savefig(naming + "_time_c={}.png".format(classifier.C))
    return
