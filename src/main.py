import numpy as np
import pandas as pd
from iu_svm import SVMClassifier
import time
from utils import *
from kernels import *
import random

np.random.seed(seed=30)

total_size = 500  # Размер выборок
dim = 2  # Размерность пространства (для простоты визуализации - R^2)
C = [
    0.1,
    1,
    100,
]  # C - параметры регуляризации (ПРИ C=3, C=1 падает хех)

sepWidth = 0.1
alpha, b = np.pi / 4, [3, 3]

# generate data
linearSepData = np.random.uniform(-1, 1, total_size * dim)
linearSepData = linearSepData.reshape((total_size, dim))

# generate target points
classes = []
for d in linearSepData:
    if d[1] >= 0:
        d[1] += sepWidth
        classes.append(1)
    else:
        d[1] -= sepWidth
        classes.append(-1)
linearClasses = np.array(classes)

# transform data
rotationM = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
linearSepData = np.sum(
    [linearSepData @ rotationM.T, np.array(b * total_size).reshape((total_size, dim))],
    axis=0,
)

# VizalizeData(linearSepData, linearClasses, "Линейно разделимые данные")

sepWidth = 0.25
sepDist = 4

# generate data
randRad = np.random.uniform(-np.pi, np.pi, total_size)
randDist = np.random.uniform(sepWidth, sepDist * 2 - sepWidth, total_size)

# generate target points
radPoints = []
classes = []

for i, d in enumerate(randDist):
    if d >= sepDist:
        d += sepWidth
        classes.append(1)
    else:
        d -= sepWidth
        classes.append(-1)

    point = np.array([d, 0])
    rotationM = np.array(
        [
            [np.cos(randRad[i]), -np.sin(randRad[i])],
            [np.sin(randRad[i]), np.cos(randRad[i])],
        ]
    )
    radPoints.append(point @ rotationM.T)

radPoints = np.array(radPoints)
print(radPoints.shape)
radClasses = np.array(classes)

# transform data
# VizalizeData(radPoints, radClasses, "Линейно неразделимые данные")
for c in C:
    if c == C[1]:
        print("here!")
    classifier = SVMClassifier(LinearKernel, C=c)
    MakeTest(classifier, linearSepData, linearClasses, True, True, True)

for c in C:
    if c == C[1]:
        print("here!")
    classifier = SVMClassifier(GaussianKernel(2.5), C=c)
    MakeTest(classifier, radPoints, radClasses, True, True, True, naming="gaussian")
"""
df = pd.read_csv("./datasets/heart.csv", sep=",", index_col=0)
df = df.drop([], axis=1)
x_sets = df.drop("output", axis=1).to_numpy()
y_sets = np.array([-1] * 165 + [1] * (x_sets.shape[0] - 165))
print(x_sets.shape)

start = time.time()
classifier = SVMClassifier(LinearKernel, C=10)
for i in range(x_sets.shape[0]):
    classifier.IncrementPair(x_sets[i], y_sets[i])
timer = time.time() - start
print(
    "S: {}, E: {}, R: {}, accuracy: {}, time: {}s".format(
        len(classifier.S),
        len(classifier.E),
        len(classifier.O),
        1 - (len(classifier.E) / x_sets.shape[0]),
        timer,
    )
)
"""
