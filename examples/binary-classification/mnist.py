import os
import sys

sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-3]))

import numpy as np

from Activations import RELU, Sigmoid
from Dense import Dense
from Input import Input
from Metrics import BinaryCrossEntropy
from Model import Model

train_data = np.loadtxt("./data/mnist_train.csv", delimiter=",")


X_train = np.array([np.array(d[1:]) for d in train_data])[:1000]
Y_train = np.array([([1] if d[0] == 1 else [0]) for d in train_data])[:1000]


m = Model()
m.add(Input(784))
m.add(Dense(128, Sigmoid))
m.add(Dense(64, RELU))
m.add(Dense(1, Sigmoid))

m.compile(metric=BinaryCrossEntropy)

m.train(X_train, Y_train, epochs=10)


X_test = X_train[:15]
Y_test = Y_train[:15]

print(Y_train)

for row, y in zip(X_test, Y_test):
    print("Is 1:", y)
    output = m.predict(row)
    print("Output", output, "Y", y)
    print("Predicted possiblity of seven: ", m.predict(row))
