import os
import sys

sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-3]))


import numpy as np

from Model import Model

data = np.loadtxt("./data/mnist_train.csv", delimiter=",")

X_data = np.array([np.array(d[1:]) for d in data])
Y_data = np.array([d[0] for d in data]).astype("int")

X_train = X_data[:49990]
Y_train = Y_data[:49990]


m = Model.load("./examples/saving-loading-models/model.data")

X_test = X_data[49990:]
Y_test = Y_data[49990:]


for row, y in zip(X_test, Y_test):
    output = m.predict(row)
    print("Output", np.argmax(output, axis=1), "Y", y)
