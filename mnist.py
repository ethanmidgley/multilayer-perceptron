import numpy as np

from Activations import RELU, Sigmoid, Softmax
from Dense import Dense
from Input import Input
from Metrics import MSE, BinaryCrossEntropy
from Model import Model

train_data = np.loadtxt("./mnist_train.csv", delimiter=",")


X_train = np.array([np.array(d[1:]) for d in train_data])[:1000]
# Separating the labels from the image
# Y_train = np.array([([[1]] if d[0] == 1 else [[0]]) for d in train_data])[:500]
Y_train = np.array([([1] if d[0] == 1 else [0]) for d in train_data])[:1000]


# def encode(x):
#     a = np.zeros(10)
#     a[int(x)] = 1
#     return a


# Y_train = np.array(list(map(encode, Y_train)))

m = Model()
m.add(Input(784))
m.add(Dense(128, Sigmoid))
m.add(Dense(64, RELU))
m.add(Dense(1, Sigmoid))

m.compile(metric=BinaryCrossEntropy, learning_rate=0.1)

m.train(X_train, Y_train, epochs=10)


X_test = X_train[:15]
Y_test = Y_train[:15]

print(Y_train)

for row, y in zip(X_test, Y_test):
    print("Is 1:", y)
    output = m.predict(row)
    print("Output", output, "Y", y)
    print("Predicted possiblity of seven: ", m.predict(row))
