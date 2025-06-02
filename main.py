import numpy as np

from Activations import RELU, Sigmoid
from Dense import Dense
from Input import Input
from Metrics import MSE, BinaryCrossEntropy
from Model import Model

m = Model()

m.add(Input(2))
m.add(Dense(16, RELU))
m.add(Dense(16, RELU))
m.add(Dense(1, Sigmoid))

# m.add(Input(2))
# m.add(Dense(2, RELU))
# m.add(Dense(2, RELU))
# m.add(Dense(1, Sigmoid))


m.compile(metric=BinaryCrossEntropy, learning_rate=0.1)


x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

y = np.array([[0], [1], [1], [0]])

m.train(x, y, epochs=100000)

for input_row in x:
    prediction = m.predict(input_row)
    print(f"Input: {input_row} => Prediction: {prediction}")
