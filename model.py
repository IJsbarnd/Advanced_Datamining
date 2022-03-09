import numpy as num
from math import tanh
from collections import Counter
from copy import deepcopy


class Perceptron():

    def __repr__(self):
        text = f'Perceptron(dim={self.dim})'
        return text

    def __init__(self, dim):
        self.dim = dim
        self.bias = 0.0
        self.weights = [0.0, 0.0]

    def predict(self, xs):
        output = []
        for instance in xs:
            output.append(num.sign(self.bias + (self.weights[0] * instance[0] + self.weights[1] * instance[1])))
        return output

    def partial_fit(self, xs, ys):
        total_error = 0
        for x, y in zip(xs, ys):
            error = num.sign(self.bias + (self.weights[0] * x[0] + self.weights[1] * x[1])) - y
            #print(error)
            total_error += abs(error)

            self.bias = self.bias - error
            self.weights[0] = self.weights[0] - error * x[0]
            self.weights[1] = self.weights[1] - error * x[1]
        return total_error

    def fit(self, xs, ys, *, epochs=0):
        if epochs == 0:
            total_error = 1
            while total_error != 0:
                total_error = self.partial_fit(xs, ys)
                #print('----------------------')
        else:
            for epoch in range(epochs):
                self.partial_fit(xs, ys)
                #print('----------------------')


class LinearRegression():

    def __repr__(self):
        text = f'Perceptron(dim={self.dim})'
        return text

    def __init__(self, dim):
        self.dim = dim
        self.bias = 0.0
        self.weights = [0.0, 0.0]

    def predict(self, xs):
        output = []
        for instance in xs:
            output.append(self.bias + (self.weights[0] * instance[0] + self.weights[1] * instance[1]))
        return output

    def partial_fit(self, xs, ys, alpha=0.01):
        total_error = 0
        for x, y in zip(xs, ys):
            error = (self.bias + (self.weights[0] * x[0] + self.weights[1] * x[1])) - y
            #print(error)
            total_error += abs(error)

            self.bias = self.bias - alpha * error
            self.weights[0] = self.weights[0] - alpha * error * x[0]
            self.weights[1] = self.weights[1] - alpha * error * x[1]
        return total_error

    def fit(self, xs, ys, *, epochs=10, alpha=0.01):
        if epochs == 0:
            total_error = 1
            while total_error != 0:
                total_error = self.partial_fit(xs, ys)
                #print('----------------------')
        else:
            for epoch in range(epochs):
                self.partial_fit(xs, ys)
                #print('----------------------')


def linear(a):
    return a


def sign(a):
    return num.sign(a)


def mean_squared_error(yhat, y):
    return (yhat - y)**2


def mean_absolute_error(yhat, y):
    return abs(yhat - y)


def derivative(function, delta=0.001):

    def wrapper_derivative(x, *args):
        return (function(x + delta, *args) - function(x - delta, *args)) / (delta * 2)

    wrapper_derivative.__name__ = function.__name__ + '’'
    wrapper_derivative.__qualname__ = function.__qualname__ + '’'
    return wrapper_derivative


class Neuron():
    def __init__(self, dim, activation=linear, loss=mean_squared_error):
        self.bias = 0.0
        self.weights = [0.0, 0.0]
        self.dim = dim
        self.activation = activation
        self.loss = loss

    def __repr__(self):
        text = f'Neuron(dim={self.dim}, activation={self.activation.__name__}, loss={self.loss.__name__})'
        return text

    def predict(self, xs):
        output = []
        for instance in xs:
            prediction = self.activation(self.bias + (self.weights[0] * instance[0] + self.weights[1] * instance[1]))
            output.append(prediction)
        return output

    def partial_fit(self, xs, ys, *, alpha=0.01):
        total_error = 0
        derlos = derivative(self.loss)
        deract = derivative(self.activation)
        for x, y in zip(xs, ys):
            error = (self.bias + (self.weights[0] * x[0] + self.weights[1] * x[1]))
            total_error += abs(error)
            lossact = derlos(error, y) * deract(error)
            self.bias = self.bias - alpha * lossact
            self.weights[0] = self.weights[0] - alpha * lossact * x[0]
            self.weights[1] = self.weights[1] - alpha * lossact * x[1]
        return total_error

    def fit(self, xs, ys, *, alpha=0.01, epochs=10):
        if epochs == 0:
            total_error = 1
            while total_error != 0:
                total_error = self.partial_fit(xs, ys)
                #print('----------------------')
        else:
            for epoch in range(epochs):
                self.partial_fit(xs, ys)
                #print('----------------------')


def mytanh(a):
    return tanh(a)


def hinge(yhats, y):
    return max(1 - yhats * y, 0)


class Layer():

    classcounter = Counter()

    def __init__(self, outputs, *, name=None, next=None):
        Layer.classcounter[type(self)] += 1
        if name is None:
            name = f'{type(self).__name__}_{Layer.classcounter[type(self)]}'
        self.inputs = 0
        self.outputs = outputs
        self.name = name
        self.next = next

    def __repr__(self):
        text = f'Layer(inputs={self.inputs}, outputs={self.outputs}, name={repr(self.name)})'
        if self.next is not None:
            text += ' + ' + repr(self.next)
        return text

    def add(self, next):
        if self.next is None:
            self.next = next
            next.set_inputs(self.outputs)
        else:
            self.next.add(next)

    def set_inputs(self, inputs):
        self.inputs = inputs

    def __add__(self, next):
        result = deepcopy(self)
        result.add(deepcopy(next))
        return result

    def __getitem__(self, index):
        if index == 0 or index == self.name:
            return self
        if isinstance(index, int):
            if self.next is None:
                raise IndexError('Layer index out of range')
            return self.next[index - 1]
        if isinstance(index, str):
            if self.next is None:
                raise KeyError(index)
            return self.next[index]
        raise TypeError(f'Layer indices must be integers or strings, not {type(index).__name__}')
