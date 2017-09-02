import numpy as np
import random

class neuralNetwork(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.neurons = sizes
        self.biases = [np.random.randn(i,1) for i in sizes[1:]]
        self.weights = [np.random.randn(k, j) for j, k in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, activation):
        for b, w in zip(self.biases, self.weights):
            activation = self.sigmoid(np.dot(w, activation) + b)
        return activation

    def SGD(self, trainingData, miniBatchSize, eta, epochs, testData = None):

        for i in xrange(epochs):
            random.shuffle(trainingData)
            miniBatches = [trainingData[j:j+miniBatchSize] for j in xrange(0,len(trainingData),miniBatchSize)]

            for k in miniBatches:
                self.updateMiniBatch(k, eta, miniBatchSize)

            if testData:
                print "Epoch {0}: {1} / {2}".format(i, self.evaluate(testData), len(testData))
            else:
                print "Epoch {0} complete".format(i)


    def updateMiniBatch(self,miniBatch, eta, miniBatchSize):

        for x, y in miniBatch:
            delWeight = [np.zeros(w.shape) for w in self.weights]
            delBiases = [np.zeros(b.shape) for b in self.biases]
            activations = [x]
            weightedInput = []

            for i in range(0,self.num_layers-1):
                weightedInput.append(np.dot(self.weights[i],activations[i]) + self.biases[i])
                activations.append(self.sigmoid(weightedInput[i]))

            delta = (activations[self.num_layers-1] - y)*self.sigmoid_derivative(activations[self.num_layers-1])

            delWeight[self.num_layers-2] += np.dot(delta, activations[self.num_layers-2].transpose())
            delBiases[self.num_layers-2] += delta

            for i in xrange(self.num_layers-2,0,-1):
                delta = (np.dot(self.weights[i].transpose(), delta))* self.sigmoid_derivative(activations[i])

                delWeight[i-1] += np.dot(delta, activations[i-1].transpose())
                delBiases[i-1] += delta

            for w, b, dw, db in zip(self.weights, self.biases, delWeight, delBiases):
                w -= (eta/miniBatchSize)*dw
                b -= (eta/miniBatchSize)*db


    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def sigmoid(self, weightedInput):
        return 1.0/(1.0 + np.exp(-weightedInput))

    def sigmoid_derivative(self, activation):
        return activation*(1-activation)



