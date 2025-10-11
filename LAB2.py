# Zadanie 1
import numpy as np


class Neuron:
    def __init__(self, weight, alfa):
        self.weight = weight
        self.alfa = alfa

    def learn_one_epoch(self, input, goal):
        predict = input * self.weight
        delta = 2 * (predict - goal) * input
        error = (predict - goal) ** 2
        self.weight -= delta * self.alfa
        print(predict)
        return error

    def learning(self, input, goal, n_epoch):
        for i in range(n_epoch):
            print(self.learn_one_epoch(input, goal))


neuron = Neuron(weight=0.5, alfa=0.1)
neuron.learning(2, 0.8, 5)

# Zadanie 2

print("Zadanie 2")


class Layer:

    def __init__(self, alfa):
        self.alfa = alfa
        self.weight = np.array([[0.1, 0.1, -0.3], [0.1, 0.2, 0.0], [0.0, 0.7, 0.1], [0.2, 0.4, 0.0], [-0.3, 0.5, 0.1]])

    def learning(self, input_data, goal, n_epoch):
        for j in range(n_epoch):
            print("Epoch: ", j)
            for i in range(input_data.shape[1]):
                print("Series: ", i)
                predicts = np.zeros(self.weight.shape[0])
                for n in range(self.weight.shape[0]):
                    predicts[n] = np.dot(input_data[:, i], self.weight[n])
                delta = (2 / self.weight.shape[0]) * np.outer(np.subtract(predicts, goal[:, i]), input_data[:, i])
                error = (1/self.weight.shape[0])*(np.sum(np.subtract(predicts, goal[:, i]))**2)
                self.weight -= self.alfa * delta
                #print(self.weight)
                #print(predicts)
                print("Error: ",error)
        #print(self.weight)


layer = Layer(0.01)
input_data = np.array([[0.5, 0.1, 0.2, 0.8], [0.75, 0.3, 0.1, 0.9], [0.1, 0.7, 0.6, 0.2]])
goal_data = np.array(
    [[0.1, 0.5, 0.1, 0.7], [1.0, 0.2, 0.3, 0.6], [0.1, -0.5, 0.2, 0.2], [0.0, 0.3, 0.9, -0.1], [-0.1, 0.7, 0.1, 0.8]])
layer.learning(input_data, goal_data, 1000 )
