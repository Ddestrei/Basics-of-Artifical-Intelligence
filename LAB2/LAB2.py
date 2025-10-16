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
print("x = 0.1")
neuron = Neuron(weight=0.5, alfa=1)
neuron.learning(2, 0.8, 5)

# Zadanie 2

print("Zadanie 2")


class Layer:

    def __init__(self, weight, alfa):
        self.alfa = alfa
        self.weight = weight

    def learning(self, input_data, goal, n_epoch):
        for j in range(n_epoch):
            print("Epoch: ", j)
            for i in range(input_data.shape[1]):
                print("Series: ", i)
                predicts = np.zeros(self.weight.shape[0])
                for n in range(self.weight.shape[0]):
                    predicts[n] = np.dot(input_data[:, i], self.weight[n])
                # print("Predicts: ", predicts)
                delta = (2 / self.weight.shape[0]) * np.outer(np.subtract(predicts, goal[:, i]), input_data[:, i])
                error =  np.sum(np.subtract(predicts, goal[:, i])** 2)
                self.weight -= self.alfa * delta
                # print(self.weight)
                # print(predicts)
                print("Error: ", error)
        print(self.weight)

    def test(self, input_data, goal):
        true = 0
        for i in range(input_data.shape[1]):
            predicts = np.zeros(self.weight.shape[0])
            for n in range(self.weight.shape[0]):
                predicts[n] = np.dot(input_data[:, i], self.weight[n])
            predicts = (predicts == predicts.max())
            #print(predicts)
            if (predicts == goal[:, i]).all():
                #print(predicts, " ", goal[:, i])
                true += 1
            else:
                #print(predicts, " ", goal[:, i])
                pass

        return true / input_data.shape[1]


layer = Layer(np.array([[0.1, 0.1, -0.3], [0.1, 0.2, 0.0], [0.0, 0.7, 0.1], [0.2, 0.4, 0.0], [-0.3, 0.5, 0.1]]), 0.01)
input_data = np.array([[0.5, 0.1, 0.2, 0.8], [0.75, 0.3, 0.1, 0.9], [0.1, 0.7, 0.6, 0.2]])
goal_data = np.array(
    [[0.1, 0.5, 0.1, 0.7], [1.0, 0.2, 0.3, 0.6], [0.1, -0.5, 0.2, 0.2], [0.0, 0.3, 0.9, -0.1], [-0.1, 0.7, 0.1, 0.8]])
layer.learning(input_data, goal_data, 1000)

# zadanie 3

print("Zadanie 3")

training = np.loadtxt("training.txt")
training_input = training[:, :-1]
training_output_raw = training[:, -1:]
training_output = np.zeros((training_output_raw.shape[0], 4))
for i in range(len(training_output)):
    if training_output_raw[i][0] == 1.0:
        training_output[i] = np.array([1, 0, 0, 0])
    elif training_output_raw[i][0] == 2.0:
        training_output[i] = np.array([0, 1, 0, 0])
    elif training_output_raw[i][0] == 3.0:
        training_output[i] = np.array([0, 0, 1, 0])
    elif training_output_raw[i][0] == 4.0:
        training_output[i] = np.array([0, 0, 0, 1])

test = np.loadtxt("test.txt")
test_input = test[:, :-1]
test_output_raw = test[:, -1:]
test_output = np.zeros((test_output_raw.shape[0], 4))
for i in range(len(test_output)):
    if test_output_raw[i][0] == 1.0:
        test_output[i] = np.array([1, 0, 0, 0])
    elif test_output_raw[i][0] == 2.0:
        test_output[i] = np.array([0, 1, 0, 0])
    elif test_output_raw[i][0] == 3.0:
        test_output[i] = np.array([0, 0, 1, 0])
    elif test_output_raw[i][0] == 4.0:
        test_output[i] = np.array([0, 0, 0, 1])

color_recognition = Layer(np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]), 0.01)
color_recognition.learning(np.transpose(training_input), np.transpose(training_output), 6)

print(color_recognition.test(np.transpose(test_input), np.transpose(test_output)))
