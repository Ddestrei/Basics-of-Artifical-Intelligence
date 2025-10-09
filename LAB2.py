# Zadanie 1

class Neuron:
    def __init__(self, weight, alfa):
        self.weight = weight
        self.alfa = alfa

    def learn_one_epoch(self, input, goal):
        predict = input*self.weight
        delta = 2*(predict - goal)*input
        error = (predict - goal)**2
        self.weight -= delta*self.alfa
        print(predict)
        return error

    def learning(self, input, goal, n_epoch):
        for i in range(n_epoch):
            print(self.learn_one_epoch(input, goal))

neuron = Neuron(weight=0.5, alfa=0.1)
neuron.learning(2, 0.8, 5)

# Zadanie 2

class Layer:



