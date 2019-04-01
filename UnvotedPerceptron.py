import random
import numpy as np
from copy import copy


class UnvotedPerceptron:

    def __init__(self):
        print("")

    def train(self, matrix, epoche, lrate):
        '''for i in range(len(matrix[0])):
            # self.weights.append(random.uniform(0,2))
            self.weights.append(random.uniform(0, 2))'''
        self.weights = np.zeros(len(matrix[0])-1) # if instead of zero there's the thing sopra il percettrone non votato non ha quasi mai performance orrende
        for epoca in range(epoche):
            for i in range(len(matrix)):
                prediction = self.predict(matrix[i][:-1])
                error = matrix[i][-1]-prediction

                for j in range(len(self.weights)):
                    self.weights[j] = self.weights[j]+(lrate*error*matrix[i][j]) # se lrate=1 => lrate è nullo
        return

    def trainV2(self, matrix, epoche):
        self.weights = np.zeros(len(matrix[0]-1))
        for epoca in range(epoche):
            for i in range(len(matrix)):
                prediction = self.predict(matrix[i][:-1])

                # if prediction not correct, then...
                if prediction != matrix[i][-1]:
                    for j in range(len(self.weights)):
                        self.weights[j] = self.weights[j] + matrix[i][j]*matrix[i][-1]

        return

    def predict(self, inputs):
        threshold = 0
        total_activation = 0
        for input,weight in zip(inputs, self.weights):
            total_activation += input*weight

        if total_activation >= threshold:
            return 1
        else:
            return -1

    def good_prediction(self, dataset_instance):
            if dataset_instance[len(dataset_instance)-1] == self.predict(dataset_instance[:-1]):
                return True
            else:
                return False


def adapt_dataset(set):
    matrix = []
    for row in set:
        temp = [1]
        for a in row:
            temp.append(a)
        # print(temp)
        matrix.append(temp)
    return np.array(matrix)

def randomize_dataset(set):
    dataset = copy(set)
    for i in range(2*len(dataset)):
        first = random.randint(0, len(dataset)-1)
        second = random.randint(0, len(dataset)-1)
        temp = dataset[first]
        dataset[first] = dataset[second]
        dataset[second] = temp
        #print(first, second)
    return dataset