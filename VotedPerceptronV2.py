import numpy as np
from copy import copy
import random

class VotedPerceptron:

    def __init__(self):
        self.percettroni = [] # this is a list of lists (2D array, of perceptrons)
        self.voti = [] # this is the list of votes for every perceptron

    def __sign(self, num):
        if num >= 0:
            return +1
        elif num < 0:
            return -1
        else:
            print("ERRORE NELLA FUNZIONE SIGN")

    def train_on_dataset(self, epoche, dataset):
        self.percettroni.append(np.zeros(len(dataset[0])-1))
        self.voti.append(0)
        for _ in range(epoche):
            for row in dataset:
                x = np.array(row[:-1])
                temp = np.inner(x, self.percettroni[-1])
                y_segnato = self.__sign(temp)
                if y_segnato == row[-1]:
                    self.voti[-1] += 1
                else:
                    newPerceptron = self.percettroni[-1] + row[-1]*np.array(row[:-1])
                    self.percettroni.append(newPerceptron)
                    self.voti.append(1)

    def predict(self, dataset_instance):
        sum = 0
        for i in range(len(self.percettroni)):
            sum += self.voti[i] * self.__sign(np.inner(np.array(dataset_instance[:-1]), self.percettroni[i]))
        return self.__sign(sum)

    def good_prediction(self, dataset_instance):
            if dataset_instance[len(dataset_instance)-1] == self.predict(dataset_instance):
                return True
            else:
                return False

def randomize_dataset(set):
    dataset = copy(set)
    for i in range(2*len(dataset)):
        first = random.randint(0, len(dataset)-1)
        second = random.randint(0, len(dataset)-1)
        temp = dataset[first]
        dataset[first] = dataset[second]
        dataset[second] = temp
        # print(first, second)
    return dataset