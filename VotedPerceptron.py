import numpy as np
import random as rand
import random
from copy import copy

class VotedPerceptron:

    def sign(self, number):
        if number < 0:
            return -1
        else:
            return 1


    def __init__(self):
        self.v_array = []  # list of perceptrons
        self.c_array = []  # weights of perceptrons
        self.k = 0

    def train_on_dataset(self, epoche, dataset):

        # self.v_array.append(np.zeros(len(dataset[0])-1))
        array = np.zeros(len(dataset[0])-1)
        '''for i in range(len(array)):
            array[i] = 2'''
        self.v_array.append(array)
        self.c_array = []
        self.c_array.append(0)
        # dataset will be a 2d array
        hi = 0
        for epoca in range(epoche):
            # dataset = randomize_dataset(dataset) # todo era per provare se migliorava la situazione
            for i in range(len(dataset)):
                y_segnato = self.sign(np.inner(self.v_array[self.k], dataset[i][:-1]))
                # print(y_segnato, " è uguale a ", dataset[i][len(dataset[0])-1], "?\t", y_segnato == dataset[i][len(dataset[0])-1]) #allows showing the progress
                if y_segnato == dataset[i][-1]:
                    self.c_array[self.k] = self.c_array[self.k]+1 # aumenta il voto di del k-esimo percettrone
                    # dunque se la predizione è corretta, il voto di questo percettrone aumenta
                else:
                    #print(dataset[i][len(dataset[0])-1])
                    temp = dataset[i][-1] * dataset[i][:-1]  # prodotto vettore-scalare
                    # print("Temp: ", temp)
                    #print(self.v_array[self.k])
                    self.v_array.append(self.v_array[self.k] + temp)
                    self.c_array.append(1)
                    self.k = self.k + 1                #hi = hi+1

    def predict(self, dataset_instance):
            s = 0
            for i in range(self.k+1):
                s = s + self.c_array[i]*self.sign(np.inner(self.v_array[i], np.array(dataset_instance[:-1])))
                #print(s)
                # print(self.sign(np.inner(self.v_array[i], np.array(dataset_instance[:-1]))))
            prediction = self.sign(s)
            return prediction

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