import numpy as np
import UnvotedPerceptron
import VotedPerceptron
from copy import copy

def main():
    # dataset1_occupancy()
    # dataset2_banknotes()
    dataset3_abeloni()



def dataset1_occupancy():
    print("\tABOUT: occupancy")
    # parte non votata
    matrice = np.loadtxt(open("Files/Occupation/occupazione.csv", "rb"), delimiter=",", skiprows=0)
    matrice = VotedPerceptron.randomize_dataset(matrice)
    testset = np.loadtxt(open("Files/Occupation/occtest1.csv", "rb"), delimiter=",", skiprows=0)
    matrix = UnvotedPerceptron.adapt_dataset(matrice)
    # print(matrix)
    unvoted = UnvotedPerceptron.UnvotedPerceptron()
    # matrix = UnvotedPerceptron.randomize_dataset(matrix)
    unvoted.train(matrix, 1, 0.9) # dati, epoche, learning rate

    unvoted_testset = UnvotedPerceptron.adapt_dataset(testset);
    measureUnvotedPerceptron(unvoted, unvoted_testset)

    # parte votata
    voted = VotedPerceptron.VotedPerceptron()
    voted.train_on_dataset(1, matrice)
    measureVotedPerformance(voted, testset)

def dataset2_banknotes(): # fixme: le operazioni di adattamento delle matrici non devono alterare le matrici, sembra che per ora sia così, cambiando l'ordine dell'esecuzione il risultato cambia
    print("\tABOUT: fake banknotes")
    # parte non votata
    matrice = np.loadtxt(open("Files/Banknotes/banknotes.csv", "rb"), delimiter=",", skiprows=0)
    matrice = VotedPerceptron.randomize_dataset(matrice)
    testset = np.loadtxt(open("Files/Banknotes/banknotes_test.csv", "rb"), delimiter=",", skiprows=0)
    matrix = UnvotedPerceptron.adapt_dataset(copy(matrice)) # matrice has already been randomized
    # print(matrix)

    # parte votata
    voted = VotedPerceptron.VotedPerceptron()
    voted.train_on_dataset(10, matrice)
    measureVotedPerformance(voted, testset)

    unvoted = UnvotedPerceptron.UnvotedPerceptron()
    # matrix = UnvotedPerceptron.randomize_dataset(matrix)
    unvoted.train(copy(matrix), 1, 0.01)  # dati, epoche, learning rate

    unvoted_testset = UnvotedPerceptron.adapt_dataset(copy(testset));
    measureUnvotedPerceptron(unvoted, unvoted_testset)

def dataset3_abeloni():
    print("\tABOUT: abelones")
    matrice = np.loadtxt(open("Files/Abelones/abaloni_test.csv", "rb"), delimiter=",", skiprows=0)
    matrice = VotedPerceptron.randomize_dataset(matrice)
    testset = np.loadtxt(open("Files/Abelones/abaloni_training.csv", "rb"), delimiter=",", skiprows=0)
    matrix = UnvotedPerceptron.adapt_dataset(copy(matrice))

    voted = VotedPerceptron.VotedPerceptron()
    voted.train_on_dataset(5, matrice)
    measureVotedPerformance(voted, testset)

    # parte non votata
    unvoted = UnvotedPerceptron.UnvotedPerceptron()
    unvoted.train(copy(matrix), 5, 1)

    unvoted_testset = UnvotedPerceptron.adapt_dataset(copy(testset))
    measureUnvotedPerceptron(unvoted, unvoted_testset)

def measureVotedPerformance(percettrone, testset):
    counter = 0
    total = 0
    for elemento in testset:
        prediction = percettrone.good_prediction(elemento)
        #print(prediction)
        if prediction:
            counter = counter+1
        total = total + 1

    print("[VOTED] Accuratezza ", (counter/total)*100, "%")
    #print("[VOTED] Presi giusti ", counter, " su ", total)


def measureUnvotedPerceptron(perceptron, testset):
    count = 0
    for i in range(len(testset)):
        res = perceptron.good_prediction(testset[i])
        if res:
            count+=1
        # print(i, "Risultato", res)

    print("[UNVOTED] Accuratezza", (count/len(testset))*100, "%")


if __name__ == "__main__":
    main()
