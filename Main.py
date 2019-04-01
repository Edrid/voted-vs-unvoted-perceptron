import numpy as np
import UnvotedPerceptron
import VotedPerceptron
from copy import copy

''' for measuring execution time:
import time

start = time.time()
print("hello")
end = time.time()
print(end - start)

'''

# Note to myself: se learning rate = 1 allora è come se non ci fosse, chiaramente
EPOCHE = 1

def main():
    dataset1_occupancy()
    dataset2_banknotes()
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
    #unvoted.train(matrix, 1, 0.9) # dati, epoche, learning rate
    unvoted.trainV2(matrix, EPOCHE)
    unvoted_testset = UnvotedPerceptron.adapt_dataset(testset);

    # parte votata
    voted = VotedPerceptron.VotedPerceptron()
    voted.train_on_dataset(EPOCHE, matrice)

    measureUnvotedPerceptron(unvoted, unvoted_testset)

    measureVotedPerformance(voted, testset)

def dataset2_banknotes(): # fixme: le operazioni di adattamento delle matrici non devono alterare le matrici, sembra che per ora sia così, cambiando l'ordine dell'esecuzione il risultato cambia
    print("\n\tABOUT: fake banknotes")
    # parte non votata
    matrice = np.loadtxt(open("Files/Banknotes/banknotes.csv", "rb"), delimiter=",", skiprows=0)
    matrice = VotedPerceptron.randomize_dataset(matrice)
    testset = np.loadtxt(open("Files/Banknotes/banknotes_test.csv", "rb"), delimiter=",", skiprows=0)
    matrix = UnvotedPerceptron.adapt_dataset(copy(matrice)) # matrice has already been randomized
    # print(matrix)

    # parte votata
    voted = VotedPerceptron.VotedPerceptron()
    voted.train_on_dataset(EPOCHE, matrice)

    unvoted = UnvotedPerceptron.UnvotedPerceptron()
    # matrix = UnvotedPerceptron.randomize_dataset(matrix)
    # unvoted.train(copy(matrix), 1, 1)  # dati, epoche, learning rate
    unvoted.trainV2(copy(matrix), EPOCHE)


    unvoted_testset = UnvotedPerceptron.adapt_dataset(copy(testset));
    measureUnvotedPerceptron(unvoted, unvoted_testset)

    measureVotedPerformance(voted, testset)



def dataset3_abeloni():
    print("\n\tABOUT: abelones")
    matrice = np.loadtxt(open("Files/Abelones/abaloni_test.csv", "rb"), delimiter=",", skiprows=0)
    matrice = VotedPerceptron.randomize_dataset(matrice)
    testset = np.loadtxt(open("Files/Abelones/abaloni_training.csv", "rb"), delimiter=",", skiprows=0)
    matrix = UnvotedPerceptron.adapt_dataset(copy(matrice))

    voted = VotedPerceptron.VotedPerceptron()
    voted.train_on_dataset(EPOCHE, matrice)

    # parte non votata
    unvoted = UnvotedPerceptron.UnvotedPerceptron()
    # unvoted.train(copy(matrix), 15, 1)
    unvoted.trainV2(copy(matrix), EPOCHE)

    unvoted_testset = UnvotedPerceptron.adapt_dataset(copy(testset))
    measureUnvotedPerceptron(unvoted, unvoted_testset)
    measureVotedPerformance(voted, testset)


def measureVotedPerformance(percettrone, testset):
    counter = 0
    total = 0
    matriceDiConfusione = np.zeros((2, 2), dtype=int)
    # print(matriceDiConfusione)
    # se prediction is 1 then mat[0,0]+1, if prediction is -1 then mat[0,1]+1.. and so on.. todo
    for elemento in testset:

        prediction = percettrone.predict(elemento)
        if prediction == 1 and elemento[-1] == 1:
            matriceDiConfusione[0, 0] += 1
        elif prediction == -1 and elemento[-1] == -1:
            matriceDiConfusione[1, 1] += 1
        elif prediction == -1 and elemento[-1] == 1:
            matriceDiConfusione[0, 1] += 1
        elif prediction == 1 and elemento[-1] == -1:
            matriceDiConfusione[1, 0] += 1
        else:
            print("----------------- ERRORE -------------------")


        prediction = percettrone.good_prediction(elemento)
        #print(prediction)
        if prediction:
            counter = counter+1
        total = total + 1

    print("[VOTED] Accuratezza ", round((counter/total)*100, 3), "%")
    print(matriceDiConfusione)
    #print("[VOTED] Presi giusti ", counter, " su ", total)


def measureUnvotedPerceptron(perceptron, testset):
    count = 0
    matriceDiConfusione = np.zeros((2, 2), dtype=int)
    for elemento in testset:

        prediction = perceptron.predict(elemento[:-1])
        if prediction == 1 and elemento[-1] == 1:
            matriceDiConfusione[0, 0] += 1
        elif prediction == -1 and elemento[-1] == -1:
            matriceDiConfusione[1, 1] += 1
        elif prediction == -1 and elemento[-1] == 1:
            matriceDiConfusione[0, 1] += 1
        elif prediction == 1 and elemento[-1] == -1:
            matriceDiConfusione[1, 0] += 1
        else:
            print("----------------- ERRORE -------------------")

        res = perceptron.good_prediction(elemento)
        if res:
            count+=1
        # print(i, "Risultato", res)

    print("[UNVOTED] Accuratezza", round((count/len(testset))*100, 3), "%")
    print(matriceDiConfusione)


if __name__ == "__main__":
    main()


'''
    Matrice di confusione:
                [Predicted 1, Predicted -1]
    [Label 1]       x11          x12
    [Label -1]      x21          x22
'''