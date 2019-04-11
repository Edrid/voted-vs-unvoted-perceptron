import numpy as np
import UnvotedPerceptron
import VotedPerceptronV2
from copy import copy
import time
import csv
import matplotlib.pyplot as plt

''' for measuring execution time: anche se...
import time

start = time.time()
print("hello")
end = time.time()
print(end - start)

'''

# Note to myself: se learning rate = 1 allora è come se non ci fosse, chiaramente
EPOCHE = 1

def main():
    run_tests_and_save_results()



def run_tests_and_save_results(): #this has to be changed in order to change the datasets and where it has to be saved
    global EPOCHE
    #print(dataset1_occupancy())
    #dataset2_banknotes()

    results = []
    voted_score = 0
    unvoted_score = 0
    for _ in range(6):
        for ciao in range(5):
            res = dataset2_banknotes()
            res = (EPOCHE, ) + res
            # print(res)
            results.append(res)
            if res[1] > res[2]:
                unvoted_score += 1
            else:
                voted_score += 1
        EPOCHE += 1

    for dato in results:
        print(dato)
    print("Voted score: ", voted_score, "Unvoted score: ", unvoted_score)

    '''with open('Files/Banknotes/adult_results.csv', mode='w') as data_file:
        data_writer = csv.writer(data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        for dato in results:
            data_writer.writerow(dato)'''

    # dataset2_banknotes()
    # dataset3_abeloni()





def dataset1_occupancy():
    print("\tABOUT: occupancy")
    # parte non votata
    matrice = np.loadtxt(open("Files/Occupation/occupazione.csv", "rb"), delimiter=",", skiprows=0)
    matrice = VotedPerceptronV2.randomize_dataset(matrice)
    testset = np.loadtxt(open("Files/Occupation/occtest1.csv", "rb"), delimiter=",", skiprows=0)
    matrix = UnvotedPerceptron.adapt_dataset(matrice)
    # print(matrix)
    unvoted = UnvotedPerceptron.UnvotedPerceptron()
    # matrix = UnvotedPerceptron.randomize_dataset(matrix)
    #unvoted.train(matrix, 1, 0.9) # dati, epoche, learning rate
    start = time.time()
    unvoted.trainV2(matrix, EPOCHE)
    end = time.time()
    unvoted_training_time = end-start

    start = time.time()
    unvoted_testset = UnvotedPerceptron.adapt_dataset(testset)
    end = time.time()
    unvoted_pred_time = end-start

    # parte votata
    start = time.time()
    voted = VotedPerceptronV2.VotedPerceptron()
    end = time.time()
    voted_training_time = end-start

    start = time.time()
    voted.train_on_dataset(EPOCHE, matrice)
    end = time.time()
    voted_pred_time = end-start

    unvoted_perf = measureUnvotedPerceptron(unvoted, unvoted_testset)

    voted_perf = measureVotedPerformance(voted, testset)

    return unvoted_perf, voted_perf, unvoted_training_time, unvoted_pred_time, voted_training_time, voted_pred_time


def dataset2_banknotes(): # le operazioni di adattamento delle matrici non devono alterare le matrici, sembra che per ora sia così, cambiando l'ordine dell'esecuzione il risultato cambia
    print("\n\tFake banknotes", "\tEpoche: ", EPOCHE)
    # parte non votata
    matrice = np.loadtxt(open("Files/Banknotes/banknotes_scaled.csv", "rb"), delimiter=",", skiprows=0)
    matrice = VotedPerceptronV2.randomize_dataset(matrice)
    testset = np.loadtxt(open("Files/Banknotes/banknotes_test.csv", "rb"), delimiter=",", skiprows=0)
    matrix = UnvotedPerceptron.adapt_dataset(copy(matrice))  # matrice has already been randomized
    # print(matrix)

    # parte votata
    voted = VotedPerceptronV2.VotedPerceptron()
    voted.train_on_dataset(EPOCHE, matrice)

    unvoted = UnvotedPerceptron.UnvotedPerceptron()
    # matrix = UnvotedPerceptron.randomize_dataset(matrix)
    # unvoted.train(copy(matrix), 1, 1)  # dati, epoche, learning rate
    start = time.time()
    unvoted.trainV2(matrix, EPOCHE)
    end = time.time()
    unvoted_training_time = end - start

    start = time.time()
    unvoted_testset = UnvotedPerceptron.adapt_dataset(testset)
    end = time.time()
    unvoted_pred_time = end - start

    # parte votata
    start = time.time()
    voted = VotedPerceptronV2.VotedPerceptron()
    end = time.time()
    voted_training_time = end - start

    start = time.time()
    voted.train_on_dataset(EPOCHE, matrice)
    end = time.time()
    voted_pred_time = end - start

    unvoted_perf = measureUnvotedPerceptron(unvoted, unvoted_testset)

    voted_perf = measureVotedPerformance(voted, testset)

    return unvoted_perf, voted_perf, unvoted_training_time, unvoted_pred_time, voted_training_time, voted_pred_time


''' def dataset3_abeloni():
    print("\n\tAbeloni", "\tEpoche: ", EPOCHE)
    matrice = np.loadtxt(open("Files/Abelones/abaloni_training.csv", "rb"), delimiter=",", skiprows=0)
    matrice = VotedPerceptronV2.randomize_dataset(matrice)
    testset = np.loadtxt(open("Files/Abelones/abaloni_test.csv", "rb"), delimiter=",", skiprows=0)
    matrix = UnvotedPerceptron.adapt_dataset(copy(matrice))

    voted = VotedPerceptronV2.VotedPerceptron()
    voted.train_on_dataset(EPOCHE, matrice)

    # parte non votata
    unvoted = UnvotedPerceptron.UnvotedPerceptron()
    # unvoted.train(copy(matrix), 15, 1)
    start = time.time()
    unvoted.trainV2(matrix, EPOCHE)
    end = time.time()
    unvoted_training_time = end - start

    start = time.time()
    unvoted_testset = UnvotedPerceptron.adapt_dataset(testset)
    end = time.time()
    unvoted_pred_time = end - start

    # parte votata
    start = time.time()
    voted = VotedPerceptronV2.VotedPerceptron()
    end = time.time()
    voted_training_time = end - start

    start = time.time()
    voted.train_on_dataset(EPOCHE, matrice)
    end = time.time()
    voted_pred_time = end - start

    unvoted_perf = measureUnvotedPerceptron(unvoted, unvoted_testset)

    voted_perf = measureVotedPerformance(voted, testset)

    return unvoted_perf, voted_perf, unvoted_training_time, unvoted_pred_time, voted_training_time, voted_pred_time'''
'''
def dataset4_madalones():
    print("\n\tABOUT: madelones")
    matrice = np.loadtxt(open("Files/Madalones/madalones_training_adapted.csv", "rb"), delimiter=",", skiprows=0)
    matrice = VotedPerceptronV2.randomize_dataset(matrice)
    testset = np.loadtxt(open("Files/Madalones/madalones_test_adapted.csv", "rb"), delimiter=",", skiprows=0)
    matrix = UnvotedPerceptron.adapt_dataset(copy(matrice))

    voted = VotedPerceptronV2.VotedPerceptron()
    voted.train_on_dataset(EPOCHE, matrice)

    # parte non votata
    unvoted = UnvotedPerceptron.UnvotedPerceptron()
    # unvoted.train(copy(matrix), 15, 1)
    unvoted.trainV2(copy(matrix), EPOCHE)

    unvoted_testset = UnvotedPerceptron.adapt_dataset(copy(testset))
    measureUnvotedPerceptron(unvoted, unvoted_testset)
    measureVotedPerformance(voted, testset)
'''
def dataset5_adult():
    print("\tABOUT: adult")
    # parte non votata
    matrice = np.loadtxt(open("Files/Adult/adult_training_normalized.csv", "rb"), delimiter=",", skiprows=0)
    matrice = VotedPerceptronV2.randomize_dataset(matrice)
    testset = np.loadtxt(open("Files/Adult/adult_test_normalized.csv", "rb"), delimiter=",", skiprows=0)
    matrix = UnvotedPerceptron.adapt_dataset(matrice)
    # print(matrix)
    unvoted = UnvotedPerceptron.UnvotedPerceptron()
    # matrix = UnvotedPerceptron.randomize_dataset(matrix)
    #unvoted.train(matrix, 1, 0.9) # dati, epoche, learning rate
    start = time.time()
    unvoted.trainV2(matrix, EPOCHE)
    end = time.time()
    unvoted_training_time = end-start

    start = time.time()
    unvoted_testset = UnvotedPerceptron.adapt_dataset(testset)
    end = time.time()
    unvoted_pred_time = end-start

    # parte votata
    start = time.time()
    voted = VotedPerceptronV2.VotedPerceptron()
    end = time.time()
    voted_training_time = end-start

    start = time.time()
    voted.train_on_dataset(EPOCHE, matrice)
    end = time.time()
    voted_pred_time = end-start

    unvoted_perf = measureUnvotedPerceptron(unvoted, unvoted_testset)

    voted_perf = measureVotedPerformance(voted, testset)

    return unvoted_perf, voted_perf, unvoted_training_time, unvoted_pred_time, voted_training_time, voted_pred_time

def dataset6_heart():
    print("\tABOUT: heart")
    # parte non votata
    matrice = np.loadtxt(open("Files/Heart/heart_train.csv", "rb"), delimiter=",", skiprows=0)
    matrice = VotedPerceptronV2.randomize_dataset(matrice)
    testset = np.loadtxt(open("Files/Heart/heart_test.csv", "rb"), delimiter=",", skiprows=0)
    matrix = UnvotedPerceptron.adapt_dataset(matrice)
    # print(matrix)
    unvoted = UnvotedPerceptron.UnvotedPerceptron()
    # matrix = UnvotedPerceptron.randomize_dataset(matrix)
    # unvoted.train(matrix, 1, 0.9) # dati, epoche, learning rate
    start = time.time()
    unvoted.trainV2(matrix, EPOCHE)
    end = time.time()
    unvoted_training_time = end - start

    start = time.time()
    unvoted_testset = UnvotedPerceptron.adapt_dataset(testset)
    end = time.time()
    unvoted_pred_time = end - start

    # parte votata
    start = time.time()
    voted = VotedPerceptronV2.VotedPerceptron()
    end = time.time()
    voted_training_time = end - start

    start = time.time()
    voted.train_on_dataset(EPOCHE, matrice)
    end = time.time()
    voted_pred_time = end - start

    unvoted_perf = measureUnvotedPerceptron(unvoted, unvoted_testset)

    voted_perf = measureVotedPerformance(voted, testset)

    return unvoted_perf, voted_perf, unvoted_training_time, unvoted_pred_time, voted_training_time, voted_pred_time


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

    # print("[VOTED] Accuratezza ", round((counter/total)*100, 3), "%")
    print("[VOTED] ")
    print(matriceDiConfusione)
    #print("[VOTED] Presi giusti ", counter, " su ", total)
    return round((counter/total)*100, 3)



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

    # print("[UNVOTED] Accuratezza", round((count/len(testset))*100, 3), "%")
    print("[UNVOTED] ")
    print(matriceDiConfusione)
    return round((count/len(testset))*100, 3)


if __name__ == "__main__":
    main()


'''
    Matrice di confusione:
                [Predicted 1, Predicted -1]
    [Label 1]       x11          x12
    [Label -1]      x21          x22
'''