import utils
import csv
import os, sys

if len(os.listdir('cleaned')) == 0:
    utils.save_cleaned()
corpus = utils.Corpus()


with open('results.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=',')
    csvwriter.writerow(["baseball_" + i for i in ['entropy', 'tf-idf', 'tdv']] + ["hockey_" + i for i in ['entropy', 'tf-idf', 'tdv']])
    for i in range(50):
        csvwriter.writerow([result_baseball[0][i], result_baseball[1][i], result_baseball[2][i], result_hockey[0][i], result_hockey[1][i], result_hockey[2][i]])
