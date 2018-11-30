import os, sys
import pickle
from tqdm import tqdm
from gensim.parsing.preprocessing import *
from gensim.utils import lemmatize
from gensim.corpora import Dictionary
import numpy as np
import matplotlib.pyplot as plt
import csv

blacklist = ['be', 'do', 're']

def clean(text):
    try:
        tmp = text.split()
        for i in range(len(tmp)):
            if '@' in tmp[i]:
                tmp[i] = ' '
        text = ' '.join(tmp)
        text = strip_multiple_whitespaces(strip_non_alphanum(text)).split()
    except:
        text = []
    words = []
    for word in text:
        tmp = lemmatize(word)
        if tmp:
            try:
                if tmp[0][:-3].decode("ISO-8859-1") in blacklist:
                    continue
                words.append(tmp[0][:-3].decode("ISO-8859-1"))
            except:
                continue
    return " ".join(words)


class Corpus():
    def __init__(self):
        self.name = "corpus.bin"
        if self.name in os.listdir():
            with open(self.name, 'rb') as f:
                tmp = pickle.load(f)
                self.dict = tmp.dict
                self.corpus = tmp.corpus
                self.tfs = tmp.tfs
                self.tfs_baseball = tmp.tfs_baseball
                self.tfs_hockey = tmp.tfs_hockey
                self.length = tmp.length
                self.length_baseball = tmp.length_baseball
                self.length_hockey = tmp.length_hockey
                self.num_baseball = tmp.num_baseball
                self.num_hockey = tmp.num_hockey
            print("corpus loaded")
        else:
            print("building corpus...")
            self.build_corpus()
            self.save()
            print("corpus saved")

    def build_corpus(self):
        self.dict = Dictionary()
        self.corpus = []
        self.tfs = {}
        self.tfs_baseball = {}
        self.tfs_hockey = {}
        self.length = 0
        self.length_baseball = 0
        self.length_hockey = 0
        self.num_baseball = 0
        self.num_hockey = 0
        txts = [i for i in os.listdir(dir_cleaned) if i.endswith(".txt")]
        print("collecting documents...")
        for txt in tqdm(txts):
            doc = read_file(dir_cleaned+txt).split()
            self.dict.add_documents([doc])
            self.length += len(doc)
            if txt.startswith("baseball"):
                self.corpus.append(Doc(0,doc))
                self.length_baseball += len(doc)
                self.num_baseball += 1
            elif txt.startswith("hockey"):
                self.corpus.append(Doc(1,doc))
                self.length_hockey += len(doc)
                self.num_hockey += 1
        print("counting tfs...")
        for d in tqdm(self.corpus):
            for t, tf in d.tfs.items():
                if d.collection == 0:
                    if t in self.tfs_baseball:
                        self.tfs_baseball[t] += tf
                        self.tfs[t] += tf
                    else:
                        self.tfs_baseball[t] = tf
                        if t in self.tfs:
                            self.tfs[t] += tf
                        else:
                            self.tfs[t] = tf
                else:
                    if t in self.tfs_hockey:
                        self.tfs_hockey[t] += tf
                        self.tfs[t] += tf
                    else:
                        self.tfs_hockey[t] = tf
                        if t in self.tfs:
                            self.tfs[t] += tf
                        else:
                            self.tfs[t] = tf
    def save(self):
        with open(self.name, "wb") as f:
            pickle.dump(self, f)


    def draw_hist(self):
        hist = np.zeros([5000,])
        for i in self.tfs.values():
            hist[i] += 1
        with open('hist.csv','w',newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',')
            csvwriter.writerow(hist)


class Doc():
    def __init__(self, collection, doc):
        self.collection = collection
        self.doc = doc
        self.tfs = {}
        for t in doc:
            if t in self.tfs:
                self.tfs[t] += 1
            else:
                self.tfs[t] = 1
