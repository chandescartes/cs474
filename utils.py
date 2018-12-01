import os, sys, datetime, time
import json, pickle
import pandas as pd
import numpy as np

from tqdm import tqdm
from gensim.parsing.preprocessing import *
from gensim.utils import lemmatize
from gensim.corpora.wikicorpus import WikiCorpus
from gensim.models import TfidfModel, FastText
from gensim.corpora import Dictionary
from gensim.models.phrases import Phrases, Phraser
import numpy as np
# import matplotlib.pyplot as plt
import csv

dir_dumps = "dumps/"
dir_data = "data/"

class Article:
    def __init__(self, title, author, time, body, section):
        self.title = title
        self.author = author
        self.time = time
        self.body = body
        self.section = section

        self.title_cleaned = clean(self.title)
        self.body_cleaned = clean(self.body)

    def __repr__(self):
        return "<Doc '{}', '{}'>".format(self.title, self.author)

    def __str__(self):
        return self.__repr__()

class Corpus():
    def __init__(self):
        self.name = "corpus.bin"
        if self.name in os.listdir(dir_dumps):
            with open(self.name, 'rb') as f:
                c = pickle.load(f)
                self.articles = c.articles
                self.phraser = c.phraser
                self.dict = c.dict
                self.tfidf = c.tfidf
            print("corpus loaded")
        else:
            self.build_corpus()
            self.save()

    def build_corpus(self):
        print("building corpus...")
        self.tfidf = None
        if "cleaned.bin" not in os.listdir(dir_dumps):
            clean_articles()
        with open(dir_dumps+"cleaned.bin", 'rb') as f:
            print("collecting articles...")
            self.articles = pickle.load(f)
        for article in tqdm(self.articles):
            article.text = article.section + article.title_cleaned + article.body_cleaned

        self.phraser = PhraserModel(corpus=self, name="tri").get_phraser()
        self.dict = Dict(corpus=self, phraser=self.phraser).get_dict()

        print("building bows...")
        for article in tqdm(self.articles):
            article.bow = self.dict.doc2bow(tokenizer([article.text], phraser=self.phraser)[0])

        self.build_tfidf()

    def build_tfidf(self):
        print("building tf-idf model...")
        self.tfidf = TfidfModel(self.get_bows(), dictionary=self.dict, smartirs='atn')

    def get_tfidf(self):
        return self.tfidf

    def get_texts(self):
        return [article.text for article in self.articles]

    def get_bows(self):
        return [article.bow for article in self.articles]

    def save(self):
        with open(dir_dumps+self.name, "wb") as f:
            pickle.dump(self, f)
        print("corpus saved")

    # def draw_hist(self):
    #     hist = np.zeros([5000,])
    #     for i in self.tfs.values():
    #         hist[i] += 1
    #     with open('hist.csv','w',newline='') as csvfile:
    #         csvwriter = csv.writer(csvfile, delimiter=',')
    #         csvwriter.writerow(hist)

class PhraserModel():
    def __init__(self, corpus, name="tri"):
        self.name = "phraser_" + name + ".bin"
        self.corpus = corpus
        if self.name in os.listdir(dir_dumps):
            with open(dir_dumps + self.name, "rb") as p:
                print(self.name, " loaded")
                self.phraser = pickle.load(p)
        else:
            print("phraser does not exist")
            print("start building...")
            self.build_bigram_phraser()
            self.save_bigram()
            with open(dir_dumps + "phraser_bi.bin", "rb") as p:
                self.bi_phraser = pickle.load(p)
            self.build_trigram_phraser()
            self.save_trigram()

    def build_bigram_phraser(self):
        texts = self.corpus.get_texts()
        tokens = []
        start = time.time()
        tokens += tokenizer(texts)
        self.phraser = Phraser(Phrases(tokens))
        end = time.time()
        print("bigram train finished! ", end-start, " seconds")

    def build_trigram_phraser(self):
        print('training trigram...')
        texts = self.corpus.get_texts()
        tokens = []
        start = time.time()
        tokens += self.bi_phraser[tokenizer(texts)]
        self.phraser = Phraser(Phrases(tokens))
        end = time.time()
        print("train finished! ", end-start, " seconds")

    def get_phraser(self):
        return self.phraser

    def save_bigram(self):
        with open(dir_dumps + "phraser_bi.bin", "wb") as p:
            pickle.dump(self.phraser, p)
        print("saved!")

    def save_trigram(self):
        with open(dir_dumps + self.name, "wb") as p:
            pickle.dump(self.phraser, p)
        print("saved!")

class Dict():
    def __init__(self, corpus, name="default", phraser=None):
        self.name = "dictionary_" + name + ".bin"
        self.corpus = corpus
        self.phraser = phraser
        if self.name in os.listdir(dir_dumps):
            with open(dir_dumps+self.name, 'rb') as dic:
                self.dict = pickle.load(dic)
            print("keyword dictionary loaded")
        else:
            print("dictionary not exists")
            print("start building...")
            self.build_dictionary()
            self.save()

    def build_dictionary(self):
        self.dict = Dictionary()
        texts = self.corpus.get_texts()
        self.dict.add_documents(tokenizer(texts, phraser=self.phraser))

    def get_dict(self):
        return self.dict

    def save(self):
        with open(dir_dumps+self.name, "wb") as dic:
            pickle.dump(self.dict, dic)

class FastTextModel:
    def __init__(self, name="default", phraser=None):
        self.name = "embedding_" + name + ".model"
        self.phraser = phraser
        if self.name in os.listdir(dir_embedding):
            self.get_embedding = FastText.load(dir_embedding+self.name)
            print("Embedding {} loaded".format(name))
        else:
            print("embedding not exists")
            print("start building...")
            self.build_embedding()
            self.save()

    def build_embedding(self):
        tickers = [i for i in os.listdir(dir_cleaned_news) if i.endswith(".csv")]
        tokenized_docs = []
        start = time.time()
        for ticker in tickers:
            df = pd.read_csv(dir_cleaned_news + ticker)
            tokenized_docs += tokenizer(df['content'], self.phraser)

        self.get_embedding = FastText(tokenized_docs, sg=1, hs=1)
        end = time.time()

        print("train finished! ", end-start, " seconds")

    # 저장
    def save(self):
        self.get_embedding.save(dir_embedding+self.name)
        print("saved!")

class Window():
    def __init__(self, start_date, delta, articles):
        self.start_date = start_date
        self.end_date = start_date + delta # FIXME
        self.articles = [article for article in articles if article.date > self.start_date and sarticle.date < self.end_date]

def clean_articles():
    print("cleaning articles...")
    dfs = []
    for file in os.listdir("data"):
        if file.endswith(".json"):
            with open(os.path.join("data", file), "r") as f:
                dfs.append(pd.DataFrame.from_dict(json.load(f)))

    articles = []

    for df in dfs:
        for i in tqdm(range(df.shape[0])):
            title = df['title'][i].strip()
            author = df[' author'][i].strip()
            time = datetime.datetime.strptime(df[' time'][i].strip(), "%Y-%m-%d %H:%M:%S")
            body = df[' body'][i].strip()
            section = df[' section'][i].strip()

            article = Article(title, author, time, body, section)
            articles.append(article)

    path = os.path.join(dir_dumps, "cleaned.bin")
    print("Saved {} articles into {}".format(len(articles), path))
    pickle.dump(articles, open(path, "wb+"))

def clean(text):
    blacklist = ['be', 'do', 're']
    try:
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

def tokenizer(texts, phraser=None):
    tokenized = []
    for text in texts:
        if type(text) is str:
            if phraser:
                tokenized += [token for token in phraser[[text.split()]]]
            else:
                tokenized += [text.split()]
        else:
            continue
    return tokenized
