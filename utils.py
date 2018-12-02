import os, sys, datetime, time
import json, pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
import csv
# import matplotlib.pyplot as plt

from nltk import word_tokenize, sent_tokenize, pos_tag, ne_chunk, ne_chunk_sents
from nltk.corpus import stopwords, wordnet
from nltk.tree import Tree
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.parsing.preprocessing import *
from gensim.models import TfidfModel, HdpModel, LdaModel
from gensim.corpora import Dictionary
from gensim.models.phrases import Phrases, Phraser

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

class Corpus:
    def __init__(self, use_phraser=False):
        self.use_phraser = use_phraser
        self.name = "corpus.bin"
        if self.name in os.listdir(dir_dumps):
            print("loading corpus...")
            with open(dir_dumps+self.name, 'rb') as f:
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
            article.text = ' '.join(article.section.split(' ') + article.title_cleaned.split(' ') + article.body_cleaned.split(' '))
        if self.use_phraser:
            self.phraser = PhraserModel(corpus=self, name="tri").get_phraser()
        else:
            self.phraser = None
        self.dict = Dict(corpus=self, phraser=self.phraser).get_dict()

        print("building bag of words...")
        for article in tqdm(self.articles):
            article.bow = self.dict.doc2bow(tokenizer([article.text], phraser=self.phraser)[0])

        # self.build_tfidf()
        self.build_hdp()

    def build_tfidf(self):
        print("building tf-idf model...")
        start = time.time()
        self.tfidf = TfidfModel(corpus=self.get_bows(), dictionary=self.dict, smartirs='lpn')
        end = time.time()
        print("tfidf finished! ", end-start, " seconds")

    def build_lda(self, num_topics):
        pass

    def build_hdp(self):
        print("building hdp model...")
        start = time.time()
        self.hdp = HdpModel(corpus=self.get_bows(), id2word=self.dict)
        end = time.time()
        print("hdp finished! ", end-start, " seconds")

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

class PhraserModel:
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

class Dict:
    def __init__(self, corpus, name="default", phraser=None):
        self.name = "dictionary_" + name + ".bin"
        self.corpus = corpus
        self.phraser = phraser
        if self.name in os.listdir(dir_dumps):
            with open(dir_dumps+self.name, 'rb') as dic:
                self.dict = pickle.load(dic)
            print("dictionary loaded")
        else:
            print("building dictionary...")
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


# tf idf 값을 통해 키워드를 추출함 - 점수 자체도 고려
class Extractor:
    def __init__(self, corpus):
        self.corpus = corpus
        self.tfidf = corpus.tfidf
        self.dict = corpus.dict
        self.phraser = corpus.phraser
        self.articles = corpus.articles
        self.bows = corpus.get_bows()

    def extract(self, k=10):
        print("extracting keywords...")

        for article in tqdm(self.articles):
            check = {}
            vector = self.tfidf[article.bow]
            for word, score in [(self.dict[i[0]], i[1]) for i in vector]:
                if score < 0:
                    continue
                if word not in check.keys():
                    check[word] = np.log(score)
                    continue
                check[word] += np.log(score)
            keywords = [(np.log(score) / np.log(2), word) for word, score in check.items() if score > 0]
            keywords.sort(reverse=True)
            article.keywords = keywords[:k]


# class FastTextModel:
#     def __init__(self, name="default", phraser=None):
#         self.name = "embedding_" + name + ".model"
#         self.phraser = phraser
#         if self.name in os.listdir(dir_embedding):
#             self.get_embedding = FastText.load(dir_embedding+self.name)
#             print("Embedding {} loaded".format(name))
#         else:
#             print("embedding not exists")
#             print("start building...")
#             self.build_embedding()
#             self.save()
#
#     def build_embedding(self):
#         tickers = [i for i in os.listdir(dir_cleaned_news) if i.endswith(".csv")]
#         tokenized_docs = []
#         start = time.time()
#         for ticker in tickers:
#             df = pd.read_csv(dir_cleaned_news + ticker)
#             tokenized_docs += tokenizer(df['content'], self.phraser)
#
#         self.get_embedding = FastText(tokenized_docs, sg=1, hs=1)
#         end = time.time()
#
#         print("train finished! ", end-start, " seconds")
#
#     # 저장
#     def save(self):
#         self.get_embedding.save(dir_embedding+self.name)
#         print("saved!")

class Window:
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
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    pos_to_wordnet = {
        'JJ'    : wordnet.ADJ,
        'JJR'   : wordnet.ADJ,
        'JJS'   : wordnet.ADJ,
        'RB'    : wordnet.ADV,
        'RBR'   : wordnet.ADV,
        'RBS'   : wordnet.ADV,
        'NN'    : wordnet.NOUN,
        'NNP'   : wordnet.NOUN,
        'NNS'   : wordnet.NOUN,
        'NNPS'  : wordnet.NOUN,
        'VB'    : wordnet.VERB,
        'VBG'   : wordnet.VERB,
        'VBD'   : wordnet.VERB,
        'VBN'   : wordnet.VERB,
        'VBP'   : wordnet.VERB,
        'VBZ'   : wordnet.VERB,
    }

    text_stripped = strip_multiple_whitespaces(re.sub(r'\d+', '', text))
    chunks = ne_chunk(pos_tag(word_tokenize(text_stripped)))
    chunks = [p for p in chunks \
        if isinstance(p, Tree) \
        or (p[1] in pos_to_wordnet and strip_non_alphanum(p[0]).strip())]

    cleaned = []

    for chunk in chunks:
        if isinstance(chunk, Tree):
            cleaned.append("^{}".format("_".join([w for w, p in chunk.leaves()])))
        else:
            if chunk[1] in pos_to_wordnet:
                wordnet_pos = pos_to_wordnet.get(chunk[1])
                cleaned.append(lemmatizer.lemmatize(chunk[0], wordnet_pos))
            else:
                cleaned.append(lemmatizer.lemmatize(chunk[0]))

    return " ".join(cleaned)

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
