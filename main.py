
import csv, pickle
from utils import Corpus, Extractor, TopicModel
import os, sys

if __name__ == '__main__':

    corpus = Corpus()
    hdpm = TopicModel(corpus, corpus.hdp)
    ldam = TopicModel(corpus, corpus.lda)
    ldasm = TopicModel(corpus, corpus.ldaseq)

    # for i in range(10):
    #     print("TEXT: ",corpus.articles[i].text)
    #     topics = hdp[corpus.articles[i].bow]
    #     for topic in topics:
    #         print(corpus.dict[topic[0]])

    # for idx, article in enumerate(corpus.articles):
    #     if idx >= 50:
    #         break
    #     print("Title: ", article.title)
    #     print("Section: ", article.section)
    #     print("Date: ", str(article.time))
    #     print("Body: ", article.body)
    #     print(article.text)


# TODO
# options = {
#     -c: build new clened.bin
#     -p: use phrase
#     -t: use title topic model
# }
