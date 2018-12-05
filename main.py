
import csv, pickle
from utils import Corpus, Extractor, TopicModel
import os, sys

if __name__ == '__main__':

    corpus0 = Corpus(0)
    ldam0 = TopicModel(corpus0, corpus0.lda)

    corpus1 = Corpus(1)
    ldam1 = TopicModel(corpus1, corpus1.lda)

    corpus2 = Corpus(2)
    ldam2 = TopicModel(corpus2, corpus2.lda)



    # hdpm = TopicModel(corpus, corpus.hdp)
    # ldasm = TopicModel(corpus, corpus.ldaseq)

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
