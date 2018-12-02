from utils import Corpus, Extractor, TopicModel
import csv
import os, sys

if __name__ == '__main__':
    corpus = Corpus(use_title=True)
    hdp = corpus.hdp
    tm = TopicModel(corpus, hdp)

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


    # print('total_number_of_words:', len(corpus.dict))
    # print(corpus.dict)
    # while True:
    #     word = input(":")
    #     try:
    #         tmp = corpus.dict.token2id[word]
    #         print(tmp)
    #     except:
    #         print("word not found!")




# TODO
# options = {
#     -c: build new clened.bin
#     -p: use phrase
#     -t: use title topic model
# }
