from utils import Corpus, Extractor
import csv
import os, sys

if __name__ == '__main__':
    corpus = Corpus()
    Extractor(corpus).extract()
    for idx, article in enumerate(corpus.articles):
        if idx >= 50:
            break
        print("Title: ", article.title)
        print("Section: ", article.section)
        print("Date: ", str(article.time))
        print("Body: ", article.body)
        print(article.text)
        print(article.keywords)

    # print('total_number_of_words:', len(corpus.dict))
    # print(corpus.dict)
    # while True:
    #     word = input(":")
    #     try:
    #         tmp = corpus.dict.token2id[word]
    #         print(tmp)
    #     except:
    #         print("word not found!")
