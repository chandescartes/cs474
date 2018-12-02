from utils import Corpus, Extractor, clean
import csv, pickle
import os, sys

from collections import defaultdict

if __name__ == '__main__':
    # corpus = Corpus()
    # lda = corpus.lda

    # topics = lda.get_topics()

    # print(topics[:10])

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

    articles = pickle.load(open(os.path.join("dumps", "cleaned.bin"), "rb"))

    for i, article in enumerate(articles):
    	print(article.body)
    	print(clean(article.body), end="\n\n")

    	if i == 15: break

    # sections = defaultdict(int)

    # for article in articles:
    # 	sections[article.section] += 1
    # print(sections)