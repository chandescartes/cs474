
import os, pickle

from utils import Article, Corpus, parse_and_save

if __name__ == '__main__':
    if not os.path.isfile(os.path.join("data", "articles.bin")):
        parse_and_save()

    # articles = pickle.load(open(os.path.join("data", "articles.bin"), "rb"))
    