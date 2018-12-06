
import os, pickle
import multiprocessing

from math import ceil
from tqdm import tqdm

from watson_developer_cloud import NaturalLanguageUnderstandingV1
from watson_developer_cloud.natural_language_understanding_v1 \
  import Features, EntitiesOptions, KeywordsOptions

from utils import Article


def load_model(path):
    with open(path, 'rb') as f:
        print("loading model...")
        model = pickle.load(f)
        return model

def call_watson(chunk, return_dict, index):
    for article in tqdm(chunk):
        response = natural_language_understanding.analyze(
          text=article.body,
          features=Features(
            entities=EntitiesOptions(),
            keywords=KeywordsOptions(limit=10))
          ).get_result()

        article.set_keywords(response.get("keywords"))
        article.set_entities(response.get("entities"))
    return_dict[index] = chunk

if __name__ == '__main__':

    natural_language_understanding = NaturalLanguageUnderstandingV1(
        version='2018-03-16',
        username='edd1f2ac-7ee6-43b9-bb78-b4f67a887df3',
        password='ybL1wtGSWZv2'
    )

    articles = load_model(os.path.join("dumps", "cleaned.bin"))

    n_processes = 8
    chunk_size = ceil(len(articles) / n_processes)
    processes = []
    return_dict = multiprocessing.Manager().dict()

    for i in range(n_processes):
        chunk = articles[chunk_size*i : chunk_size*(i+1)]
        process = multiprocessing.Process(target=call_watson, args=(chunk, return_dict, i))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    articles = []
    for i in range(n_processes):
        articles += return_dict.get(i)

    # print(articles)
    print("Saved {} articles".format(len(articles)))
    pickle.dump(articles, open(os.path.join("dumps", "cleaned_watson.bin"), "wb+"))
    # articles = pickle.load(open(os.path.join("dumps", "cleaned_watson.bin"), "rb"))
    # print(articles[0].entities)

