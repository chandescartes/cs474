import os, sys, datetime, time
import json, pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
import csv
from itertools import combinations
from math import sqrt
# import matplotlib.pyplot as plt

from gensim.parsing.preprocessing import *
from gensim.corpora import Dictionary
from gensim.models import TfidfModel, HdpModel, LdaModel, LsiModel
from gensim.summarization.summarizer import summarize, summarize_corpus

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

    def set_entities(self, entities):
        # List of maps. Keys: text, relevance, count
        self.entities = entities

    def set_keywords(self, keywords):
        # List of maps. Keys: type, text, relevance, disambiguation
        self.keywords = keywords

class Corpus:
    def __init__(self, year, use_phraser=True, title_weight=5):
        self.use_phraser = use_phraser
        self.title_weight = title_weight
        self.year = year
        
    def build_corpus(self):
        print("building corpus...")
        with open(dir_dumps+"cleaned_watson.bin", 'rb') as f:
            print("collecting articles...")
            articles = pickle.load(f)
            articles.sort(key=lambda a: a.time)

        start_time = datetime.datetime(2015+self.year,1,1,0,0,0)
        end_time = datetime.datetime(2015+self.year+1,1,1,0,0,0)
        
        self.articles = []
        for article in articles:
            if article.time < start_time:
                continue
            elif article.time > end_time:
                break
            else:
                title = article.title_cleaned.split() * self.title_weight
                article.text = ' '.join(article.section.split() + title + article.body_cleaned.split())
                self.articles.append(article)
                
        if self.use_phraser:
            self.phraser = PhraserModel(corpus=self, name="tri").get_phraser()
        else:
            self.phraser = None
            
        self.dict = Dict(corpus=self, phraser=self.phraser).get_dict()

        print("building bag of words...")
        time.sleep(1)
        
        for article in tqdm(self.articles):
            article.bow = self.dict.doc2bow(tokenizer([article.text], phraser=self.phraser)[0])


    def build_tfidf(self):
        print("building tf-idf model...")
        start = time.time()
        self.tfidf = TfidfModel(corpus=self.get_bows(), dictionary=self.dict, smartirs='lpn')
        end = time.time()
        print("tfidf finished! {:.2f} seconds".format(end-start))

    def build_lda(self, num_topics=100):
        corpus = self.get_bows()
        print("building LDA model...")
        start = time.time()
        self.lda = LdaModel(corpus=corpus, num_topics=num_topics, id2word=self.dict)
        end = time.time()
        print("LDA finished! {:.2f} seconds".format(end-start))

    def build_lsi(self, num_topics=100):
        corpus = self.get_bows()
        print("building LSI model...")
        start = time.time()
        self.lsi = LsiModel(corpus=corpus, num_topics=num_topics, id2word=self.dict)
        end = time.time()
        print("LSI finished! {:.2f} seconds".format(end-start))

    def build_hdp(self):
        print("building HDP model...")
        start = time.time()
        self.hdp = HdpModel(corpus=self.get_bows(), id2word=self.dict)
        end = time.time()
        print("HDP finished! {:.2f} seconds".format(end-start))

    def get_keywords(self):
        return [article.keywords for article in self.articles]

    def get_texts(self):
        return [article.text for article in self.articles]

    def get_bows(self):
        return [article.bow for article in self.articles]


class Issue:
    def __init__(self, articles, keywords, use_phraser=True):
        self.articles = articles
        self.use_phraser = use_phraser
        self.keywords = keywords

    def build_issue(self):
        print("building issue...")
        
        if self.use_phraser:
            self.phraser = PhraserModel(corpus=self, name="tri").get_phraser()
        else:
            self.phraser = None
            
        self.dict = Dict(corpus=self, phraser=self.phraser).get_dict()

        print("building bag of words...")
        time.sleep(1)
        
        for article in tqdm(self.articles):
            article.bow = self.dict.doc2bow(tokenizer([article.text], phraser=self.phraser)[0])

    def build_tfidf(self):
        print("building tf-idf model...")
        start = time.time()
        self.tfidf = TfidfModel(corpus=self.get_bows(), dictionary=self.dict, smartirs='lpn')
        end = time.time()
        print("tfidf finished! {:.2f} seconds".format(end-start))

    def build_lda(self, num_topics=100):
        corpus = self.get_bows()
        print("building LDA model...")
        start = time.time()
        self.lda = LdaModel(corpus=corpus, num_topics=num_topics, id2word=self.dict)
        end = time.time()
        print("LDA finished! {:.2f} seconds".format(end-start))

    def build_lsi(self, num_topics=100):
        corpus = self.get_bows()
        print("building LSI model...")
        start = time.time()
        self.lsi = LsiModel(corpus=corpus, num_topics=num_topics, id2word=self.dict)
        end = time.time()
        print("LSI finished! {:.2f} seconds".format(end-start))

    
    def get_keywords(self):
        return [article.keywords for article in self.articles]

    def get_texts(self):
        return [article.text for article in self.articles]

    def get_bows(self):
        return [article.bow for article in self.articles]

     
class PhraserModel:
    def __init__(self, corpus, name="tri"):
        self.corpus = corpus
        print("building phrasers...")
        self.build_bigram_phraser()
        self.bi_phraser = self.phraser
        self.build_trigram_phraser()
        
    def build_bigram_phraser(self):
        from gensim.models.phrases import Phrases, Phraser
        texts = self.corpus.get_texts()
        tokens = []
        start = time.time()
        tokens += tokenizer(texts)
        self.phraser = Phraser(Phrases(tokens))
        end = time.time()
        print("bigram train finished! {:.2f} seconds".format(end-start))

    def build_trigram_phraser(self):
        from gensim.models.phrases import Phrases, Phraser
        texts = self.corpus.get_texts()
        tokens = []
        start = time.time()
        tokens += self.bi_phraser[tokenizer(texts)]
        self.phraser = Phraser(Phrases(tokens))
        end = time.time()
        print("trigram train finished! {:.2f} seconds".format(end-start))

    def get_phraser(self):
        return self.phraser


class Dict:
    def __init__(self, corpus, phraser=None):
        self.corpus = corpus
        self.phraser = phraser
        self.build_dictionary()

    def build_dictionary(self):
        print("building dictionary...")
        self.dict = Dictionary()
        texts = self.corpus.get_texts()
        self.dict.add_documents(tokenizer(texts, phraser=self.phraser))

    def get_dict(self):
        return self.dict

class Extractor:
    def __init__(self, corpus):
        self.corpus = corpus
        self.tfidf = corpus.tfidf
        self.dict = corpus.dict
        self.articles = corpus.articles
        self.bows = corpus.get_bows()

    def extract(self, k=10):
        print("extracting keywords...")
        time.sleep(1)
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
            article.tfidf_keywords = keywords[:k]

class IssueModel:
    def __init__(self, corpus, model):
        self.corpus = corpus
        self.model = model

    def build_issues(self, threshold=0.5):
        print("building issues...")
        time.sleep(1)
        self.num_issues = self.model.get_topics().shape[0]
        self.issues = [[] for i in range(self.num_issues)]
        self.issue_scores = {}
        

        for i in range(self.num_issues):
            self.issue_scores[i] = 0
            
        articles = self.corpus.articles
        
        # issue_sums = [0] * self.num_issues
        # for article in articles:
        #     vector = self.model[article.bow]
        #     for e in vector:
        #         issue_sums[e[0]] += abs(e[1])

        for article in tqdm(articles):
            vector = self.model[article.bow]
            article.issue_scores = {}
            vector.sort(key=lambda p : p[1], reverse=True)
            for e in vector:
                self.issue_scores[e[0]] += sqrt(e[1])  # FIXME
                article.issue_scores[e[0]] = sqrt(e[1]) # FIXME
            if vector[0][1] > threshold:
                self.issues[vector[0][0]].append(article)
            
                    

            # norm_vector = []
            # for e in vector:
            #     norm_vector.append((e[0],e[1] / sqrt(issue_sums[e[0]])))
            # norm_vector.sort(key=lambda p : p[1], reverse=True)
            # article.issue_scores = {}
            # for e in norm_vector:
            #     self.issue_scores[e[0]] += e[1]
            # self.issues[norm_vector[0][0]].append(article)
            # article.issue_scores[norm_vector[0][0]] = norm_vector[0][1]
            
                    
        self.sorted_issues = list(self.issue_scores.items())
        self.sorted_issues.sort(key=lambda p:p[1],reverse=True)
        

        # self.build_summaries()
        self.extract_keywords()


    def build_summaries(self):
        self.summaries = []
        print("summarizing collections...")
        time.sleep(1)
        for i in tqdm(range(self.num_issues)):
            if len(self.issues[i]) == 0:
                summary = ''
            elif len(self.issues[i]) == 1:
                summary = self.issues[i][0].title_cleaned
            else:
                text = ''
                for article in self.issues[i]:
                    text = '\n'.join([artcle.title_cleaned for article in self.collections[i]])
                summary = summarize(text, word_count=5)
            self.summaries.append(summary)

    def extract_keywords(self):
        self.keywords = []
        print("extracting keywords...")
        time.sleep(1)
        for i in tqdm(range(self.num_issues)):
            dic = {}
            for article in self.issues[i]:
                for keyword in article.tfidf_keywords:
                    if keyword[1] not in dic:
                        dic[keyword[1]] = article.issue_scores[i] 
                        # dic[keyword[1]] = keyword[0]
                    else:
                        dic[keyword[1]] += article.issue_scores[i] 
                        # dic[keyword[1]] += keyword[0]
            sorted_dic = list(dic.items())
            sorted_dic.sort(key=lambda p:p[1],reverse=True)
            self.keywords.append(sorted_dic[:10])

    def show_issues(self, topn=10, k=20):
        cnt = 0
        for i, val in self.sorted_issues:
            if cnt >= topn:
                break
            print("ID: {:3d} Score: {:6.2f} N: {:4d} Keywords: ".format(i, val, len(self.issues[i])), ', '.join(keyword[0] for keyword in self.keywords[i]))
            for j, article in enumerate(self.issues[i]):
                if j >= k:
                    break
                print("\t",j,"\t",article.title)
            cnt += 1

    def show_issue_names(self, topn=10):
        cnt = 0
        for i, val in self.sorted_issues:
            if cnt >= topn:
                break
            print("ID: {:3d} Score: {:6.2f} N: {:4d} Topic: ".format(i, val, len(self.issues[i])), self.model.print_topic(i))
            cnt += 1
    def show_top_issues(self, topn=10):
        cnt = 0
        for i, val in self.sorted_issues:
            if cnt >= topn:
                break;
            print("ID: {:3d} Score: {:6.2f} N: {:4d} Keywords: ".format(i, val, len(self.issues[i])), ', '.join(keyword[0] for keyword in self.keywords[i]))
            cnt += 1

class EventModel:
    def __init__(self, issue, model):
        self.issue = issue
        self.model = model

    def build_events(self, threshold=0.5):
        print("building events...")
        time.sleep(1)
        self.num_events = self.model.get_topics().shape[0]
        self.events = [[] for i in range(self.num_events)]
        self.event_scores = {}
        
        
        dependency = np.zeros((self.num_events, self.num_events))
        self.union_cnt = np.zeros((self.num_events, self.num_events))
        
        for i in range(self.num_events):
            self.event_scores[i] = 0
        articles = self.issue.articles


        for article in tqdm(articles):
            vector = self.model[article.bow]
            article.event_scores = {}
            vector.sort(key=lambda p : p[1], reverse=True)
            for e in vector:
                self.event_scores[e[0]] += sqrt(e[1])
                article.event_scores[e[0]] = sqrt(e[1])
            if vector[0][1] > threshold:
                self.events[vector[0][0]].append(article)            
            
            for pair in combinations(vector, r=2):
                dependency[pair[0][0]][pair[1][0]] += sqrt(pair[0][1])*sqrt(pair[1][1])
                dependency[pair[1][0]][pair[0][0]] += sqrt(pair[0][1])*sqrt(pair[1][1])
                self.union_cnt[pair[0][0]][pair[1][0]] += 1
                self.union_cnt[pair[1][0]][pair[0][0]] += 1
                    
        self.sorted_events = list(self.event_scores.items())
        self.sorted_events.sort(key=lambda p:p[1],reverse=True)
        

        self.dependency = np.divide(dependency, self.union_cnt)
        np.nan_to_num(self.dependency, copy=False)

        # self.build_summaries()
        self.extract_keywords()

    def build_summaries(self):
        self.summaries = []
        print("summarizing collections...")
        time.sleep(1)
        for i in tqdm(range(self.num_events)):
            if len(self.events[i]) == 0:
                summary = ''
            elif len(self.events[i]) == 1:
                summary = self.events[i][0].title_cleaned
            else:
                text = ''
                for article in self.events[i]:
                    text = '\n'.join([artcle.title_cleaned for article in self.events[i]])
                summary = summarize(text, word_count=5)
            self.summaries.append(summary)

    def extract_keywords(self):
        self.keywords = []
        print("extracting keywords...")
        time.sleep(1)
        for i in tqdm(range(self.num_events)):
            dic = {}
            for article in self.events[i]:
                for keyword in article.tfidf_keywords:
                    if keyword[1] not in dic:
                        dic[keyword[1]] = article.event_scores[i] # FIXME appearnace or tfidf?
                        # dic[keyword[1]] = keyword[0]
                    else:
                        dic[keyword[1]] += article.event_scores[i] #FIXME appearance or tfidf?
                        # dic[keyword[1]] += keyword[0]
            sorted_dic = list(dic.items())
            sorted_dic.sort(key=lambda p:p[1],reverse=True)
            self.keywords.append(sorted_dic[:10])

    def show_events(self, topn=10, k=20):
        cnt = 0
        for i, val in self.sorted_events:
            if cnt >= topn:
                break
            print("ID: {:3d} Score: {:6.2f} N: {:4d} Keywords: ".format(i, val, len(self.events[i])), ', '.join(keyword[0] for keyword in self.keywords[i]))
            for j, article in enumerate(self.events[i]):
                if j >= k:
                    break
                print("\t",j,"\t",article.title)
            cnt += 1

    def show_event_names(self, topn=10):
        cnt = 0
        for i, val in self.sorted_events:
            if cnt >= topn:
                break
            print("ID: {:3d} Score: {:6.2f} N: {:4d} Topic: ".format(i, val, len(self.events[i])), self.model.print_topic(i))
            cnt += 1
    def show_top_events(self, topn=10):
        cnt = 0
        for i, val in self.sorted_events:
            if cnt >= topn:
                break;
            print("ID: {:3d} Score: {:6.2f} N: {:4d} Keywords: ".format(i, val, len(self.events[i])), ', '.join(keyword[0] for keyword in self.keywords[i]))
            cnt += 1

    def build_independents(self, threshold=0.5):
        matrix = self.dependency
        graph = {node: set(i for i, x in enumerate(matrix[node]) if x > threshold)
            for node in range(len(matrix))}
        components = []
        for component in connected_components(graph):
            clist = []
            for node in component:
                clist.append(node)
            components.append(clist)

        self.independents = components

    def filter_events(self, k=20):
        top_events = [topic[0] for topic in self.sorted_events[:k]]
        self.filtered_independents = []
        for independent in self.independents:
            filtered_independent = []
            for event in independent:
                if event in top_events:
                    filtered_independent.append(event)
            if len(filtered_independent) > 0:
                self.filtered_independents.append(filtered_independent) 

    def build_event_times(self):
        self.event_times = {}
        print("building event times...")
        time.sleep(1)
        event_ids = []
        for independent in self.filtered_independents:
            event_ids += independent
            
        for event_id in tqdm(event_ids):
            time_sum = 0
            weight_sum = 0
            for article in self.events[event_id]:
                time_sum += int(article.time.strftime('%s')) * article.event_scores[event_id]
                weight_sum += article.event_scores[event_id]
            avg_time = time_sum / weight_sum
            self.event_times[event_id] = avg_time

    def build_sorted_independents(self):
        self.sorted_independents = []
        for independent in self.filtered_independents:
            independent.sort(key=lambda event_id:self.event_times[event_id])
            self.sorted_independents.append(independent)

    def build_event_details(self):
        self.event_details = {}
        print("building event details...")
        time.sleep(1)
        event_ids = []
        for independent in self.sorted_independents:
            event_ids += independent
            
        for event_id in tqdm(event_ids):
            event_detail = {}
            event_detail['time'] = datetime.datetime.utcfromtimestamp(self.event_times[event_id])
            event_detail['entities'] = self.get_top_entities(event_id)
            event_detail['keywords'] = self.keywords[event_id]
            self.event_details[event_id] = event_detail

    def show_independents(self):
        for independent in self.sorted_independents:
            print("\t"," -> ".join([str(event_id) for event_id in independent]))

    def show_event_details(self, event_id, num_entities):
        event_detail = self.event_details[event_id]
        print("Event ID: {:3d}".format(event_id))
        print("Event Keywords: ")
        print("\t",', '.join([keyword[0] for keyword in event_detail['keywords']]))
        print("Time: {}".format(event_detail['time'].strftime("%Y-%m-%d")))
        print("Entities: ")
        for entity in event_detail['entities'][:num_entities]:
            print("\t{}, {}".format(entity[0], entity[1].upper()))

    def show_issue_summary(self, num_entities):
        print("Issue Keywords: ")
        print("\t",', '.join([keyword[0] for keyword in self.issue.keywords]))
        print("")
        print("Events: ")
        self.show_independents()
        print("")
        event_ids = []
        for independent in self.sorted_independents:
            event_ids += independent

        for event_id in event_ids:
            print("")
            self.show_event_details(event_id, num_entities=num_entities)

    def get_top_entities(self, event_id):
        articles = self.events[event_id]
        invalid_types = set(["EmailAddress"])
        blacklist = set(["Yonhap"])
        dict_entities = dict()

        for article in articles:
            for entity in article.entities[:10]:
                if entity["type"] in invalid_types:
                    continue
                if entity["text"] in blacklist:
                    continue

                key = (entity["text"], entity["type"])
                if key in dict_entities:
                    dict_entities[key] += sqrt(article.event_scores[event_id])
                else:
                    dict_entities[key] =  sqrt(article.event_scores[event_id])
        
        entities = [(k[0], k[1], v) for k, v in dict_entities.items()]
        entities.sort(key=lambda t: t[2], reverse=True)
        return entities

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

def call_watson(chunk, return_dict, index, nlu):
    for article in tqdm(chunk):
        for i in range(3):
            try:
                response = nlu.analyze(
                  text=article.body,
                  features=Features(
                    entities=EntitiesOptions(),
                    keywords=KeywordsOptions(limit=10))
                  ).get_result()

                article.set_keywords(response.get("keywords"))
                article.set_entities(response.get("entities"))
                break
            except WatsonApiException:
                print("Method failed with status code " + str(ex.code) + ": " + ex.message)
                if i == 2:
                    print(article.body)
                    print("FATAL FATAL FATAL FATAL FATAL FATAL FATAL FATAL FATAL FATAL")

    return_dict[index] = chunk

def build_watson():
    nlu = NaturalLanguageUnderstandingV1(
        version='2018-03-16',
        username='edd1f2ac-7ee6-43b9-bb78-b4f67a887df3',
        password='ybL1wtGSWZv2'
    )

    articles = load_model(os.path.join("dumps", "cleaned.bin"))
    articles = [a for a in articles if a.body.strip()]

    n_processes = 24
    chunk_size = ceil(len(articles) / n_processes)
    processes = []
    return_dict = multiprocessing.Manager().dict()

    for i in range(n_processes):
        chunk = articles[chunk_size*i : chunk_size*(i+1)]
        process = multiprocessing.Process(target=call_watson, args=(chunk, return_dict, i, nlu))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    articles = []
    for i in range(n_processes):
        articles += return_dict.get(i)

    print("Saved {} articles".format(len(articles)))
    pickle.dump(articles, open(os.path.join("dumps", "cleaned_watson.bin"), "wb+"))

    return articles

def clean(text):
    from nltk import word_tokenize, sent_tokenize, pos_tag, ne_chunk, ne_chunk_sents
    from nltk.corpus import stopwords, wordnet
    from nltk.tree import Tree
    from nltk.stem.wordnet import WordNetLemmatizer

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    my_stop_words = ['Yonhap', 'heraldcorp.com', 'say', 'said', 'th', 'st', 'nd', 'rd']
    stop_words = stop_words.union(set(my_stop_words))
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
        or (p[1] in pos_to_wordnet
            and strip_non_alphanum(p[0]).strip()
            and p[0] not in stop_words
            and p[0][0] not in string.punctuation)]

    cleaned = []

    for chunk in chunks:
        if isinstance(chunk, Tree):
            w = "_".join([w for w, p in chunk.leaves()])
            ws = w.split(".")
            w = w if len(ws[0]) == 1 else " ".join(ws)
            if w.lower() == 'yonhap': continue
            cleaned.append(w)
        else:
            w = chunk[0][1:] if chunk[0][0] in string.punctuation else chunk[0]
            ws = [y for y in chunk[0].split(".") if y]

            if chunk[1] in pos_to_wordnet and len(ws) == 1:
                wordnet_pos = pos_to_wordnet.get(chunk[1])
                cleaned.append(lemmatizer.lemmatize(w, wordnet_pos))
            else:
                for y in ws:
                    y = lemmatizer.lemmatize(y)
                    if not y.lower() in stop_words:
                        cleaned.append(y)

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

def connected_components(neighbors):
    seen = set()
    def component(node):
        nodes = set([node])
        while nodes:
            node = nodes.pop()
            seen.add(node)
            nodes |= neighbors[node] - seen
            yield node
    for node in neighbors:
        if node not in seen:
            yield component(node)

def get_independents(matrix, threshold= 0.5):
    graph = {node: set(i for i, x in enumerate(matrix[node]) if x > threshold)
        for node in range(len(matrix))}
    components = []
    for component in connected_components(graph):
        clist = []
        for node in component:
            clist.append(node)
        components.append(clist)

    return components




def save(obj, filename):
    with open(dir_dumps+filename, "wb") as f:
        print("saving "+filename+"...")
        pickle.dump(obj, f)
        print(filename+" saved")

def load(filename):
    with open(dir_dumps+filename, 'rb') as f:
        print("loading "+filename+"...")
        obj = pickle.load(f)
        print(filename+" loaded")
        return obj