# Issue Trend Analysis and Issue Tracking
# Copyright (C) CS474 Team #4 


import csv, pickle
import utils
from utils import Corpus, Issue, Extractor, IssueModel, EventModel
import os, sys
import warnings

warnings.filterwarnings('ignore')

years = [2015, 2016, 2017]
num_issues = 50
num_events = 50
num_keywords = 10


# ----------------------------------------------------

# ## Part 0: Preprocessing

# ### Clean articles: lemmatize, remove stopwords (Already Done)
# **_Caution!_ This overwrites dumps/cleaned.bin file. ** (Involves multiprocessing)


# utils.clean_articles()

# ### Detect Entities: IBM Watson NLU (Already Done)
# **_Caution!_ This overwrites dumps/cleaned_watson.bin file. ** (Involves multiprocessing and Watson API calls)


# utils.build_watson()


# Part 1: Issue Trend Analysis

# Initialize Corpuses
corpus = {}
for year in years:
    corpus[year] = Corpus(year=year-2015)


# Build Corpuses: Load cleaned articles, build phrasers, dictionary, and BOWs
for year in years:
    print("Corpus "+str(year)+":")
    corpus[year].build_corpus()
    print("Corpus "+str(year)+" Done\n")


# Extract Keywords from each Article using tf-ifd
for year in years:
    print("Corpus "+str(year)+":")
    corpus[year].build_tfidf()
    corpus[year].extractor = Extractor(corpus[year])
    corpus[year].extractor.extract(k=num_keywords)
    print("Corpus "+str(year)+" Done\n")


# Build LDA model, cluster articles into issues
for year in years:
    print("Corpus "+str(year)+":")
    corpus[year].build_lda(num_topics=num_issues)
    corpus[year].issue_model = IssueModel(corpus=corpus[year], model=corpus[year].lda)
    corpus[year].issue_model.build_issues()
    print("Corpus "+str(year)+" Done\n")


# Init Issues (for Issue Tracking)
issues = []
for year in years:
    issue_model = corpus[year].issue_model
    top_issue_id = issue_model.sorted_issues[0][0]
    issues.append(Issue(articles=issue_model.issues[top_issue_id], keywords=issue_model.keywords[top_issue_id]))



# ----------------------------------------------------

# ### Show Results


# Select year to show
show_year = 2017


# Show top trending issues 
corpus[show_year].issue_model.show_top_issues()


# Show Articles from Top Issues 
corpus[show_year].issue_model.show_issues(k=5)


# ----------------------------------------------------

# ## Part 2: Issue Tracking


# Build Issues
for i, issue in enumerate(issues):
    print("Issue "+str(i+1)+":")
    issue.build_issue()
    print("Issue "+str(i+1)+" Done\n")


# Extract keywords from each Article using tf-idf
for i, issue in enumerate(issues):
    print("Issue "+str(i+1)+":")
    issue.build_tfidf()
    issue.extractor = Extractor(issue)
    issue.extractor.extract(k=num_keywords)
    print("Issue "+str(i+1)+" Done\n")

    
# Build LDA model, cluster articles into events
for i, issue in enumerate(issues):
    print("Issue "+str(i+1)+":")
    issue.build_lda(num_topics=num_events)
    issue.event_model = EventModel(issue=issue, model=issue.lda)
    issue.event_model.build_events(threshold=0.5)
    print("Issue "+str(i+1)+" Done\n")


# Divide events into set of independent events
threshold=[0.15, 0.15, 0.15]
for i, issue in enumerate(issues):
    print("Issue "+str(i+1)+":")
    issue.event_model.build_independents(threshold=threshold[i])
    issue.event_model.filter_events(k=5)
    issue.event_model.build_event_times()
    issue.event_model.build_sorted_independents()
    issue.event_model.build_event_details()
    print("Issue "+str(i+1)+" Done\n")


# ----------------------------------------------------

# ### Show Results

# Select Issue
i = 2


# Show Issue Keywords
print(', '.join([keyword[0] for keyword in issues[i].keywords]))


# Show Top Events  
issues[i].event_model.show_top_events()


# Show Articles from Top Events 
issues[i].event_model.show_events(k=5)


# Show Issue Summary
issues[i].event_model.show_issue_summary(num_entities=10)

