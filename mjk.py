#!/usr/bin/env python


import sys
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans,MiniBatchKMeans
from math import sqrt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
import json
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation

#df = pd.read_pickle("articles.pkl")
df = pd.read_csv("data/train.tsv", delimiter='\t')
print df.columns
'''
[u'document_type', u'web_url', u'lead_paragraph', u'abstract',
u'snippet', u'news_desk', u'word_count', u'source', u'section_name',
u'subsection_name', u'_id', u'pub_date', u'print_page', u'headline',
u'content'],
'''

snip = np.array(df['boilerplate'])
target = np.array(df['label'])
#print df['content'].head()

# title = json.loads(snip)['title']
# titlelen = len(title)/3
snip = json.loads(snip)['title'] + ' ' + json.loads(snip)['body']

vectorizer = TfidfVectorizer(max_df=0.5,
                             stop_words='english')


X = vectorizer.fit_transform(snip)

clf_lr  = LogisticRegression(C=1, penalty='l1', tol=0.01)
scores = cross_validation.cross_val_score(clf_lr, X, target, cv=5)
print "%s -- %s" % (clf_lr.__class__, np.mean(scores))