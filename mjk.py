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
from sklearn.ensemble import RandomForestClassifier

def fromJson(a):
	t = ' '
	b = ' '
	j = json.loads(a)
	if j.get('title'):
		t = j['title']
	if j.get('body'):
		t = j['body']
	return t + ' ' + b

#df = pd.read_pickle("articles.pkl")
df = pd.read_csv("data/train.tsv", delimiter='\t')
print df.columns
'''
[u'document_type', u'web_url', u'lead_paragraph', u'abstract',
u'snippet', u'news_desk', u'word_count', u'source', u'section_name',
u'subsection_name', u'_id', u'pub_date', u'print_page', u'headline',
u'content'],
'''
snip = df['boilerplate'].apply(fromJson)
snip = np.array(snip)

#snip = np.array(df['boilerplate'])
target = np.array(df['label'])
#print df['content'].head()

# title = json.loads(snip)['title']
# titlelen = len(title)/3
#snip = json.loads(snip)['title'] + ' ' + json.loads(snip)['body']

vectorizer = TfidfVectorizer(max_df=0.5,
                             stop_words='english')


X = vectorizer.fit_transform(snip)

model  = LogisticRegression(C=1, penalty='l1', tol=0.01)
scores = cross_validation.cross_val_score(model, X, target, cv=5)
print "%s -- %s" % (model.__class__, np.mean(scores))

model = RandomForestClassifier(verbose=10, n_estimators=1, n_jobs=-1, max_features=None)
#model.fit(X.toarray(), target)
scores = cross_validation.cross_val_score(model, X, target, cv=5)
print "%s -- %s" % (model.__class__, np.mean(scores))
