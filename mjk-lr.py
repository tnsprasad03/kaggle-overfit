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
from sklearn.ensemble import GradientBoostingClassifier

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
[u'url', u'urlid', u'boilerplate', u'alchemy_category', 
u'alchemy_category_score', u'avglinksize', u'commonlinkratio_1', 
u'commonlinkratio_2', u'commonlinkratio_3', u'commonlinkratio_4',
u'compression_ratio', u'embed_ratio', u'framebased', u'frameTagRatio', 
u'hasDomainLink', u'html_ratio', u'image_ratio', u'is_news', 
u'lengthyLinkDomain', u'linkwordscore', u'news_front_page', 
u'non_markup_alphanum_characters', u'numberOfLinks', u'numwords_in_url', 
u'parametrizedLinkRatio', u'spelling_errors_ratio', u'label']
'''
snip = df['boilerplate'].apply(fromJson)

snip = np.array(snip)

target = np.array(df['label'])

vectorizer = TfidfVectorizer(max_df=0.5,
                             stop_words='english')



X = vectorizer.fit_transform(snip)
print X
snip = pd.DataFrame(X.toarray(0))
snip['numberOfLinks'] = df['numberOfLinks']


model  = LogisticRegression(C=1, penalty='l2', tol=0.01)
scores = cross_validation.cross_val_score(model, snip, target, cv=5)
print "%s -- %s" % (model.__class__, np.mean(scores))

