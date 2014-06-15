#!/usr/bin/env python

import getopt
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
from sklearn.grid_search import GridSearchCV

from sklearn.svm import SVC

param_grid_lr = [
        {'C': [ 1,1.5,2,3,4,5], 'penalty': ['l2'] },
]

param_grid_svc = [
        {'C': [ 1,1.5,2,3,4,5], 'gamma': ['0', '.1', '1'], 'kernel': ['rbf', 'linear', 'poly'] },
]

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
#print df.columns
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


vectorizer = TfidfVectorizer(stop_words='english')

X = vectorizer.fit_transform(snip)
print "Vectorization complete"

words = vectorizer.get_feature_names()
print len(words)
snip = pd.DataFrame(X.toarray(0))
for i in ['html_ratio', 'numberOfLinks', 'non_markup_alphanum_characters', 'frameTagRatio', 'avglinksize', 'spelling_errors_ratio', 'linkwordscore', 'commonlinkratio_2', 'parametrizedLinkRatio', 'commonlinkratio_1', 'commonlinkratio_3', 'image_ratio']:
        snip[i] = df[i]




#scores = cross_validation.cross_val_score(model, snip, target, cv=5)
#print "%s -- %s" % (model.__class__, np.mean(scores))



if __name__ == "__main__":

    try:
        opts, args = getopt.getopt(sys.argv[1:], "ls", ["logistic", "svm"])
    except getopt.GetoptError as err:
        # print help information and exit:
        print str(err) # will print something like "option -a not recognized"
        sys.exit(2)
    output = None
    verbose = False
    model = None
    param_grid = None
    for o, a in opts:
        if o == "-v":
            verbose = True
        elif o in ("-l", "--logistic"):
                model  = LogisticRegression()
                param_grid = param_grid_lr
        elif o in ("-s", "--svm"):
                model  = SVC()
                param_grid = param_grid_svc
    else:
            assert False, "unhandled option"
            sys.exit()
    # ...

    clf = GridSearchCV(model, param_grid, n_jobs=1, verbose=10)
    clf.fit(snip, target)


