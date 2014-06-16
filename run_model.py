#!/usr/bin/env python
import sys, getopt
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans,MiniBatchKMeans
from math import sqrt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
import json
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import scale
import pdb

def createsubmission(model,vectorizer):
   traindata = list(np.array(pd.read_table('data/train.tsv'))[:,2])
   testdata = list(np.array(pd.read_table('data/test.tsv'))[:,2])
   y = np.array(pd.read_table('data/train.tsv'))[:,-1]
   X_all = traindata + testdata
   lentrain = len(traindata)
   print "fitting pipeline"
   vectorizer.fit(X_all)
   print "transforming data"
   X_all = vectorizer.transform(X_all)

   X = X_all[:lentrain]
   X_test = X_all[lentrain:]

   model.fit(X,y)
   pred = model.predict_proba(X_test)[:,1]
   testfile = pd.read_csv('data/test.tsv', sep="\t", na_values=['?'], index_col=1)
   pred_df = pd.DataFrame(pred, index=testfile.index, columns=['label'])
   pred_df.to_csv(str(model).split("(")[0] + '.csv')


def fromJson(a):
   t = ' '
   b = ' '
   j = json.loads(a)
   if j.get('title'):
      t = j['title']
   if j.get('body'):
      t = j['body']
   return t + ' ' + b


def executeModel(m):
   
   #df = pd.read_pickle("articles.pkl")
   df = pd.read_csv("data/train.tsv", na_values='?',delimiter='\t')
   #df = df[:1000]
   print df.columns
   df = df.fillna(df.mean())
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

   vectorizer = TfidfVectorizer(max_df=0.5,stop_words='english',norm=u'l2')
   #vectorizer = CountVectorizer(max_df=0.5,stop_words='english')



   X = vectorizer.fit_transform(snip).toarray()
   print X.shape

   wordsdf = pd.read_pickle('bagimportant.pkl')
   filtereddf = wordsdf[wordsdf['importance'] > 0.0001][['importance','index']]
   # print type(X[:,list(filtereddf['index'])])
   # print type(np.array(filtereddf['importance'])[np.newaxis,:])
   X = X[:,list(filtereddf['index'])] * np.exp(np.array(filtereddf['importance'])[np.newaxis,:]*100)

   print X.shape
   #pdb.set_trace()
   # snip = pd.DataFrame(X.toarray(0))
   # snip['numberOfLinks'] = df['numberOfLinks']
   
   #X2 = np.array(scale(df.iloc[:,5]))
   #X3 = np.array(scale(df.iloc[:,10]))
   #X4 = np.array(scale(df.iloc[:,16]))
   #print "Printing shape of X"+ str (X.shape)
   #X2 = np.mat(scale(df.non_markup_alphanum_characters.values)).T
   df_feats = df[['non_markup_alphanum_characters','frameTagRatio','avglinksize','spelling_errors_ratio','linkwordscore','html_ratio','compression_ratio','numberOfLinks','commonlinkratio_1','commonlinkratio_2','commonlinkratio_3','commonlinkratio_4','parametrizedLinkRatio','image_ratio','alchemy_category_score','numwords_in_url']]
   #df_feats = scale(df)
   X_feats = np.mat(df_feats.values)

   #X = np.hstack((X.toarray(),X2,X3,X4))
   #X = np.hstack((X.toarray(),X_feats))
   X = np.hstack((X,X_feats))
   #sys.exit()
   
   if m == 'lr':
      model  = LogisticRegression(C=1, penalty='l2', tol=0.0001)
      scores = cross_validation.cross_val_score(model, X, target, cv=5)
      print "%s -- %s" % (model.__class__, np.mean(scores))
      createsubmission(model, vectorizer)

   if m == 'rf':
      model = RandomForestClassifier(verbose=10, n_estimators=1, n_jobs=-1, max_features=None)
      scores = cross_validation.cross_val_score(model, X, target, cv=5)
      print "%s -- %s" % (model.__class__, np.mean(scores))
      createsubmission(model, vectorizer)


def usage():
      print 'run_model.py -m lr/rf ( lr for logistic and rf for random forest)'
def main(argv):
   model = ""

   try:
      opts, args = getopt.getopt(argv,"hm:",["lr"])
   except getopt.GetoptError:
      usage()
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         usage()
         sys.exit()
      elif opt in ("-m"):
         model = arg
   print 'Input option is %s' % (model)
   
   if (model):
      executeModel(model)
   else:
      print "Unable to run Model. Please check options"


if __name__ == "__main__":
   main(sys.argv[1:])
