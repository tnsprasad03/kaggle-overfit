
# coding: utf-8

# In[1]:

get_ipython().magic(u'pylab inline')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import scale
from nltk.stem.porter import PorterStemmer 
from nltk.corpus import stopwords
import json
import itertools


# In[2]:

df = pd.read_csv('data/train.tsv',sep='\t',na_values='?')


# In[138]:

df.head()


# In[109]:

df = df.fillna(df.mean())


# In[3]:

y = df['label']
df = df.drop('label',axis=1)
print df.columns


# In[150]:

for cat in df['alchemy_category'].unique():
    df[str(cat)] = np.where(df['alchemy_category'] == cat, 1, 0)
df.columns


# In[155]:

df.iloc[:,-14:].head(20)


# In[6]:

# blah.shape


# In[4]:

target = np.array(y)


# In[5]:

from sklearn import linear_model
# clf = linear_model.Lasso(alpha=0.1)
# clf.fit(blah,target)
# print(clf.coef_)
# print(clf.intercept_)


# In[9]:

# alphas = np.linspace(0.01,0.1)
# coefs = []
# for i,a in enumerate(alphas):
#     clf = linear_model.Lasso(alpha=a)
#     clf.fit(blah,target)
#     coefs.append(clf.coef_)


# In[10]:

# plt.figure(figsize=(15,15))
# for i in xrange(22): #column
#     plt.plot(alphas,np.array(coefs)[:,i],label=str(i+1),linewidth=4)
# plt.legend()


# In[11]:

# shape(coefs)


# In[12]:

# coefs[16]


# In[13]:

# df.iloc[:,4:-1].head()
# commonlinkratio_3, frameTagRatio, linkwordscore


# In[6]:

def fromJson(a):
    t = ' '
    b = ' '
    j = json.loads(a)
    if j.get('title'):
        t = j['title']
    if j.get('body'):
        t = j['body']
    return t + ' ' + b


# In[7]:

snip = df['boilerplate'].apply(fromJson)
snip = np.array(snip)

engStops = stopwords.words('english')

# vectorizer = TfidfVectorizer(max_df=0.5,stop_words=engStops,norm=u'l2')
vectorizer = CountVectorizer(max_df=0.5,stop_words=engStops)
X = vectorizer.fit_transform(snip)
words = vectorizer.get_feature_names()


# In[105]:

# len(words[5919:-288])
len(words)


# In[63]:

Xmean = np.mean(X.todense(),axis=0)


# In[81]:

inds = np.array(np.argsort(Xmean)[::-1])
inds = inds[0][-10000:]


# In[92]:

Xnew = np.array([X[:,i].toarray() for i in inds])


# In[102]:

words = [words[i] for i in inds]


# In[99]:

X = Xnew[:,:,0].T


# In[19]:

alphas = np.logspace(-5,-2,num=5)
coefs_words = []
for i,a in enumerate(alphas):
    clf = linear_model.Lasso(alpha=a)
    clf.fit(X,target)
    coefs_words.append(clf.coef_)


# In[ ]:

plt.figure(figsize=(15,15))
for i in xrange(X.shape[1]): #column
    plt.plot(alphas,np.array(coefs_words)[:,i],label=str(i+1),linewidth=1)
plt.legend()


# In[27]:

set(coefs_words[3])


# In[34]:

np.argsort(coefs_words[3])[::-1][:15]


# In[35]:

word_inds = [62994, 50476, 22515, 17610, 63033,  6614, 15331, 18132, 47761, 73706, 32220, 17871, 15688, 54659, 28414]

for word_ind in word_inds:
    print vectorizer.get_feature_names()[word_ind]


# In[33]:

x = np.array([1,2,3,4,5])
np.argsort(x)[::-1][:3]


# In[8]:

from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

def find_feature_importance(y,X,features,n):
    # This is important
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    train_index,test_index = train_test_split(df.index)

    forest = RF()
    forest_fit = forest.fit(X[train_index], y[train_index])
    forest_predictions = forest_fit.predict(X[test_index])

    importances = forest_fit.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],axis=0)
    indices = np.argsort(importances)[:-n-1:-1]

    # features = df.iloc[:,4:-1].columns
    # features = words
    #print features
    # Print the feature ranking
    print("Feature ranking:")

    #print indices
    for f in range(n):
        print("%d. %s (%f) %i" % (f + 1, features[indices[f]], importances[indices[f]], indices[f]))

    # Plot the feature importances of the forest
    #import pylab as pl
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(n), importances[indices], yerr=std[indices], color="r", align="center")
    plt.xticks(range(n), indices)
    plt.xlim([-1, n])
    plt.show()
    
    featlist = []
    indlist = []
    implist = []
    for f in xrange(len(features)):
        indlist.append(f)
        featlist.append(features[f])
        implist.append(importances[f])
    return indlist, featlist, implist


# In[179]:

a = y
b = df.iloc[:,4:]
c = df.iloc[:,4:].columns
n = b.shape[1]
find_feature_importance(a,b,c,n);


# In[39]:

inds = [5100, 15100, 25100, 35100, 45100, 55100, 65100, 75100]
imp_feats = []
imp_inds = []
imp_imps = []
for ind in inds:
    Xn = X[:,ind:ind+10000].todense()
    wordsn = words[ind:ind+10000]
    iinds, ifeats, iimps = find_feature_importance(y,Xn,wordsn,10)
    imp_feats.append(ifeats)
    imp_inds.append(list(np.array(iinds)+ind))
    imp_imps.append(iimps)
imp_feats = list(itertools.chain(*imp_feats))
imp_inds = list(itertools.chain(*imp_inds))
imp_imps = list(itertools.chain(*imp_imps))


# In[40]:

df2 = pd.DataFrame(pd.Series(np.array(imp_inds)),columns=['index'])
#df2 = df2.add(pd.Series(np.array(imp_feats)),axis=0)
df2['feature'] = pd.Series(np.array(imp_feats))
df2['importance'] = pd.Series(np.array(imp_imps))
df2.head()


# In[41]:

df2.to_pickle('bagimportant.pkl')


# In[52]:

plt.plot(sorted(df2[df2['importance']>0.005]['importance']),'o')


# In[ ]:



