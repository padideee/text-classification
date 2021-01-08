import os
import json
import numpy as np
import pandas as pd
import re, unicodedata
import nltk
nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import accuracy_score






def remove_non_ascii(words):
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return ''.join(new_words)

def remove_ponctuation(document):
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in document]
    return ''.join(stripped)

def remove_stop_words(document):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(document)
    filtered_sentence = []
    for w in word_tokens: 
        if w not in stop_words: 
            filtered_sentence.append(w)
    Stem_words = []
    ps =PorterStemmer()
    for w in filtered_sentence:
        rootWord=ps.stem(w)
        Stem_words.append(rootWord)
    return ' '.join(Stem_words)

def Lemmatize(document):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(document)
    filtered_sentence = []
    for w in word_tokens: 
        if w not in stop_words: 
            filtered_sentence.append(w)
    lemma_word = []
    wordnet_lemmatizer = WordNetLemmatizer()
    for w in filtered_sentence:
        word1 = wordnet_lemmatizer.lemmatize(w, pos = "n")
        word2 = wordnet_lemmatizer.lemmatize(word1, pos = "v")
        word3 = wordnet_lemmatizer.lemmatize(word2, pos = ("a"))
        lemma_word.append(word3)
    return ' '.join(lemma_word)

def process(df, t):
    df[t] = df[t].apply(lambda x : x.lower())
    df[t] = df[t].apply(lambda x : x.strip())
    df[t] = df[t].apply(lambda x : re.sub('\n', ' ', x))
    df[t] = df[t].apply(lambda x : remove_ponctuation(x))
    df[t] = df[t].apply(lambda x : re.sub('\[[^]]*\]', '', x))
    df[t] = df[t].apply(lambda x : re.sub(r'[^\w\s]', ' ', x))
    df[t] = df[t].apply(lambda x : re.sub(r"\s+[a-zA-Z]\s+", " ", x))
    df[t] = df[t].apply(lambda x : re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", x))
    df[t] = df[t].apply(lambda x : remove_non_ascii(x))
    df[t] = df[t].apply(lambda x : remove_stop_words(x))
    df[t] = df[t].apply(lambda x : Lemmatize(x))
    print(df.head())
    return df



data_file = 'data/arxiv-metadata-oai-snapshot.json'
def get_metadata():
    with open(data_file, 'r') as f:
        for line in f:
            yield line

data_dir = 'data'
train, test = pd.read_csv(os.path.join(data_dir, 'train.csv')), pd.read_csv(os.path.join(data_dir, 'test.csv'))

cat_tr = train['Category']
cat_tr = np.array(cat_tr)
cat_tr = np.unique(cat_tr)

metadata = get_metadata()
for paper in metadata:
    paper_dict = json.loads(paper)
    print('Categories: {}\n\nAbstract: {}\nRef: {}'.format(paper_dict.get('categories'), paper_dict.get('abstract'), paper_dict.get('journal-ref')))
    break

titles = []
abstracts = []
metadata = get_metadata()
for paper in metadata:
    paper_dict = json.loads(paper)
    ref = paper_dict.get('categories')
    try:
        cat = str(ref)
        for i in cat_tr:
            if ref == cat_tr[i]:
                titles.append(paper_dict.get("categories"))
                abstracts.append(paper_dict.get('abstract'))
    except:
        pass 

print(len(titles), len(abstracts))

papers = pd.DataFrame({
    'title': titles,
    'abstract': abstracts
})
papers.head()

del titles, abstracts

papers = papers[['title','abstract']]
papers.columns = ['target_text', 'input_text']
papers = papers.dropna()

train_df = process(papers, 'input_text')
test = process(test, 'Abstract')

X = train_df['input_text']
y = train_df['target_text']
test_column = test['Abstract']

del papers, train_df, test, train

print(X)


docs = np.array(X)

y = np.array(y)

test_column = np.array(test_column)

l = docs

lt = test_column


dic_unique_indx = pd.Series(y).factorize()
y_uniq_indx = dic_unique_indx[0]





################   pipelines   ################


################   BernoulliNB

text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', BernoulliNB()),
])

text_clf.fit(l, y_uniq_indx)
predd_bernoulli = text_clf.predict(l)

## train accuracy

A = accuracy_score(y_uniq_indx,predd_bernoulli)

with open('info.csv', 'a') as f:
    f.write('BernoulliNB accuracy:'+str(A)+'\n')

## test

print("test predict start")
result_test_all = text_clf.predict(lt)
factorized_names = dic_unique_indx[1]
y_test = []
for i in result_test_all:
    y_test.append(factorized_names[i])

with open('bernoulli.csv', 'a') as f:
    f.write('Id,Category\n')
    for i,line in enumerate(y_test):
        f.write(str(i)+','+line+'\n')


################   Multinomial

text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])

text_clf.fit(l, y_uniq_indx)
predd_multi = text_clf.predict(l)

## train accuracy

A = accuracy_score(y_uniq_indx,predd_multi)

with open('info.csv', 'a') as f:
    f.write('MultinomialNB accuracy:'+str(A)+'\n')

## test

print("test predict start")
result_test_all = text_clf.predict(lt)
factorized_names = dic_unique_indx[1]
y_test = []
for i in result_test_all:
    y_test.append(factorized_names[i])

with open('multinomial.csv', 'a') as f:
    f.write('Id,Category\n')
    for i,line in enumerate(y_test):
        f.write(str(i)+','+line+'\n')


################   SVM

text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss='hinge', penalty='l2',
                          alpha=1e-3, random_state=42,
                          max_iter=5, tol=None)),
])

text_clf.fit(l, y_uniq_indx)
predd_svm = text_clf.predict(l)

## train accuracy

A = accuracy_score(y_uniq_indx,predd_svm)

with open('info.csv', 'a') as f:
    f.write('SVM accuracy:'+str(A)+'\n')

## test

print("test predict start")
result_test_all = text_clf.predict(lt)
factorized_names = dic_unique_indx[1]
y_test = []
for i in result_test_all:
    y_test.append(factorized_names[i])

with open('svm.csv', 'a') as f:
    f.write('Id,Category\n')
    for i,line in enumerate(y_test):
        f.write(str(i)+','+line+'\n')


################   GuassianNB

text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('change',FunctionTransformer(lambda x: x.todense(), accept_sparse=True)),
    ('clf', GaussianNB()),
])

text_clf.fit(l, y_uniq_indx)
predd_g = text_clf.predict(l)

##train accuracy

A = accuracy_score(y_uniq_indx,predd_g)

with open('info.csv', 'a') as f:
    f.write('GuassianNB accuracy:'+str(A)+'\n')


##test

print("test predict start")
result_test_all = text_clf.predict(lt)
factorized_names = dic_unique_indx[1]
y_test = []
for i in result_test_all:
    y_test.append(factorized_names[i])

with open('guassian.csv', 'a') as f:
    f.write('Id,Category\n')
    for i,line in enumerate(y_test):
        f.write(str(i)+','+line+'\n')

