######## BernoullliNB implementation from scratch (without libraries)


import os
import numpy as np
import pandas as pd
import re, unicodedata


data_dir = '.'
train, test = pd.read_csv(os.path.join(data_dir, 'train.csv')), pd.read_csv(os.path.join(data_dir, 'test.csv'))


stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", 
 "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", 
 "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", 
 "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", 
 "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", 
 "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", 
 "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when",
 "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", 
 "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]


def remove_non_ascii(words):
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return ''.join(new_words)

def remove_stop_words(words):
    querywords = words.split()
    resultwords  = [i for i in querywords if i not in stop_words]
    return ' '.join(resultwords)

def process(df, t):
    df[t] = df[t].apply(lambda x : x.lower())
    df[t] = df[t].apply(lambda x : x.strip())
    df[t] = df[t].apply(lambda x : re.sub('\n', ' ', x))
    df[t] = df[t].apply(lambda x : re.sub('\[[^]]*\]', '', x))
    df[t] = df[t].apply(lambda x : re.sub(r'[^\w\s]', ' ', x))
    df[t] = df[t].apply(lambda x : re.sub(r"\s+[a-zA-Z]\s+", " ", x))
    df[t] = df[t].apply(lambda x : remove_non_ascii(x))
    df[t] = df[t].apply(lambda x : remove_stop_words(x))
    return df


train = process(train, 'Abstract')
test = process(test, 'Abstract')


class BernoulliVectorizer:
    def __init__(self):
        self.vocab = []
        self.vocab_counter = {}
        
    def build_vocab(self, data):
        for document in data:
            for word in document.split(' '):
                if word in self.vocab:
                    self.vocab_counter[word] += 1
                else:
                    self.vocab.append(word)
                    self.vocab_counter[word] = 1
                
    def transform(self, data):
        i = 0
        new_data=[]
        for document in data:
            tokens = document.split(' ') 
            bin_vect = np.zeros(len(self.vocab))
            for word_idx in range(len(self.vocab)):
                for e in tokens: 
                    if e == self.vocab[word_idx]:
                        bin_vect[word_idx ] = 1
            new_data.append( bin_vect)   
            i += 1          
        return new_data
    
    def fit_transform(self, data):
        self.build_vocab(data)
        return self.transform(data)



class BernoulliNB:

    def __init__(self, alpha):
        self.alpha = alpha
        pass

    def fit(self, X, y):

        num_y = np.unique(y)
        self.n_classes = len(num_y)
        n_classes = self.n_classes
        
        y_uniq_indx=[]
        for c in y:
            for i in range(n_classes):
                if c == num_y[i]:
                    y_uniq_indx.append(i)

        self.counts = np.zeros(n_classes)
        for i in y_uniq_indx:
            self.counts[i] += 1
        self.counts = self.counts / len(y)

        self.params = np.zeros((n_classes, len(X[1])))
        for idx in range(len(X)):
            self.params[y_uniq_indx[idx]] += X[idx]
        self.params += self.alpha 
        self.class_sums = np.zeros(self.n_classes)
        for i in y_uniq_indx:
            self.class_sums[i] += 1
        self.class_sums += self.n_classes*self.alpha
        self.params = self.params / self.class_sums[:, np.newaxis]
        
        
    def predict(self, X):
        
        neg_prob = np.log(1.0 - (self.params.astype('float')))
        jll = np.dot(X, (np.log(self.params.astype('float')) - neg_prob).T)
        jll += np.log(self.counts) + neg_prob.sum(axis=1)
        return np.argmax(jll, axis=1)
    


B = BernoulliVectorizer()
train['Abstract'] = B.fit_transform(train['Abstract'])
test['Abstract'] = B.transform(test['Abstract'])

X = train['Abstract']
y = train['Category']

num_y = np.unique(y) #unique labels names
n_classes = len(num_y) #count of unique labels

d=[]
for i in X:
    d.append(i)
d=np.array(d)

# assign numbers to labels
y_uniq_indx=[]
for c in y:
    for i in range(n_classes):
        if c == num_y[i]:
            y_uniq_indx.append(i)

clf = BernoulliNB(1.0)
clf.fit(d, y)

train_pred = clf.predict(d) #prediction on train set



#accuracy for training set
count=0
for i in range(len(train_pred)):
    if train_pred[i] == y_uniq_indx[i]:
        count +=1
print('Accuracy for train set is: ', count / len(train_pred))


test_column = test['Abstract']
tt=[]
for i in test_column:
    tt.append(i)
tt=np.array(tt)


result_test_all = clf.predict(tt) #prediction on test set

# Changing to real names of labels
y_test=[]
for c in result_test_all:
    for i in range(n_classes):
        if c == i:
            y_test.append(num_y[i])


with open('to_submit.csv', 'w') as f:
    f.write('Id,Category\n')
    for i,line in enumerate(y_test):
        f.write(str(i)+','+line+'\n')

