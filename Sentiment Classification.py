#使用NB做情感分类
import matplotlib.pyplot as plt
import re
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
#文件读取
def read_data(path,is_pos=None):
    reviews,labels = [],[]
    with open(path,'r',encoding='utf-8') as file:
        review_start = False
        review_text = []
        for line in file:
            line = line.strip()
            if not line:continue
            if not review_start and line.startswith('<review'):
                review_start = True
                if 'label' in line:
                    labels.append(int(line.split('"')[-2]))
                continue
            if review_start and line == "</review>":
                review_start = False
                reviews.append('.'.join(review_text))
                review_text = []
                continue
            if review_start:
                review_text.append(line)
    if is_pos:
        labels = [1]*len(reviews)
    elif not is_pos is None:
        labels = [0]*len(reviews)
    return reviews,labels
#读取数据 对其预处理
def process_file():
    train_pos_file = 'data/train.positive.txt'
    train_neg_file = 'data/train.negative.txt'
    test_comb_file = 'data/test.combined.txt'
    train_pos_cmts, train_pos_lbs = read_data(train_pos_file, True)
    train_neg_cmts, train_neg_lbs = read_data(train_neg_file, False)
    train_comments = train_pos_cmts + train_neg_cmts
    train_labels = train_pos_lbs + train_neg_lbs
    test_comments, test_labels = read_data(test_comb_file)
    return train_comments, train_labels, test_comments,test_labels

train_comments, train_labels, test_comments, test_labels = process_file()
#print (train_comments[1], train_labels[1])

def stop_word(path):
    stopword = set()
    with open(path,'r',encoding='utf-8') as file:
        for line in file:
            stopword.add(line.strip())
    return stopword

def clean_non_chinese_symbols(text):
    '''
    处理特殊字符
    '''
    text = re.sub('[!！]+', "!", text)
    text = re.sub('[?？]+', "?", text)
    text = re.sub("[a-zA-Z#$%&\'()*+,-./:;：<=>@，。★、…【】《》“”‘’[\\]^_`{|}~]+", " UNK ", text)
    return re.sub("\s+", " ", text)

def clean_number(text):
    '''
    处理数字
    '''
    return re.sub('/d+','NUM',text)
#处理单个句子
def process_text(text,stopword):
    text = clean_non_chinese_symbols(text)
    text = clean_number(text)
    text = ' '.join([item for item in jieba.cut(text) if item and not item in stopword])
    return text

path_stopword = 'data/stopwords.txt'
stopwords = stop_word(path_stopword)

train_comments_new = [process_text(comment,stopwords) for comment in train_comments]
test_comments_new = [process_text(comment,stopwords) for comment in test_comments]
print(train_comments_new[0],test_comments_new[0])

tfidf = TfidfVectorizer()
X_train = tfidf.fit_transform(train_comments_new)
y_train = train_labels
X_test = tfidf.transform(test_comments_new)
y_test = test_labels
#print (np.shape(X_train), np.shape(X_test), np.shape(y_train), np.shape(y_test))
#朴素贝叶斯
NB = MultinomialNB()
NB.fit(X_train,y_train)
y_pre = NB.predict(X_test)
print('Accuracy:',accuracy_score(y_test,y_pre))
#KNN分类
knn_c = KNeighborsClassifier()
knn_c.fit(X_train,y_train)
y_pre = knn_c.predict(X_test)
print('Accuracy',accuracy_score(y_test,y_pre))
#KNN回归 这玩意儿别用。。。。
'''
X_sta = StandardScaler()
X_train_sta = X_sta.fit_transform(X_train.toarray())
X_test_sta = X_sta.transform(X_test.toarray())
knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(X_train_sta,y_train)
y_pre = knn.predict(X_test_sta)
print("Accuary:",accuracy_score(y_test,y_pre))
'''
#逻辑回归
clf = LogisticRegression(solver="liblinear")
clf.fit(X_train,y_train)
y_pre = clf.predict(X_test)
print('Accuray:',accuracy_score(y_test,y_pre))


