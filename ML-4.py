import pandas as pd
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
#import nltk
#nltk.download('stopwords')
from sklearn.feature_extraction.text import CountVectorizer
df = pd.read_csv('data\spam.csv',encoding='latin')
df.rename(columns = {'v1':'Label','v2':'Text'},inplace=True)
df['numLabel'] = df['Label'].map({'ham':0,'spam':1})
text_length = [len(df.loc[i,'Text']) for i in range(len(df))]
'''
plt.hist(text_length,100,facecolor='blue')
plt.xlim([0,200])
plt.show()
'''
stopset = set(stopwords.words('english'))
#vectorizer = CountVectorizer(stop_words=stopwords,binary=True)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df.Text)
y = df.numLabel
