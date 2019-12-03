import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime as dt
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing

df = pd.read_csv("data\stock.csv")
'''
print(df.head())
print(df.columns)
print(np.shape(df))
'''
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')  #变索引
df.sort_values(by=['date'],inplace=True,ascending=True)

df.dropna(axis=0,inplace=True) #删除缺失值的行
num = 5
df['label'] = df['close'].shift(-num)
Data = df.drop(['label','price_change','p_change'],axis=1)

X = Data.values
X = preprocessing.scale(X)  #标准化
X = X[:-num]
df.dropna(inplace=True)
Target = df.label
y = Target.values

print(np.shape(X),np.shape(y))

X_train,y_train = X[:550,:],y[:550]
X_test,y_test = X[550:,:],y[550:]

lr = LinearRegression()
lr.fit(X_train,y_train)
print(lr.score(X_test,y_test))
