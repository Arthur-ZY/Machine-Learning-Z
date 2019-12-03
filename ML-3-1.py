import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns

data = pd.read_csv("data\Banking.csv",header = 0)
data = data.dropna()

data['education'] = np.where(data['education']=='basic.9y','Basic',data['education'])
data['education'] = np.where(data['education']=='basic.4y','Basic',data['education'])
data['education'] = np.where(data['education']=='basic.6y','Basic',data['education'])
#print(data['education'].unique())

#print(data['y'].value_counts())

count_no_op = len(data[data['y']==0])
count_op = len(data[data['y']==1])

#print(data.groupby('y').mean())   #分组


'''
count_op = len(data['y']==1)
pct_of_no_sub = count_no_op/(count_no_op+count_op)
print(pct_of_no_sub)

'''
#smote 过采样 knn 解决数据不平衡问题
cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
for var in cat_vars:
    cat_list = pd.get_dummies(data[var],prefix=var)
    data = data.join(cat_list)
data_final = data.drop(cat_vars,axis=1)

X = data_final.loc[:,data_final.columns!='y']
y = data_final.loc[:,data_final.columns=='y'].values.ravel()
from imblearn.over_sampling import SMOTE
os = SMOTE(random_state=42)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
#print(X_test) DataFrame
columns = X_train.columns
os_data_X,os_data_y = os.fit_sample(X_train,y_train)
#print(X) DataFrame
#print(os_data_X) Numpy
os_data_X = pd.DataFrame(data=os_data_X,columns=columns)
#os_data_y = pd.DataFrame(data=os_data_y,columns=['y'])

print(os_data_y)
log = LogisticRegression()
log.fit(os_data_X,os_data_y)
y_pred = log.predict(X_test)
print(log.score(X_test,y_test))
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))