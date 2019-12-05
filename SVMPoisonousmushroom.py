import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import classification_report,accuracy_score


data = pd.read_csv("data\mushrooms.csv")
mushroom_encoded = pd.get_dummies(data)
X_trian = mushroom_encoded.iloc[:,2:]
y_train = mushroom_encoded.iloc[:,1]

pca = PCA(n_components=117,whiten=True,random_state=42)
scv = SVC(kernel='linear',class_weight='balanced')
mode = make_pipeline(pca,scv)
Xtrain,Xtest,ytrain,ytest = train_test_split(X_trian,y_train,random_state=42)


param_grid = {'svc__C':[1,10,100]}
grid = GridSearchCV(mode,param_grid)
grid.fit(Xtrain,ytrain)
mode = grid.best_estimator_
yfit = mode.predict(Xtest)




