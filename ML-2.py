from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np

iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=2002)#默认0.25测试

clf = KNeighborsClassifier(n_neighbors=3)

clf.fit(X_train,y_train)

correct = np.count_nonzero((clf.predict(X_test)==y_test)==True) #等于true的个数
print(accuracy_score(y_test,clf.predict(X_test)))
print(correct / len(X_test))

