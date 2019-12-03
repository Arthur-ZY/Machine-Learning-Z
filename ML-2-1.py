from sklearn import datasets
from collections import Counter
from sklearn.model_selection import train_test_split
import numpy as np

iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=2002)

def euc_dis(instance1,instance2):
   dist = np.sqrt(sum((instance1 - instance2)**2))
   return dist

def knn_classify(X,y,testinstance,k):
    distances = [euc_dis(x,testinstance) for x in X]
    kneighbor = np.argsort(distances)[:k]   #kneighbor是下标 用优先级队列优化排序
    count = Counter(y[kneighbor])
    return count.most_common()[0][0]

prediction = [knn_classify(X_train,y_train,data,3) for data in X_test]
correct = np.count_nonzero((prediction==y_test)==True)
print(correct/len(X_test))


