import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("data\car-data.csv")
print(df.columns)
df_colors = df["Color"].str.get_dummies().add_prefix("color:")  #one-hot
df_type = df["Type"].apply(str).str.get_dummies().add_prefix("Type:")
df = pd.concat([df,df_colors,df_type],axis=1)
df = df.drop(["Brand","Type","Color"],axis=1)

#matrix = df.corr()
#f,ax = plt.subplots(figsize = (8,6))
#sns.heatmap(matrix,square=True)
#f.show()

X = df[["Construction Year","Days Until MOT","Odometer"]]
y = df["Ask Price"].values.reshape(-1,1)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=2003)

print(y_test)
X_normalizer = StandardScaler()
X_train = X_normalizer.fit_transform(X_train)
X_test = X_normalizer.transform(X_test)

y_normalizer = StandardScaler()
y_train = y_normalizer.fit_transform(y_train)
y_test = y_normalizer.transform(y_test)
print(y_test)

knn = KNeighborsRegressor(n_neighbors=2)
knn.fit(X_train,y_train.ravel())  #副本 扁平化

y_pred = knn.predict(X_test)
y_pre_v = y_normalizer.inverse_transform(y_pred)
y_test_v = y_normalizer.inverse_transform(y_test)

plt.scatter(y_pre_v,y_test_v)
plt.xlabel("Prediction")
plt.ylabel("Real value")

diagonal = np.linspace(500,1500,100)
plt.plot(diagonal,diagonal,"-r")
plt.xlabel("Predicted ask price")
plt.ylabel("Ask price")
#plt.show()

print(mean_absolute_error(y_pre_v,y_test_v))


