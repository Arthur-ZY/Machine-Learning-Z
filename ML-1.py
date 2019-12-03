import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv("data\Advertising.csv")
#print(data.head())
#print(data.columns)



X = data["TV"].values.reshape(-1,1)
Y = data["sales"].values.reshape(-1,1)

reg = LinearRegression()
reg.fit(X,Y)
print("a = {:.5}".format(reg.coef_[0][0]))
print("b = {:.5}".format(reg.intercept_[0]))
print("Y = {:.5}X + {:.5}".format(reg.coef_[0][0],reg.intercept_[0]))

predictions = reg.predict(X)
plt.figure(figsize=(18,18))
plt.scatter(data["TV"],data["sales"],c = "black")
plt.plot(data["TV"],predictions,c = "blue",linewidth = 2)
plt.xlabel("TV")
plt.ylabel("sales")
plt.show()

predictions = reg.predict([[100]])
print(predictions[0][0])