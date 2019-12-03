import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv("data\height.vs.temperature.csv")
#print(data.head(),data.columns)

X = data["height"].values.reshape(-1,1)
y = data["temperature"].values.reshape(-1,1)
reg = LinearRegression()
reg.fit(X,y)
prediction = reg.predict(X)

plt.figure(figsize=(18,18))
plt.plot(data["height"],prediction,c = "blue",linewidth = 2)
#plt.show()
pre = reg.predict([[100],[50]])

print("{:.5},{:.5}".format(reg.intercept_[0],reg.coef_[0][0]))
print("{:.5},{:.5}".format(pre[0][0],pre[1][0]))
