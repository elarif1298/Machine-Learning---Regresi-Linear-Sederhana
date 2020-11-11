#memanggil library yang dubutuhkan
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

#memanggil dataset
data = pd.read_csv("dataku-ukuransepatu.csv", delimiter=";")
print(data.shape)
data

X = data['ukuransepatu'].values
Y = data['beratbadan'].values

plt.scatter(X, Y, label='Data')

plt.xlabel('beratbadan')
plt.ylabel('ukuransepatu')

plt.legend()
plt.show()

mean_x = np.mean(X.copy())
mean_y = np.mean(Y.copy())

n = len(X)

num = 0
den = 0

for i in range(n):
    num += (X[i] - mean_x) * (Y[i] - mean_y)
    den += (X[i] - mean_x) ** 2
    
    b1 = float(num) / float(den)
    b0 = mean_y - (b1 * mean_x)
    
    print('Mean ukuran sepatu = '+ str(mean_x))
    print('Mean berat badan = '+ str(mean_y))
    print(b0, b1)
    
  max_x = np.max(X) + 10
min_x = np.min(X) - 10

x = np.linspace(min_x, max_x, 80)
y = b0 + b1 * x

plt.plot(x, y, label='Linier Regression')
plt.scatter(X, Y, label='Scatter Plot')

plt.xlabel('ukuransepatu')
plt.ylabel('beratbadan')

plt.legend()
plt.show()

y = b0 + b1 * 42
print(y)

rmse = 0
for i in range(n):
    y_pred = b0 + b1 * X[i]
    rmse += (Y[i] - y_pred) ** 2
    rmse = np.sqrt(rmse/n)
    print(rmse)

ss_t = 0
ss_r = 0

for i in range(n):
    y_pred = b0 + b1 * X[i]
    ss_t += (Y[i] - mean_y) ** 2
    ss_r += (Y[i] - y_pred) ** 2
    
r2 = 1 - (ss_r/ss_t)
print(r2)

df = pd.DataFrame([[39, 50],[40, 53],[38, 40],[41, 60],[42, 70],[42, 78],[43, 69],[43, 55],[43, 70],[44, 72]])
df.columns = ['x','y']
df.head()

x_train = df['x'].values[:,np.newaxis]
y_train = df['y'].values

lm = LinearRegression()
lm.fit(x_train,y_train) #fase training
lm.coef_  

lm.intercept_

x_test = [[35],[36],[37],[38],[39],[40],[41],[42],[43],[44],[45]]
p = lm.predict(x_test)
print(p)

#prepare plot
pb = lm.predict(x_train)
dfc = pd.DataFrame({'x': df['x'],'y':pb})
plt.scatter(df['x'],df['y'])
plt.plot(dfc['x'],dfc['y'], color='red',)
plt.xlabel('ukuran sepatu')
plt.ylabel('beratbadan')
plt.show()
