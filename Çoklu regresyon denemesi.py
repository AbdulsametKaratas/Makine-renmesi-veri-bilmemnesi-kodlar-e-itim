# Kütüphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Veri Ön İşleme

# Veri Yükleme
datas = pd.read_csv('emlak.csv')

# Encoder: Kategorik -> Numeric
from sklearn.preprocessing import LabelEncoder
datas2 = datas.apply(LabelEncoder().fit_transform)

c = datas2.iloc[:, :1]
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features='all')
c = ohe.fit_transform(c).toarray()
print(c)

emlakvergisi = pd.DataFrame(data=c, index=range(14), columns=['o', 'r', 's'])
lastdatas = pd.concat([emlakvergisi, datas.iloc[:, 1:3]], axis=1)
lastdatas = pd.concat([datas2.iloc[:, -2:], lastdatas], axis=1)

# Verilerin Eğitim ve Test için Bölünmesi
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    lastdatas.iloc[:, :-1], lastdatas.iloc[:, -1:], test_size=0.33, random_state=0
)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

print(y_pred)

import statsmodels.formula.api as sm
X = np.append(arr=np.ones((14, 1)).astype(int), values=lastdatas.iloc[:, :-1], axis=1)
X_l = lastdatas.iloc[:, [0, 1, 2, 3, 4, 5]].values
r_ols = sm.OLS(endog=lastdatas.iloc[:, -1:], exog=X_l)
r = r_ols.fit()
print(r.summary())

lasdatas = lastdatas.iloc[:, 1:]  
import statsmodels.formula.api as sm
X = np.append(arr=np.ones((14, 1)).astype(int), values=lasdatas.iloc[:, :-1], axis=1)
X_l = lasdatas.iloc[:, [0, 1, 2, 3, 4]].values
r_ols = sm.OLS(endog=lasdatas.iloc[:, -1:], exog=X_l)
r = r_ols.fit()
print(r.summary())

x_train = x_train.iloc[:, 1:]
x_test = x_test.iloc[:, 1:]

regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)
