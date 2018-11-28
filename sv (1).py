# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 12:50:12 2017

@author: Aprameya
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values
                
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
sc_y = StandardScaler()
y = sc_y.fit_transform(y)
                
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X,y)
                
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))


plt.scatter(X ,y , color = 'red')
plt.plot(X , regressor.predict(X), color = 'blue')
plt.title('Truth or bluff lin reg')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()