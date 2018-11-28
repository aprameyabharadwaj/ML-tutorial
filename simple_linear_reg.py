# -*- coding: utf-8 -*-


# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values     
                
                


from sklearn.cross_validation import train_test_split 
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 1/3 , random_state = 0)


"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train , y_train)


y_pred = regressor.predict(X_test)

"""plt.scatter(X_train , y_train , color = 'red')
plt.plot(X_train , regressor.predict(X_train), color = 'blue' )
plt.title('Salary vs Experience (Traning Set) ' )
plt.xlabel('years of exp')
plt.ylabel('salary')
plt.show()"""



plt.scatter(X_test , y_test , color = 'red')
plt.plot(X_test , regressor.predict(X_test), color = 'blue' )
plt.title('Salary vs Experience (Test Set) ' )
plt.xlabel('years of exp')
plt.ylabel('salary') 
plt.show()