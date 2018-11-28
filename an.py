# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 00:29:03 2017

@author: Aprameya
"""

import theano 
import keras 
import tensorflow


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values


from sklearn.preprocessing import LabelEncoder , OneHotEncoder
labelencoder_X1 = LabelEncoder()
X[: , 1] =labelencoder_X1.fit_transform(X[: , 1])
labelencoder_X2 = LabelEncoder()
X[: , 2] =labelencoder_X2.fit_transform(X[: , 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[: ,1 :]

                
from sklearn.cross_validation import train_test_split 
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.20 , random_state = 0)


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(output_dim=6, init = 'uniform', activation = 'relu', input_dim=11))
classifier.add(Dense(output_dim=6, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim=1, init = 'uniform', activation = 'sigmoid', ))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train , y_train, batch_size = 10 , nb_epoch =100 )

y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test , y_pred)

