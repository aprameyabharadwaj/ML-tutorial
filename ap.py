import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)


trans= [] 
for i in range(0,7501) :
    trans.append([str(dataset.values[i,j]) for j in range(0,20)])
    
    
from apyori import apriori
rules = apriori(trans, min_support = 0.003 ,min_confidence = 0.2 , min_lift =3, min_length = 2)

results = list(rules)