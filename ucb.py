import numpy as np
import matplotlib as plt
import pandas as pd
import math

dataset  =pd.read_csv('Ads_CTR_Optimisation.csv')
d = 10
num  = [0] *d 
sum = [0]*d
N = 10000
for n in range(0,N):
    for i in range(0,d) :
        avg = sum[i]/num[i]
        de = math.sqrt (3/2 * math.log(n+1) / num[i] )
        upper = avg + de