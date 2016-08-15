from math import sqrt
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import warnings
import random




def k_near(data,predict,k=3):
    if(len(data)>=k):
        warnings.warn("Stupid action")
    euc_distances=[]
    for i in data:
        for ii in data[i]:
            distance=sqrt((ii[0]-predict[0])**2 + (ii[1]-predict[1])**2)
            euc_distances.append([distance,i])
    votes=[]
    for i in sorted(euc_distances)[:k]:
        votes.append(i[1])

    vote_result=Counter(votes).most_common(1)[0][0]
    return vote_result

##dataset={'k':[[1,2],[2,3],[3,1]],'y':[[6,5],[7,7],[8,6]]}
##new_feature=[5,7]



def form_dataset(data):
    dataset={}
    dataset['b']=[]
    dataset['m']=[]
    for i in data:
        if(i[-1]==2):
            dataset['b'].append(i[:9])
        else:
            dataset['m'].append(i[:9])
    return dataset







df=pd.read_csv('cancer.csv')
df.replace('?',-99999,inplace=True)
df.drop(['id'],1,inplace=True)
data=df.astype(float).values.tolist()

##z=len(data)-1
predict=data[-1]
predict=predict[:-1]
data=data[:-1]

random.shuffle(data)
dataset=form_dataset(data)

v=k_near(dataset,predict)
