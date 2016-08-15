import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation,neighbors,svm
df=pd.read_csv('cancer.csv')
df.replace('?',-99999,inplace=True)
X=np.array(df.drop(['class'],1))
y=np.array(df['class'])

X=preprocessing.scale(X)
X_train,X_test,y_train,y_test=cross_validation.train_test_split(X,y,test_size=0.2)
clf=svm.SVC()
clf.fit(X_train,y_train)
acc=clf.score(X_test,y_test)

print acc
