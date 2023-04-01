import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle

df=pd.read_csv('Iris.csv')
df=df.drop(columns=['Id'],axis=0)
x=np.array(df.iloc[:,0:4])
y=np.array(df.iloc[:,-1])

le=LabelEncoder()
y=le.fit_transform(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
sv=SVC(kernel='linear').fit(x_train,y_train)

pickle.dump(sv,open('Iris.pkl','wb'))
