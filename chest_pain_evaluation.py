#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
df=pd.read_csv('Desktop/Heart.csv')
df


# In[17]:


df.shape #Used to give the total number of rows and columns


# In[5]:


df.values #Used to display the values


# In[11]:


df.describe


# In[14]:


df.isnull()


# In[18]:


df.dtypes


# In[20]:


df.Ca[df.Ca==0].count()


# In[22]:


df.Fbs[df.Fbs==0].count()


# In[23]:


df.isin([0]).any().any()


# In[24]:


df['Age'].mean()


# In[15]:


cols=[1,2,3,4]
pf=df[df.columns[cols]]


# In[48]:


pf


# In[4]:


X=df[['Age','Sex','RestBP','Chol']]
X




# In[5]:


Y=df[['ChestPain']]
Y


# In[10]:


from sklearn.model_selection import train_test_split #train_test_split --> Function of sklearn.model_selection
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25)
#X_train
#X_test
#Y_train
#Y_test


# In[11]:


X_train


# In[12]:


len(X_train) 


# In[1]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.datasets import load_digits

digits= load_digits()


# In[2]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test= train_test_split(digits.data,digits.target,test_size=0.3)


# In[4]:


lr=LogisticRegression()
lr.fit(X_train,Y_train)
lr.score(X_test,Y_test)


# In[13]:


svm=SVC()
svm.fit(X_train,Y_train)
svm.score(X_test,Y_test)


# In[12]:


fc=RandomForestClassifier(n_estimators=40)
fc.fit(X_train,Y_train)
fc.score(X_train,Y_train)


# In[19]:


from sklearn.model_selection import KFold
kf=KFold(n_splits=3)
kf


# In[ ]:




