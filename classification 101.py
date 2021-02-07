#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv('BankNote_Authentication.csv')
df.head()


# In[3]:


df.isna()


# In[4]:


sns.pairplot(df,hue='class')


# In[8]:


X = df.drop('class', axis = 1)
y = df['class']


# In[40]:


import sklearn
from sklearn.metrics import accuracy_score


# In[7]:


from sklearn.model_selection import train_test_split


# In[9]:


train_test_split(X, y, test_size=0.33, random_state=42)


# In[11]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=101)


# In[12]:


from sklearn.model_selection import GridSearchCV


# In[13]:


from sklearn.ensemble import RandomForestClassifier


# In[14]:


n_estimators = [64,100,128,200]
max_features = [2,3,4]
bootstrap = [True,False]
oob_score = [True,False]


# In[17]:


param_grid = {'n_estimators':n_estimators,
              'max_features':max_features,
              'bootstrap':bootstrap,
              'oob_score':oob_score}


# In[18]:


rfc = RandomForestClassifier()


# In[19]:


grid = GridSearchCV(rfc,param_grid)


# In[20]:


grid.fit(X_train,y_train)


# In[21]:


grid.best_params_


# In[24]:


rfc = RandomForestClassifier(max_features : 2,n_estimators : 200,oob_score : True)


# In[25]:


rfc = RandomForestClassifier(max_features=2,n_estimators=200,oob_score=True)


# In[26]:


rfc.fit(X_train,y_train)


# In[27]:


rfc.oob_score_


# In[28]:


from sklearn.metrics import plot_confusion_matrix,classification_report


# In[29]:


predictions = rfc.predict(X_test)


# In[30]:


print(classification_report(predictions,y_test))


# In[33]:


plot_confusion_matrix(rfc,X_test,y_test)


# In[43]:


errors = []
misclassifications = []
accuracy_Scpre = []

for n in range(1,200):
    
    rfc = RandomForestClassifier(n_estimators=n,max_features=2)
    rfc.fit(X_train,y_train)
    predictions = rfc.predict(X_test)
    n_missed = np.sum(predictions !=y_test)
    err = 1 - accuracy_score(y_test,predictions)
    errors.append(err)
    misclassifications.append(n_missed)
    


# In[44]:


plt.plot(range(1,200), errors)


# In[45]:


plt.plot(range(1,200), misclassifications)


# In[ ]:




