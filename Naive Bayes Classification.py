#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv(r"C:\Users\Administrator\Downloads\Diabetese prediction.csv")
df


# In[ ]:





# In[5]:


df.isna().sum()


# In[6]:


sns.countplot(df["diabetes"])


# In[7]:


x=df.iloc[:, :2].values
y =df.iloc[:, -1].values


# In[8]:


x


# In[9]:


y


# In[10]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.2, random_state=0)


# In[13]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[14]:


x_train


# In[15]:


x_test


# In[19]:


from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train,y_train)


# In[20]:


y_pred = classifier.predict(x_test)


# In[21]:


y_pred


# In[34]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
cm


# In[36]:


from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,y_pred)
ac


# In[37]:


sns.heatmap(cm, annot=True)


# In[ ]:


+
  

