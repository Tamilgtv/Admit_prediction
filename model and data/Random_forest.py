#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import os


# In[7]:


df = pd.read_csv("pga15/admit.csv")


# In[3]:


pwd


# In[8]:


df


# In[9]:


df.isna().sum()


# ### data has no missing values

# In[10]:


df.dtypes


# In[14]:


numeric_data = df.loc[:,['GRE','TOEFL','CGPA']]


# In[15]:


numeric_data


# In[26]:


plt.boxplot(df['GRE'])
plt.show()


# In[27]:


plt.boxplot(df['TOEFL'])
plt.show()


# In[28]:


plt.boxplot(df['CGPA'])
plt.show()


# In[31]:


sns.countplot(data=df, x='Admit')
plt.show()


# In[5]:


admit = df.drop(['SOP','LOR ','Research'],axis =1)


# In[6]:


admit


# In[7]:


x = admit.iloc[:,:4]


# In[9]:


x


# In[8]:


y = admit.iloc[:,-1]


# In[11]:


y


# In[12]:


from sklearn.model_selection import train_test_split


# In[13]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,train_size=0.7,random_state=0)


# In[14]:


x_train


# In[10]:


from sklearn.ensemble import RandomForestClassifier


# In[46]:


cl = RandomForestClassifier(n_estimators= 1000, criterion= 'entropy',random_state=3,oob_score = True, max_features ='auto')


# In[47]:


cl.fit(x_train,y_train)


# In[26]:


cl = RandomForestClassifier(n_estimators= 500, criterion= 'entropy',random_state=0,max_depth=10,oob_score = True, max_features ='auto')


# In[11]:


model = RandomForestClassifier(n_estimators= 1000, criterion= 'entropy',random_state=3,oob_score = True, max_features ='auto')


# In[12]:


model.fit(x,y)


# In[13]:


predict = model.predict(x)


# In[14]:


predict


# In[17]:


pickle.dump(model,open('modelpk.pkl','wb'))


# In[18]:


modelpk = pickle.load(open('modelpk.pkl','rb'))


# In[48]:


y_pred = cl.predict(x_test)


# In[49]:


y_pred


# In[50]:


from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(y_test,y_pred)


# In[51]:


cm


# In[ ]:





# In[52]:


accuracy=(cm[0,0]+cm[1,1])/cm.sum()*100
print('Accuracy of the model:',accuracy,'%')


# In[54]:


estimator = cl.estimators_[3]


# In[55]:


estimator


# In[56]:


from sklearn.tree import export_graphviz
import collections
import graphviz


# In[57]:


x.columns


# In[58]:


export_graphviz(estimator, out_file='tree6.dot',
                feature_names = ['GRE', 'TOEFL', 'Univ_Rating', 'CGPA'],
                class_names = np.array(['not Admit','admit']),
                rounded = True, proportion = False,
                precision = 2, filled = True)

from subprocess import call
call(['dot', '-Tpng', 'tree6.dot', '-o', 'tree6.png', '-Gdpi=600'])

# Display in python
import matplotlib.pyplot as plt
plt.figure(figsize = (14, 18))
plt.imshow(plt.imread('tree6.png'))
plt.axis('off');
plt.show();


# In[ ]:




