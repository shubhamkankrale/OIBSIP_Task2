#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[3]:


import seaborn as sea


# In[4]:


import matplotlib.pyplot as mat


# In[5]:


sam=pd.read_csv('CarPrice_Assignment.csv')


# In[6]:


sam


# In[7]:


sam.head()


# In[8]:


sam.describe()


# In[9]:


sam.shape


# In[10]:


sam.columns


# In[11]:


sam.info()


# In[12]:


sam.CarName.unique()


# In[13]:


#Distribution Of categorial data
print(sam.fueltype.value_counts())
print(sam.carbody.value_counts())
print(sam.doornumber.value_counts())


# In[14]:


#encoding the fueltype column
sam.replace({'fueltype':{'Petrol':0,'Diesel':1,'gas':2}},inplace=True)
#encoding the carbody column
sam.replace({'carbody':{'connvertible':0,'hatchback':1,'sedan':2}},inplace=True)
#encoding the doornumber column
sam.replace({'doornumber':{'two':0,'four':1}},inplace=True)


# In[15]:


sam.head()


# In[16]:


sea.set_style("whitegrid")
mat.figure(figsize=(15,10))
sea.displot(sam.price)
mat.show()


# In[17]:


print(sam.corr())


# In[18]:


mat.figure(figsize=(20,15))
correlations= sam.corr()
sea.heatmap(correlations, cmap="coolwarm", annot=True)
mat.show()


# In[19]:


predict = "price"
data = sam[["symboling", "wheelbase", "carlength", 
             "carwidth", "carheight", "curbweight", 
             "enginesize", "boreratio", "stroke", 
             "compressionratio", "horsepower", "peakrpm", 
             "citympg", "highwaympg", "price"]]


# In[20]:


x = np.array(data.drop([predict], 1))
y = np.array(data[predict])


# In[21]:


from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)


# In[22]:


from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(xtrain, ytrain)
predictions = model.predict(xtest)


# In[23]:


from sklearn.metrics import mean_absolute_error
model.score(xtest, predictions)


# In[ ]:




