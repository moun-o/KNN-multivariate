#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from scipy.spatial import distance
pd.options.mode.chained_assignment=None

#PART 0 :-----------> parameters

#choose k, here for example i chosed the first 5 nearest neighbours
k=5

#the column to be dropped from the dataset

drop_cols=['room_type','city','zipcode','state','longitude','latitude','host_response_rate','host_acceptance_rate','host_response_rate','host_listings_count']





# In[3]:


#PART 1 :-----------> data process

#read data
airbnb_list=pd.read_csv('paris_airbnb.csv')
print(airbnb_list.info())

#delete the non float and non interesting colomuns
airbnb_list=airbnb_list.drop(drop_cols,axis=1)


# In[4]:


print(airbnb_list.info())
#check how many NULL values 
print(airbnb_list.isnull().sum())


# In[5]:


#clean the column price, remove , and $
airbnb_list['price']=airbnb_list['price'].astype(str).str.replace('$','')
airbnb_list['price']=airbnb_list['price'].astype(str).str.replace(',','')

#convert price column to float
airbnb_list['price']=airbnb_list['price'].astype('float')


# In[6]:


#delete rows of bedrooms bathrooms and beds where are only little null values and 2 cols cleaning_fee and deposit_fee
airbnb_list=airbnb_list.drop(['cleaning_fee','security_deposit'],axis=1)
airbnb_list=airbnb_list.dropna(axis=0)
#check how many NULL values 
print(airbnb_list.isnull().sum())


# In[7]:


#random the dataset
airbnb_list=airbnb_list.loc[np.random.permutation(len(airbnb_list))]

#noramalize data based on normal standard distribution
normalized_data=(airbnb_list-airbnb_list.mean())/(airbnb_list.std())
normalized_data=normalized_data.dropna(axis=0)
normalized_data['price']=airbnb_list['price']

#separate into train data and test data 75% 25%
sup=int(0.75*len(normalized_data))
train_data=normalized_data.iloc[0:sup]
test_data=normalized_data.iloc[sup:]
train_data.info()


# In[8]:


train_data.info()


# In[9]:


from sklearn.neighbors import KNeighborsRegressor
#init your knn model
knn=KNeighborsRegressor(n_neighbors=k,algorithm='brute')

#train features init
features = train_data.columns.tolist()
features.remove('price')
train_features=train_data[features]

#train target init
train_target = train_data ['price']

#luanch knn learning
knn.fit(train_features,train_target)


# In[10]:


predictions= knn.predict(test_data[features])


# In[11]:


print(predictions)


# In[12]:


#PART 5 :-----------> Evaluate
from sklearn.metrics import mean_squared_error

errorMSE=mean_squared_error(test_data['price'],predictions)

print('error mse= ', errorMSE, 'rmse= ',errorMSE**0.5)

