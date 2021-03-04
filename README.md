## KNN K nearest neighbors (multivariate case)

To make it short, KNN algorithm is a supervised classfication algorithm who estimates the output based on the K most similar entries which are called nearest neighbours.

In the multivariate case, we consider many columns and not only one

## How we compute similarity

similarity usually is computed based on distance between two entries, the distance metrics often are Euclidiean or City-Block.

## implementation
### Part 0 : libraries and parameters
We start by importing pandas and numpy, these 2 libraries are 
```
import pandas as pd
import numpy as np

```
then you fix k, which represent the size of neighbors to elect
```
#choose k, here for example i chosed the first 5 nearest neighbours
k=5


#the column to be dropped from the dataset which are not float or doesn't affect semantically the price

drop_cols=['room_type','city','zipcode','state','longitude','latitude','host_response_rate','host_acceptance_rate','host_response_rate','host_listings_count']

```
### PART 1 : Load data, process it and prepare it

For example here we use Airbnb Data, we have to predict the price based on accommodates


```
#read data
airbnb_list=pd.read_csv('paris_airbnb.csv')

#delete the non float and unwanted colomuns
airbnb_list=airbnb_list.drop(drop_cols,axis=1)
```

after reading the dataset, we have to clean the price column from symbols and convert from string to float


```
#clean the column price, remove comme ',' and '$'
airbnb_list['price']=airbnb_list['price'].str.replace('$','')
airbnb_list['price']=airbnb_list['price'].str.replace(',','')

#convert price column to float
airbnb_list['price']=airbnb_list['price'].astype('float')
```

if we execute
``` 
airbnb_list.isnull().sum()
```
we notice that many columns has empty/NULL values, so we delete those columns!

accommodates            0
bedrooms               24
bathrooms              58
beds                   14
price                   0
cleaning_fee         1750
security_deposit     1680
minimum_nights          0
maximum_nights          0
number_of_reviews       0

so we drop the columns (cleaning_fee,security_deposit)

```
airbnb_list=airbnb_list.drop(['cleaning_fee','security_deposit'],axis=1)
```

then we drop the rows with empty values to correct the columns (bedrooms,bathrooms,bed)
```
airbnb_list=airbnb_list.dropna(axis=0)
```

now we have clean data to use
accommodates         0
bedrooms             0
bathrooms            0
beds                 0
price                0
minimum_nights       0
maximum_nights       0
number_of_reviews    0


### PART 2 : Normalization and data split
we use the normal standard distribution to normalize out data

```
#normalize based on standard normal distribution
normalized_data=(airbnb_list-airbnb_list.mean())/(airbnb_list.std())
#facultative: drop the NULL lines after the data normalization
normalized_data=normalized_data.dropna(axis=0)
```
but it's important to not normalize the price column which represent the target, so we restore it
```
#denormalize the price column
normalized_data['price']=airbnb_list['price']
```

then we randomize the dataset, it's very important to do this in order to avoid what we call bias in machine learning, so you will divide the data randomly

```
#random the dataset
airbnb_list=airbnb_list.loc[np.random.permutation(len(airbnb_list))]
```
then for example we choose 75% for train, 25% for test

```
#separate into train data and test data 75% 25%
sup=int(0.75*len(normalized_data))
train_data=normalized_data.iloc[0:sup]
test_data=normalized_data.iloc[sup:]
```

### PART 3 : Training implementation

we use sklearn, so import  KNeighborsRegressor
```
from sklearn.neighbors import KNeighborsRegressor
```

we intialise the KNN model, brute algorithm is the Naive KNN without other ML optimisations
```
#init your knn model
knn=KNeighborsRegressor(n_neighbors=k,algorithm='brute')

```

```
#train features init
features = train_data.columns.tolist()

#remove price column from featues

features.remove('price')
train_features=train_data[features]


#train target init
train_target = train_data ['price']

#luanch knn learning
knn.fit(train_features,train_target)
```
In this part, we will apply the model on test data
### PART 4 : Predict the price and exploit model
```
predictions= knn.predict(test_data[features])
print(predictions)
```
### PART 5 : Evaluation
In this part we use sklearn.metrics to evalute the model
MSE: Mean Squared Error
RMSE: Root Mean Square Error
```
from sklearn.metrics import mean_squared_error

errorMSE=mean_squared_error(test_data['price'],predictions)

print('mse= ', errorMSE, '\\ rmse= ',errorMSE**0.5)
```


