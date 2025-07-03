---
title: Dataquest Guided Project - Predicting Bike Rentals
date: 2018-02-14 00:00 +/-0000
categories: [DataQuest]
tags: [dataquest, data science, basics]
---


In this project, we are going to look at 'bike_rental_hour.csv', a dataset that contains the hourly and daily count of rental bikes between years 2011 and 2012 in the Capital bikeshare system. From the dataset, we are going to apply various machine learning algorithms to generate a model that can predict the number of bike rentals.

For more information on this dataset click [here](https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset).


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
%matplotlib inline

bike_rentals = pd.read_csv("bike_rental_hour.csv")
bike_rentals.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>instant</th>
      <th>dteday</th>
      <th>season</th>
      <th>yr</th>
      <th>mnth</th>
      <th>hr</th>
      <th>holiday</th>
      <th>weekday</th>
      <th>workingday</th>
      <th>weathersit</th>
      <th>temp</th>
      <th>atemp</th>
      <th>hum</th>
      <th>windspeed</th>
      <th>casual</th>
      <th>registered</th>
      <th>cnt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2011-01-01</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0.24</td>
      <td>0.2879</td>
      <td>0.81</td>
      <td>0.0</td>
      <td>3</td>
      <td>13</td>
      <td>16</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2011-01-01</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0.22</td>
      <td>0.2727</td>
      <td>0.80</td>
      <td>0.0</td>
      <td>8</td>
      <td>32</td>
      <td>40</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>2011-01-01</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0.22</td>
      <td>0.2727</td>
      <td>0.80</td>
      <td>0.0</td>
      <td>5</td>
      <td>27</td>
      <td>32</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>2011-01-01</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0.24</td>
      <td>0.2879</td>
      <td>0.75</td>
      <td>0.0</td>
      <td>3</td>
      <td>10</td>
      <td>13</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>2011-01-01</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0.24</td>
      <td>0.2879</td>
      <td>0.75</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



### Data Analysis and Feature Engineering

---


```python
plt.hist(bike_rentals['cnt'])
```




    (array([ 6972.,  3705.,  2659.,  1660.,   987.,   663.,   369.,   188.,
              139.,    37.]),
     array([   1. ,   98.6,  196.2,  293.8,  391.4,  489. ,  586.6,  684.2,
             781.8,  879.4,  977. ]),
     <a list of 10 Patch objects>)




    
![png](/posts_images/2018-02-14-DataQuestGuidedProjectPredictingBikeRentals/output_3_1.png)
    



```python
bike_rentals['cnt'].describe()
```




    count    17379.000000
    mean       189.463088
    std        181.387599
    min          1.000000
    25%         40.000000
    50%        142.000000
    75%        281.000000
    max        977.000000
    Name: cnt, dtype: float64



Using a histogram, we've quickly plotted the distribution of the 'cnt' column. This is the total number of bike rentals for a particular hour of a day. We can see that this is a right skewed distribution. The 50% percentile, or the median is 142. 

We can add a feature to the dataset. By splitting the day to four time brackets, we can create a new column, 'time_label'.


```python
def assign_label(hour):
    if hour > 6 and hour <= 12:
        return 1
    elif hour > 12 and hour <= 18:
        return 2
    elif hour > 18 and hour <= 24:
        return 3
    else:
        return 4
```


```python
bike_rentals['time_label'] = bike_rentals['hr'].apply(assign_label)
```

Next, Let's take a look at the correlation of this dataset.


```python
bike_rentals.corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>instant</th>
      <th>season</th>
      <th>yr</th>
      <th>mnth</th>
      <th>hr</th>
      <th>holiday</th>
      <th>weekday</th>
      <th>workingday</th>
      <th>weathersit</th>
      <th>temp</th>
      <th>atemp</th>
      <th>hum</th>
      <th>windspeed</th>
      <th>casual</th>
      <th>registered</th>
      <th>cnt</th>
      <th>time_label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>instant</th>
      <td>1.000000</td>
      <td>0.404046</td>
      <td>0.866014</td>
      <td>0.489164</td>
      <td>-0.004775</td>
      <td>0.014723</td>
      <td>0.001357</td>
      <td>-0.003416</td>
      <td>-0.014198</td>
      <td>0.136178</td>
      <td>0.137615</td>
      <td>0.009577</td>
      <td>-0.074505</td>
      <td>0.158295</td>
      <td>0.282046</td>
      <td>0.278379</td>
      <td>0.006533</td>
    </tr>
    <tr>
      <th>season</th>
      <td>0.404046</td>
      <td>1.000000</td>
      <td>-0.010742</td>
      <td>0.830386</td>
      <td>-0.006117</td>
      <td>-0.009585</td>
      <td>-0.002335</td>
      <td>0.013743</td>
      <td>-0.014524</td>
      <td>0.312025</td>
      <td>0.319380</td>
      <td>0.150625</td>
      <td>-0.149773</td>
      <td>0.120206</td>
      <td>0.174226</td>
      <td>0.178056</td>
      <td>0.006467</td>
    </tr>
    <tr>
      <th>yr</th>
      <td>0.866014</td>
      <td>-0.010742</td>
      <td>1.000000</td>
      <td>-0.010473</td>
      <td>-0.003867</td>
      <td>0.006692</td>
      <td>-0.004485</td>
      <td>-0.002196</td>
      <td>-0.019157</td>
      <td>0.040913</td>
      <td>0.039222</td>
      <td>-0.083546</td>
      <td>-0.008740</td>
      <td>0.142779</td>
      <td>0.253684</td>
      <td>0.250495</td>
      <td>0.004770</td>
    </tr>
    <tr>
      <th>mnth</th>
      <td>0.489164</td>
      <td>0.830386</td>
      <td>-0.010473</td>
      <td>1.000000</td>
      <td>-0.005772</td>
      <td>0.018430</td>
      <td>0.010400</td>
      <td>-0.003477</td>
      <td>0.005400</td>
      <td>0.201691</td>
      <td>0.208096</td>
      <td>0.164411</td>
      <td>-0.135386</td>
      <td>0.068457</td>
      <td>0.122273</td>
      <td>0.120638</td>
      <td>0.005782</td>
    </tr>
    <tr>
      <th>hr</th>
      <td>-0.004775</td>
      <td>-0.006117</td>
      <td>-0.003867</td>
      <td>-0.005772</td>
      <td>1.000000</td>
      <td>0.000479</td>
      <td>-0.003498</td>
      <td>0.002285</td>
      <td>-0.020203</td>
      <td>0.137603</td>
      <td>0.133750</td>
      <td>-0.276498</td>
      <td>0.137252</td>
      <td>0.301202</td>
      <td>0.374141</td>
      <td>0.394071</td>
      <td>-0.305052</td>
    </tr>
    <tr>
      <th>holiday</th>
      <td>0.014723</td>
      <td>-0.009585</td>
      <td>0.006692</td>
      <td>0.018430</td>
      <td>0.000479</td>
      <td>1.000000</td>
      <td>-0.102088</td>
      <td>-0.252471</td>
      <td>-0.017036</td>
      <td>-0.027340</td>
      <td>-0.030973</td>
      <td>-0.010588</td>
      <td>0.003988</td>
      <td>0.031564</td>
      <td>-0.047345</td>
      <td>-0.030927</td>
      <td>-0.000586</td>
    </tr>
    <tr>
      <th>weekday</th>
      <td>0.001357</td>
      <td>-0.002335</td>
      <td>-0.004485</td>
      <td>0.010400</td>
      <td>-0.003498</td>
      <td>-0.102088</td>
      <td>1.000000</td>
      <td>0.035955</td>
      <td>0.003311</td>
      <td>-0.001795</td>
      <td>-0.008821</td>
      <td>-0.037158</td>
      <td>0.011502</td>
      <td>0.032721</td>
      <td>0.021578</td>
      <td>0.026900</td>
      <td>0.002636</td>
    </tr>
    <tr>
      <th>workingday</th>
      <td>-0.003416</td>
      <td>0.013743</td>
      <td>-0.002196</td>
      <td>-0.003477</td>
      <td>0.002285</td>
      <td>-0.252471</td>
      <td>0.035955</td>
      <td>1.000000</td>
      <td>0.044672</td>
      <td>0.055390</td>
      <td>0.054667</td>
      <td>0.015688</td>
      <td>-0.011830</td>
      <td>-0.300942</td>
      <td>0.134326</td>
      <td>0.030284</td>
      <td>-0.000640</td>
    </tr>
    <tr>
      <th>weathersit</th>
      <td>-0.014198</td>
      <td>-0.014524</td>
      <td>-0.019157</td>
      <td>0.005400</td>
      <td>-0.020203</td>
      <td>-0.017036</td>
      <td>0.003311</td>
      <td>0.044672</td>
      <td>1.000000</td>
      <td>-0.102640</td>
      <td>-0.105563</td>
      <td>0.418130</td>
      <td>0.026226</td>
      <td>-0.152628</td>
      <td>-0.120966</td>
      <td>-0.142426</td>
      <td>-0.031821</td>
    </tr>
    <tr>
      <th>temp</th>
      <td>0.136178</td>
      <td>0.312025</td>
      <td>0.040913</td>
      <td>0.201691</td>
      <td>0.137603</td>
      <td>-0.027340</td>
      <td>-0.001795</td>
      <td>0.055390</td>
      <td>-0.102640</td>
      <td>1.000000</td>
      <td>0.987672</td>
      <td>-0.069881</td>
      <td>-0.023125</td>
      <td>0.459616</td>
      <td>0.335361</td>
      <td>0.404772</td>
      <td>-0.112537</td>
    </tr>
    <tr>
      <th>atemp</th>
      <td>0.137615</td>
      <td>0.319380</td>
      <td>0.039222</td>
      <td>0.208096</td>
      <td>0.133750</td>
      <td>-0.030973</td>
      <td>-0.008821</td>
      <td>0.054667</td>
      <td>-0.105563</td>
      <td>0.987672</td>
      <td>1.000000</td>
      <td>-0.051918</td>
      <td>-0.062336</td>
      <td>0.454080</td>
      <td>0.332559</td>
      <td>0.400929</td>
      <td>-0.107018</td>
    </tr>
    <tr>
      <th>hum</th>
      <td>0.009577</td>
      <td>0.150625</td>
      <td>-0.083546</td>
      <td>0.164411</td>
      <td>-0.276498</td>
      <td>-0.010588</td>
      <td>-0.037158</td>
      <td>0.015688</td>
      <td>0.418130</td>
      <td>-0.069881</td>
      <td>-0.051918</td>
      <td>1.000000</td>
      <td>-0.290105</td>
      <td>-0.347028</td>
      <td>-0.273933</td>
      <td>-0.322911</td>
      <td>0.240154</td>
    </tr>
    <tr>
      <th>windspeed</th>
      <td>-0.074505</td>
      <td>-0.149773</td>
      <td>-0.008740</td>
      <td>-0.135386</td>
      <td>0.137252</td>
      <td>0.003988</td>
      <td>0.011502</td>
      <td>-0.011830</td>
      <td>0.026226</td>
      <td>-0.023125</td>
      <td>-0.062336</td>
      <td>-0.290105</td>
      <td>1.000000</td>
      <td>0.090287</td>
      <td>0.082321</td>
      <td>0.093234</td>
      <td>-0.152740</td>
    </tr>
    <tr>
      <th>casual</th>
      <td>0.158295</td>
      <td>0.120206</td>
      <td>0.142779</td>
      <td>0.068457</td>
      <td>0.301202</td>
      <td>0.031564</td>
      <td>0.032721</td>
      <td>-0.300942</td>
      <td>-0.152628</td>
      <td>0.459616</td>
      <td>0.454080</td>
      <td>-0.347028</td>
      <td>0.090287</td>
      <td>1.000000</td>
      <td>0.506618</td>
      <td>0.694564</td>
      <td>-0.354446</td>
    </tr>
    <tr>
      <th>registered</th>
      <td>0.282046</td>
      <td>0.174226</td>
      <td>0.253684</td>
      <td>0.122273</td>
      <td>0.374141</td>
      <td>-0.047345</td>
      <td>0.021578</td>
      <td>0.134326</td>
      <td>-0.120966</td>
      <td>0.335361</td>
      <td>0.332559</td>
      <td>-0.273933</td>
      <td>0.082321</td>
      <td>0.506618</td>
      <td>1.000000</td>
      <td>0.972151</td>
      <td>-0.477057</td>
    </tr>
    <tr>
      <th>cnt</th>
      <td>0.278379</td>
      <td>0.178056</td>
      <td>0.250495</td>
      <td>0.120638</td>
      <td>0.394071</td>
      <td>-0.030927</td>
      <td>0.026900</td>
      <td>0.030284</td>
      <td>-0.142426</td>
      <td>0.404772</td>
      <td>0.400929</td>
      <td>-0.322911</td>
      <td>0.093234</td>
      <td>0.694564</td>
      <td>0.972151</td>
      <td>1.000000</td>
      <td>-0.494422</td>
    </tr>
    <tr>
      <th>time_label</th>
      <td>0.006533</td>
      <td>0.006467</td>
      <td>0.004770</td>
      <td>0.005782</td>
      <td>-0.305052</td>
      <td>-0.000586</td>
      <td>0.002636</td>
      <td>-0.000640</td>
      <td>-0.031821</td>
      <td>-0.112537</td>
      <td>-0.107018</td>
      <td>0.240154</td>
      <td>-0.152740</td>
      <td>-0.354446</td>
      <td>-0.477057</td>
      <td>-0.494422</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
correlations = bike_rentals.corr()
correlations['cnt']
```




    instant       0.278379
    season        0.178056
    yr            0.250495
    mnth          0.120638
    hr            0.394071
    holiday      -0.030927
    weekday       0.026900
    workingday    0.030284
    weathersit   -0.142426
    temp          0.404772
    atemp         0.400929
    hum          -0.322911
    windspeed     0.093234
    casual        0.694564
    registered    0.972151
    cnt           1.000000
    time_label   -0.494422
    Name: cnt, dtype: float64



We are not really seeing very strong correlations in these columns. The 'casual' and 'registered' columns are simply subcategories of the 'cnt' column. These columns leak information on the target column so we'll have to drop them. The 'dteday' column is just the date, and can't be used in this machine learning exercise.



```python
columns = bike_rentals.columns.drop(['cnt', 'casual', 'dteday', 'registered'])
columns
```




    Index(['instant', 'season', 'yr', 'mnth', 'hr', 'holiday', 'weekday',
           'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed',
           'time_label'],
          dtype='object')



### Applying Machine Learning

---

In order to prepare for machine learning, we'll need to split the data into a training set and a testing set. We can use the math module to randomly sample 80% of the data and assign it to the training set. Then we can use the remaining 20% as the testing set.

We will use MSE for the evauation of this machine learning algorithm. MSE works well since the target column 'cnt' is continuous.


```python
import math

#Sample 80% of the data randomly and assigns it to train.
eighty_percent_values = math.floor(bike_rentals.shape[0]*0.8)
train = bike_rentals.sample(n=eighty_percent_values, random_state = 1)

#Selects the remaining 20% to test.
test = bike_rentals.drop(train.index)
```


```python
train.shape[0] + test.shape[0] == bike_rentals.shape[0]
```




    True



Let's start by trying a simple linear regression model and checking the error of both the testing set and the training set.


```python
lr = LinearRegression()
lr.fit(train[columns], train['cnt'])
predictions_test = lr.predict(test[columns])
mse_test = mean_squared_error(test['cnt'], predictions_test)
mse_test
```




    15848.500195099274




```python
predictions_train = lr.predict(train[columns])
mse_train = mean_squared_error(train['cnt'], predictions_train)
mse_train
```




    16262.308571201114



Both the training set and the test set showed high error.The linear regression model is probably not the best for this dataset. We can use the decision tree to see if we can improve our predictions. The linear regression method is great for datasets with lots of continuous data, but a lot of the columns in this dataset is not continuous, but rather categorical.

Let's start by using a single tree model.


```python
tree = DecisionTreeRegressor(min_samples_leaf=5)
tree.fit(train[columns], train['cnt'])
predictions = tree.predict(test[columns])

mse = mean_squared_error(test['cnt'], predictions)
mse
```




    2734.3272105876699



As we can see, the decision tree model reduced our error signficantly. We can further improve our results if we use a forest of decision trees to reduce overfitting.


```python
tree = RandomForestRegressor(min_samples_leaf=2, n_estimators=250)
tree.fit(train[columns], train['cnt'])
predictions = tree.predict(test[columns])

mse = mean_squared_error(test['cnt'], predictions)
mse
```




    1851.3854653532792



We specified the hyperparameter values 'min_samples_leaf' and 'n_estimators', we can optimize these values by using a for loop.


```python
mse_leaf=[]
for i in range(1, 10):
    tree = RandomForestRegressor(min_samples_leaf=i, n_estimators=250)
    tree.fit(train[columns], train['cnt'])
    predictions = tree.predict(test[columns])

    mse = mean_squared_error(test['cnt'], predictions)
    mse_leaf.append(mse)
mse_leaf
```




    [1854.7577391990794,
     1839.0620577774473,
     1925.0742908051077,
     1969.0264103911857,
     2049.1460500826547,
     2074.0831040747389,
     2132.6560175009272,
     2189.186577585444,
     2285.3912359439528]




```python
n_trees = [250, 500, 750]
mse_trees=[]
for i in n_trees:
    tree = RandomForestRegressor(min_samples_leaf=1, n_estimators=i)
    tree.fit(train[columns], train['cnt'])
    predictions = tree.predict(test[columns])

    mse = mean_squared_error(test['cnt'], predictions)
    mse_trees.append(mse)
mse_trees
```




    [1823.6687851507481, 1828.0975978550055, 1812.9346447554024]



Using 750 trees and 1 min_samples_leaf, we managed to slightly lower the MSE down to 1812. The random forest regressor is a powerful tool. However, using a large amount of trees in conjunction with a for loop takes a very long time to process.

---

#### Learning Summary

Concepts explored:: pandas, matplotlib, features engineering, linear regression, decision trees, random forests, MSE

Functions, methods, and properties used:.hist(), .apply(), .corr(), .columns, .drop(), .sample(), .index, .floor(),.fit() .predict(), .mean_squared_error(), .append()

The files used for this project can be found in my [GitHub repository](https://github.com/sengkchu/Dataquest-Guided-Projects-Solutions/tree/master/Guided%20Project_%20Predicting%20Bike%20Rentals).
