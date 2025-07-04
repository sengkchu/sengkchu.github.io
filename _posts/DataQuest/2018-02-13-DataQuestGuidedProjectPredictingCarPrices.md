---
title: Dataquest Guided Project - Predicting Car Prices
date: 2018-02-13 00:00 +/-0000
categories: [DataQuest]
tags: [dataquest, data science, basics]
image:
  path: /posts_images/2018-02-13-DataQuestGuidedProjectPredictingCarPrices/cover.PNG
---


In this project we are going to look at 'imports-85.data'. This file contains specifications of vehicles in 1985. For more information on the data set click [here](https://archive.ics.uci.edu/ml/datasets/automobile). 

We are going to explore the fundamentals of machine learning using the k-nearest neighbors algorithm from scikit-learn. First, we'll import the libraries we'll need.


```python
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
cars = pd.read_csv("imports-85.data")

cars.head()
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
      <th>3</th>
      <th>?</th>
      <th>alfa-romero</th>
      <th>gas</th>
      <th>std</th>
      <th>two</th>
      <th>convertible</th>
      <th>rwd</th>
      <th>front</th>
      <th>88.60</th>
      <th>...</th>
      <th>130</th>
      <th>mpfi</th>
      <th>3.47</th>
      <th>2.68</th>
      <th>9.00</th>
      <th>111</th>
      <th>5000</th>
      <th>21</th>
      <th>27</th>
      <th>13495</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>?</td>
      <td>alfa-romero</td>
      <td>gas</td>
      <td>std</td>
      <td>two</td>
      <td>convertible</td>
      <td>rwd</td>
      <td>front</td>
      <td>88.6</td>
      <td>...</td>
      <td>130</td>
      <td>mpfi</td>
      <td>3.47</td>
      <td>2.68</td>
      <td>9.0</td>
      <td>111</td>
      <td>5000</td>
      <td>21</td>
      <td>27</td>
      <td>16500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>?</td>
      <td>alfa-romero</td>
      <td>gas</td>
      <td>std</td>
      <td>two</td>
      <td>hatchback</td>
      <td>rwd</td>
      <td>front</td>
      <td>94.5</td>
      <td>...</td>
      <td>152</td>
      <td>mpfi</td>
      <td>2.68</td>
      <td>3.47</td>
      <td>9.0</td>
      <td>154</td>
      <td>5000</td>
      <td>19</td>
      <td>26</td>
      <td>16500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>164</td>
      <td>audi</td>
      <td>gas</td>
      <td>std</td>
      <td>four</td>
      <td>sedan</td>
      <td>fwd</td>
      <td>front</td>
      <td>99.8</td>
      <td>...</td>
      <td>109</td>
      <td>mpfi</td>
      <td>3.19</td>
      <td>3.40</td>
      <td>10.0</td>
      <td>102</td>
      <td>5500</td>
      <td>24</td>
      <td>30</td>
      <td>13950</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>164</td>
      <td>audi</td>
      <td>gas</td>
      <td>std</td>
      <td>four</td>
      <td>sedan</td>
      <td>4wd</td>
      <td>front</td>
      <td>99.4</td>
      <td>...</td>
      <td>136</td>
      <td>mpfi</td>
      <td>3.19</td>
      <td>3.40</td>
      <td>8.0</td>
      <td>115</td>
      <td>5500</td>
      <td>18</td>
      <td>22</td>
      <td>17450</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>?</td>
      <td>audi</td>
      <td>gas</td>
      <td>std</td>
      <td>two</td>
      <td>sedan</td>
      <td>fwd</td>
      <td>front</td>
      <td>99.8</td>
      <td>...</td>
      <td>136</td>
      <td>mpfi</td>
      <td>3.19</td>
      <td>3.40</td>
      <td>8.5</td>
      <td>110</td>
      <td>5500</td>
      <td>19</td>
      <td>25</td>
      <td>15250</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 26 columns</p>
</div>



It looks like this dataset does not include the column names. We'll have to add in the column names manually using the documentation [here](https://archive.ics.uci.edu/ml/datasets/automobile). 


```python
colnames = ['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 
        'drive-wheels', 'engine-location', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type', 
        'num-of-cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-rate', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']


cars = pd.read_csv("imports-85.data", names=colnames)
```


```python
cars.head()
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
      <th>symboling</th>
      <th>normalized-losses</th>
      <th>make</th>
      <th>fuel-type</th>
      <th>aspiration</th>
      <th>num-of-doors</th>
      <th>body-style</th>
      <th>drive-wheels</th>
      <th>engine-location</th>
      <th>wheel-base</th>
      <th>...</th>
      <th>engine-size</th>
      <th>fuel-system</th>
      <th>bore</th>
      <th>stroke</th>
      <th>compression-rate</th>
      <th>horsepower</th>
      <th>peak-rpm</th>
      <th>city-mpg</th>
      <th>highway-mpg</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>?</td>
      <td>alfa-romero</td>
      <td>gas</td>
      <td>std</td>
      <td>two</td>
      <td>convertible</td>
      <td>rwd</td>
      <td>front</td>
      <td>88.6</td>
      <td>...</td>
      <td>130</td>
      <td>mpfi</td>
      <td>3.47</td>
      <td>2.68</td>
      <td>9.0</td>
      <td>111</td>
      <td>5000</td>
      <td>21</td>
      <td>27</td>
      <td>13495</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>?</td>
      <td>alfa-romero</td>
      <td>gas</td>
      <td>std</td>
      <td>two</td>
      <td>convertible</td>
      <td>rwd</td>
      <td>front</td>
      <td>88.6</td>
      <td>...</td>
      <td>130</td>
      <td>mpfi</td>
      <td>3.47</td>
      <td>2.68</td>
      <td>9.0</td>
      <td>111</td>
      <td>5000</td>
      <td>21</td>
      <td>27</td>
      <td>16500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>?</td>
      <td>alfa-romero</td>
      <td>gas</td>
      <td>std</td>
      <td>two</td>
      <td>hatchback</td>
      <td>rwd</td>
      <td>front</td>
      <td>94.5</td>
      <td>...</td>
      <td>152</td>
      <td>mpfi</td>
      <td>2.68</td>
      <td>3.47</td>
      <td>9.0</td>
      <td>154</td>
      <td>5000</td>
      <td>19</td>
      <td>26</td>
      <td>16500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>164</td>
      <td>audi</td>
      <td>gas</td>
      <td>std</td>
      <td>four</td>
      <td>sedan</td>
      <td>fwd</td>
      <td>front</td>
      <td>99.8</td>
      <td>...</td>
      <td>109</td>
      <td>mpfi</td>
      <td>3.19</td>
      <td>3.40</td>
      <td>10.0</td>
      <td>102</td>
      <td>5500</td>
      <td>24</td>
      <td>30</td>
      <td>13950</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>164</td>
      <td>audi</td>
      <td>gas</td>
      <td>std</td>
      <td>four</td>
      <td>sedan</td>
      <td>4wd</td>
      <td>front</td>
      <td>99.4</td>
      <td>...</td>
      <td>136</td>
      <td>mpfi</td>
      <td>3.19</td>
      <td>3.40</td>
      <td>8.0</td>
      <td>115</td>
      <td>5500</td>
      <td>18</td>
      <td>22</td>
      <td>17450</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 26 columns</p>
</div>



### Data Cleaning and Preparing the Features

---

Looks like we managed to fix the dataframe. The k-nearest neighbors algorithm uses the distance formula to determine the nearest neighbors. That means, we can only use numerical columns for this machine learning algorithm. So we'll have to do a little bit of data cleaning.

Here are some of the issues with this dataframe:

+ There are missing values with the string '?'.
+ There are many non numerical columns.

First, we'll replace the string value '?' with NaN. That way, we can use the .isnull() method to determine which columns have missing values. 

Using the documentation, we can determine which columns are numerical. Then we can drop them from the dataframe.


```python
cars = cars.replace("?", np.nan)
```


```python
to_drop = ["symboling", "make", "fuel-type", "aspiration", "num-of-doors", "body-style", "drive-wheels", "engine-location", "engine-type", "num-of-cylinders", "fuel-system", "engine-size"]

cars_num = cars.drop(to_drop, axis=1)
```


```python
cars_num.head()
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
      <th>normalized-losses</th>
      <th>wheel-base</th>
      <th>length</th>
      <th>width</th>
      <th>height</th>
      <th>curb-weight</th>
      <th>bore</th>
      <th>stroke</th>
      <th>compression-rate</th>
      <th>horsepower</th>
      <th>peak-rpm</th>
      <th>city-mpg</th>
      <th>highway-mpg</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>88.6</td>
      <td>168.8</td>
      <td>64.1</td>
      <td>48.8</td>
      <td>2548</td>
      <td>3.47</td>
      <td>2.68</td>
      <td>9.0</td>
      <td>111</td>
      <td>5000</td>
      <td>21</td>
      <td>27</td>
      <td>13495</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>88.6</td>
      <td>168.8</td>
      <td>64.1</td>
      <td>48.8</td>
      <td>2548</td>
      <td>3.47</td>
      <td>2.68</td>
      <td>9.0</td>
      <td>111</td>
      <td>5000</td>
      <td>21</td>
      <td>27</td>
      <td>16500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>94.5</td>
      <td>171.2</td>
      <td>65.5</td>
      <td>52.4</td>
      <td>2823</td>
      <td>2.68</td>
      <td>3.47</td>
      <td>9.0</td>
      <td>154</td>
      <td>5000</td>
      <td>19</td>
      <td>26</td>
      <td>16500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>164</td>
      <td>99.8</td>
      <td>176.6</td>
      <td>66.2</td>
      <td>54.3</td>
      <td>2337</td>
      <td>3.19</td>
      <td>3.40</td>
      <td>10.0</td>
      <td>102</td>
      <td>5500</td>
      <td>24</td>
      <td>30</td>
      <td>13950</td>
    </tr>
    <tr>
      <th>4</th>
      <td>164</td>
      <td>99.4</td>
      <td>176.6</td>
      <td>66.4</td>
      <td>54.3</td>
      <td>2824</td>
      <td>3.19</td>
      <td>3.40</td>
      <td>8.0</td>
      <td>115</td>
      <td>5500</td>
      <td>18</td>
      <td>22</td>
      <td>17450</td>
    </tr>
  </tbody>
</table>
</div>




```python
cars_num = cars_num.astype("float")
cars_num.isnull().sum()
```




    normalized-losses    41
    wheel-base            0
    length                0
    width                 0
    height                0
    curb-weight           0
    bore                  4
    stroke                4
    compression-rate      0
    horsepower            2
    peak-rpm              2
    city-mpg              0
    highway-mpg           0
    price                 4
    dtype: int64



We are going to use the machine learning algorithm to determine the price of a car. It doesn't make sense to keep rows with missing values in the 'price' column. So we'll just drop them entirely.

For the 'bore' and 'stroke' columns, we'll use the mean to fill in the missing values.


```python
cars_num = cars_num.dropna(subset=["price"])
cars_num.isnull().sum()
```




    normalized-losses    37
    wheel-base            0
    length                0
    width                 0
    height                0
    curb-weight           0
    bore                  4
    stroke                4
    compression-rate      0
    horsepower            2
    peak-rpm              2
    city-mpg              0
    highway-mpg           0
    price                 0
    dtype: int64




```python
cars_num = cars_num.fillna(cars_num.mean())
cars_num.isnull().sum()
```




    normalized-losses    0
    wheel-base           0
    length               0
    width                0
    height               0
    curb-weight          0
    bore                 0
    stroke               0
    compression-rate     0
    horsepower           0
    peak-rpm             0
    city-mpg             0
    highway-mpg          0
    price                0
    dtype: int64




```python
cars_num.head()
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
      <th>normalized-losses</th>
      <th>wheel-base</th>
      <th>length</th>
      <th>width</th>
      <th>height</th>
      <th>curb-weight</th>
      <th>bore</th>
      <th>stroke</th>
      <th>compression-rate</th>
      <th>horsepower</th>
      <th>peak-rpm</th>
      <th>city-mpg</th>
      <th>highway-mpg</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>122.0</td>
      <td>88.6</td>
      <td>168.8</td>
      <td>64.1</td>
      <td>48.8</td>
      <td>2548.0</td>
      <td>3.47</td>
      <td>2.68</td>
      <td>9.0</td>
      <td>111.0</td>
      <td>5000.0</td>
      <td>21.0</td>
      <td>27.0</td>
      <td>13495.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>122.0</td>
      <td>88.6</td>
      <td>168.8</td>
      <td>64.1</td>
      <td>48.8</td>
      <td>2548.0</td>
      <td>3.47</td>
      <td>2.68</td>
      <td>9.0</td>
      <td>111.0</td>
      <td>5000.0</td>
      <td>21.0</td>
      <td>27.0</td>
      <td>16500.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>122.0</td>
      <td>94.5</td>
      <td>171.2</td>
      <td>65.5</td>
      <td>52.4</td>
      <td>2823.0</td>
      <td>2.68</td>
      <td>3.47</td>
      <td>9.0</td>
      <td>154.0</td>
      <td>5000.0</td>
      <td>19.0</td>
      <td>26.0</td>
      <td>16500.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>164.0</td>
      <td>99.8</td>
      <td>176.6</td>
      <td>66.2</td>
      <td>54.3</td>
      <td>2337.0</td>
      <td>3.19</td>
      <td>3.40</td>
      <td>10.0</td>
      <td>102.0</td>
      <td>5500.0</td>
      <td>24.0</td>
      <td>30.0</td>
      <td>13950.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>164.0</td>
      <td>99.4</td>
      <td>176.6</td>
      <td>66.4</td>
      <td>54.3</td>
      <td>2824.0</td>
      <td>3.19</td>
      <td>3.40</td>
      <td>8.0</td>
      <td>115.0</td>
      <td>5500.0</td>
      <td>18.0</td>
      <td>22.0</td>
      <td>17450.0</td>
    </tr>
  </tbody>
</table>
</div>



The k-nearest neighbors algorithm uses the euclidean distance to determine the closest neighbor. 

\\[ Distance = \sqrt{\{(q_1-p_1)}^2+\{(q_2-p_2)}^2+...{(q_n-p_n)}^2} \\]

Where q and p represent two rows and the subscript representing a column. However, each column have different scaling. For example, if we take row 2, and row 3. The peak RPM has a difference of 500, while the difference in width is 0.7. The algorithm will give extra weight towards the difference in peak RPM.

That is why it is important to normalize the dataset into a unit vector. After normalization we'll have values from -1 to 1. For more information on feature scaling click [here](https://en.wikipedia.org/wiki/Feature_scaling).

\\[ x' = \frac{x - mean(x)}{x(max) - x(min)}\\]

In pandas this would be:

\\[ df' = \frac{df - df.mean()}{df.max() - df.min()}\\]

Where df is any dataframe.


```python
normalized_cars = (cars_num-cars_num.mean())/(cars_num.max()-cars_num.min())
normalized_cars['price'] = cars_num['price']
```


```python
normalized_cars.head()
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
      <th>normalized-losses</th>
      <th>wheel-base</th>
      <th>length</th>
      <th>width</th>
      <th>height</th>
      <th>curb-weight</th>
      <th>bore</th>
      <th>stroke</th>
      <th>compression-rate</th>
      <th>horsepower</th>
      <th>peak-rpm</th>
      <th>city-mpg</th>
      <th>highway-mpg</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000000</td>
      <td>-0.297289</td>
      <td>-0.080612</td>
      <td>-0.152911</td>
      <td>-0.413889</td>
      <td>-0.002974</td>
      <td>0.099492</td>
      <td>-0.274716</td>
      <td>-0.072767</td>
      <td>0.035528</td>
      <td>-0.047995</td>
      <td>-0.116086</td>
      <td>-0.097015</td>
      <td>13495.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.000000</td>
      <td>-0.297289</td>
      <td>-0.080612</td>
      <td>-0.152911</td>
      <td>-0.413889</td>
      <td>-0.002974</td>
      <td>0.099492</td>
      <td>-0.274716</td>
      <td>-0.072767</td>
      <td>0.035528</td>
      <td>-0.047995</td>
      <td>-0.116086</td>
      <td>-0.097015</td>
      <td>16500.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.000000</td>
      <td>-0.125277</td>
      <td>-0.044791</td>
      <td>-0.033253</td>
      <td>-0.113889</td>
      <td>0.103698</td>
      <td>-0.464793</td>
      <td>0.101474</td>
      <td>-0.072767</td>
      <td>0.236463</td>
      <td>-0.047995</td>
      <td>-0.171642</td>
      <td>-0.123331</td>
      <td>16500.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.219895</td>
      <td>0.029242</td>
      <td>0.035806</td>
      <td>0.026577</td>
      <td>0.044444</td>
      <td>-0.084820</td>
      <td>-0.100508</td>
      <td>0.068141</td>
      <td>-0.010267</td>
      <td>-0.006528</td>
      <td>0.156087</td>
      <td>-0.032753</td>
      <td>-0.018068</td>
      <td>13950.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.219895</td>
      <td>0.017580</td>
      <td>0.035806</td>
      <td>0.043671</td>
      <td>0.044444</td>
      <td>0.104086</td>
      <td>-0.100508</td>
      <td>0.068141</td>
      <td>-0.135267</td>
      <td>0.054220</td>
      <td>0.156087</td>
      <td>-0.199420</td>
      <td>-0.228594</td>
      <td>17450.0</td>
    </tr>
  </tbody>
</table>
</div>



### Applying Machine Learning
---

Suppose we have a dataframe named 'train', and a row named 'test'. The idea behind k-nearest neighbors is to find k number of rows from 'train' with the lowest distance to 'test'. Then  we can determine the average of the target column of 'train' of those five rows and return the result to 'test'. 

We are going to write a function that uses the KNeighborsRegressor class from scikit-learn. This works a little bit differently, the class actually generates a model that fits the training dataset. It is a regression method using k-nearest neighbors. More information on this can be found in the [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html#sklearn.neighbors.KNeighborsRegressor).


```python
#Returns the root mean squared error using KNN
def knn_train_test(features, target_col, df):
    #randomize sets
    np.random.seed(1)
    randomed_index = np.random.permutation(df.index)
    randomed_df = df.reindex(randomed_index)
    
    half_point = int(len(randomed_df)/2)
    
    #assign test and training sets
    train_df = randomed_df.iloc[0:half_point]
    test_df = randomed_df.iloc[half_point:]
    
    #training
    knn = KNeighborsRegressor()
    knn.fit(train_df[[features]], train_df[[target_col]])
    
    #test
    predictions = knn.predict(test_df[[features]])
    mse = mean_squared_error(test_df[[target_col]], predictions)
    rmse = mse**0.5
    return rmse
```

We can write a for loop and use the function for each column. That way, we can see the RMSE of each column.


```python
features = normalized_cars.columns.drop('price')
rmse = {}
for item in features:
    rmse[item] = knn_train_test(item, 'price', normalized_cars)

results = pd.Series(rmse)
results.sort_values()
```




    horsepower           4010.414152
    curb-weight          4401.118255
    highway-mpg          4652.697833
    width                4908.609914
    city-mpg             4973.940485
    length               5429.900973
    wheel-base           5460.787788
    compression-rate     6610.812153
    bore                 6806.695830
    normalized-losses    7304.373172
    peak-rpm             7678.470979
    height               7842.199226
    stroke               8005.611387
    dtype: float64



It looks like the 'horsepower' column has the least amount of error. We should definitely keep this list in mind when using the function for multiple features.

But first, let's modify the function to include k value or the number of neighbors as a parameter. Then we can loop through a list of K values and features to determine which K value and features are most optimal in our machine learning model.


```python
def knn_train_test2(features, target_col, df, k_values):
    #randomize sets
    np.random.seed(1)
    randomed_index = np.random.permutation(df.index)
    randomed_df = df.reindex(randomed_index)
    
    half_point = int(len(randomed_df)/2)
    
    #assign test and training sets
    train_df = randomed_df.iloc[0:half_point]
    test_df = randomed_df.iloc[half_point:]
    
    k_rmse = {}
    #training
    for k in k_values:
        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(train_df[[features]], train_df[[target_col]])
        
        #test
        predictions = knn.predict(test_df[[features]])
        mse = mean_squared_error(test_df[[target_col]], predictions)
        rmse = mse**0.5
        k_rmse[k] = rmse
    return k_rmse
```


```python
#input k parameter as a list, use function to return a dictionary of dictionaries
k = [1, 3, 5, 7, 9]
features = normalized_cars.columns.drop('price')
feature_k_rmse = {}

for item in features:
    feature_k_rmse[item] = knn_train_test2(item, 'price', normalized_cars, k)
    
feature_k_rmse
```




    {'bore': {1: 8602.58848450066,
      3: 6984.239489480916,
      5: 6806.695830075582,
      7: 6939.105845651802,
      9: 6915.297375013411},
     'city-mpg': {1: 5863.190943471308,
      3: 4672.77285307275,
      5: 4973.94048466108,
      7: 5413.390882677539,
      9: 5277.1766643494775},
     'compression-rate': {1: 8087.205346523092,
      3: 7375.063685578359,
      5: 6610.812153159129,
      7: 6732.801282941515,
      9: 7024.485525463435},
     'curb-weight': {1: 5288.0195725810245,
      3: 5022.318011757233,
      5: 4401.118254793124,
      7: 4330.6701276238755,
      9: 4633.425879994758},
     'height': {1: 8942.012951995952,
      3: 8378.23385277286,
      5: 7842.199225717336,
      7: 7709.0699416548505,
      9: 7777.1734491607085},
     'highway-mpg': {1: 6022.866724754784,
      3: 4671.390389789466,
      5: 4652.697832525993,
      7: 4817.230104360727,
      9: 5261.877043557105},
     'horsepower': {1: 4170.054848037801,
      3: 3985.1389178696736,
      5: 4010.4141521891734,
      7: 4351.268271181572,
      9: 4514.504641478055},
     'length': {1: 4611.990241761035,
      3: 5129.672039752984,
      5: 5429.900972639673,
      7: 5311.883616635263,
      9: 5383.054514833446},
     'normalized-losses': {1: 7829.153502413683,
      3: 7515.021862294153,
      5: 7304.373172258108,
      7: 7634.134919298568,
      9: 7682.244506601594},
     'peak-rpm': {1: 9511.480067750124,
      3: 8537.550899973421,
      5: 7678.470978516542,
      7: 7520.322843484608,
      9: 7364.560980451443},
     'stroke': {1: 9116.495955406906,
      3: 7338.68466990294,
      5: 8005.611386699424,
      7: 7788.91860301835,
      9: 7702.038219213702},
     'wheel-base': {1: 4493.734068810494,
      3: 5208.39331165465,
      5: 5460.78778823338,
      7: 5448.173408324034,
      9: 5738.621574471594},
     'width': {1: 4559.257297950061,
      3: 4648.149766156945,
      5: 4908.609914413773,
      7: 4781.944236558163,
      9: 4719.070452207012}}




```python
best_features = {}
plt.figure(figsize=(10, 12))

for key, value in feature_k_rmse.items():
    x = list(value.keys())
    y = list(value.values())
    
    order = np.argsort(x)
    x_ordered = np.array(x)[order]
    y_ordered = np.array(y)[order]
    print(key)
    print('average_rmse: '+str(np.mean(y)))
    best_features[key] = np.mean(y)

    plt.plot(x_ordered, y_ordered, label=key)
    plt.xlabel("K_value")
    plt.ylabel("RMSE")
plt.legend()
plt.show()
```

    horsepower
    average_rmse: 4206.276166151255
    wheel-base
    average_rmse: 5269.94203029883
    width
    average_rmse: 4723.406333457191
    peak-rpm
    average_rmse: 8122.477154035228
    city-mpg
    average_rmse: 5240.0943656464315
    stroke
    average_rmse: 7990.349766848265
    length
    average_rmse: 5173.30027712448
    bore
    average_rmse: 7249.585404944475
    height
    average_rmse: 8129.737884260341
    curb-weight
    average_rmse: 4735.110369350003
    highway-mpg
    average_rmse: 5085.212418997615
    compression-rate
    average_rmse: 7166.0735987331045
    normalized-losses
    average_rmse: 7592.985592573221
    


    
![png](/posts_images/2018-02-13-DataQuestGuidedProjectPredictingCarPrices/output_25_1.png)
    


This figure is a bit confusing to look at. A better way is to sort the values of the best_features which contains the features as the key and the average RMSE as the values.


```python
sorted_features_list = sorted(best_features, key=best_features.get)
sorted_features_list
```




    ['horsepower',
     'width',
     'curb-weight',
     'highway-mpg',
     'length',
     'city-mpg',
     'wheel-base',
     'compression-rate',
     'bore',
     'normalized-losses',
     'stroke',
     'peak-rpm',
     'height']



Now we know which features have the lowest amount of error, we can begin applying the function to multiple features at once.


```python
def knn_train_test3(features, target_col, df):
    #randomize sets
    np.random.seed(0)
    randomed_index = np.random.permutation(df.index)
    randomed_df = df.reindex(randomed_index)
    
    half_point = int(len(randomed_df)/2)
    
    #assign test and training sets
    train_df = randomed_df.iloc[0:half_point]
    test_df = randomed_df.iloc[half_point:]
    
    #training
    knn = KNeighborsRegressor(n_neighbors=5)
    knn.fit(train_df[features], train_df[[target_col]])
    #test
    predictions = knn.predict(test_df[features])
    mse = mean_squared_error(test_df[[target_col]], predictions)
    rmse = mse**0.5
    return rmse
```


```python
k_rmse_features ={}

best_two_features = sorted_features_list[0:2]
best_three_features = sorted_features_list[0:3]
best_four_features = sorted_features_list[0:4]
best_five_features = sorted_features_list[0:5]


k_rmse_features["best_two_rmse"]  = knn_train_test3(best_two_features, 'price', normalized_cars)
k_rmse_features["best_three_rmse"] = knn_train_test3(best_three_features, 'price', normalized_cars)
k_rmse_features["best_four_rmse"] = knn_train_test3(best_four_features, 'price', normalized_cars)
k_rmse_features["best_five_rmse"] = knn_train_test3(best_five_features, 'price', normalized_cars)
```


```python
k_rmse_features
```




    {'best_five_rmse': 3533.7489988020734,
     'best_four_rmse': 3404.6909417321376,
     'best_three_rmse': 3214.9121121904577,
     'best_two_rmse': 3635.0424706141075}



Let looks like using the best three features gave us the lowest RMSE. 

Now, let's try varying the K values. We can further tune our machine learning model by finding the optimal K value to use.


```python
def knn_train_test4(features, target_col, df, k_values):
    #randomize sets
    np.random.seed(0)
    randomed_index = np.random.permutation(df.index)
    randomed_df = df.reindex(randomed_index)
    
    half_point = int(len(randomed_df)/2)
    
    #assign test and training sets
    train_df = randomed_df.iloc[0:half_point]
    test_df = randomed_df.iloc[half_point:]
    
    k_rmse = {}
    #training
    for k in k_values:
        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(train_df[features], train_df[[target_col]])
        #test
        predictions = knn.predict(test_df[features])
        mse = mean_squared_error(test_df[[target_col]], predictions)
        rmse = mse**0.5
        k_rmse[k] = rmse
    return k_rmse
```


```python
#input k parameter as a list, use function to return a dictionary of dictionaries
k = list(range(1,25))
features = [best_three_features, best_four_features, best_five_features]
feature_k_rmse2 = {}
feature_k_rmse2["best_three_features"] = knn_train_test4(best_three_features, 'price', normalized_cars, k)
feature_k_rmse2["best_four_features"] = knn_train_test4(best_four_features, 'price', normalized_cars, k)
feature_k_rmse2["best_five_features"] = knn_train_test4(best_five_features, 'price', normalized_cars, k)
```


```python
feature_k_rmse2
```




    {'best_five_features': {1: 2925.271682398335,
      2: 3052.8623032449823,
      3: 3142.306197305374,
      4: 3461.561181031581,
      5: 3533.7489988020734,
      6: 3374.173588712601,
      7: 3375.877721539395,
      8: 3324.876540782833,
      9: 3312.502627933449,
      10: 3366.851324397034,
      11: 3447.2822267103743,
      12: 3496.4998131087186,
      13: 3547.2160454903924,
      14: 3615.6147456955036,
      15: 3579.8430331574878,
      16: 3678.503216850985,
      17: 3750.96940429519,
      18: 3815.3901236791603,
      19: 3851.386198466123,
      20: 3939.9382237297746,
      21: 3975.0410078767027,
      22: 4005.3349453198225,
      23: 4054.671493676685,
      24: 4106.550851309336},
     'best_four_features': {1: 2870.800286876242,
      2: 2924.256000834373,
      3: 3217.2983830519292,
      4: 3392.3729838615423,
      5: 3404.6909417321376,
      6: 3532.1939129716366,
      7: 3523.454893817346,
      8: 3405.9189129672363,
      9: 3400.2247346580157,
      10: 3549.4612599577126,
      11: 3539.83054253496,
      12: 3562.1876094096256,
      13: 3675.4563993723814,
      14: 3770.166928307887,
      15: 3818.056456588688,
      16: 3804.305744235327,
      17: 3859.410326582298,
      18: 3863.6635808098727,
      19: 3905.7831606401637,
      20: 3927.4475182570377,
      21: 3956.7797179695103,
      22: 3966.0139478992855,
      23: 4016.972042009795,
      24: 4025.3723801365027},
     'best_three_features': {1: 2879.8872399542806,
      2: 2824.3660223522415,
      3: 2869.933838736334,
      4: 3141.137015378538,
      5: 3214.9121121904577,
      6: 3280.710541238062,
      7: 3421.1652350874347,
      8: 3379.778228190607,
      9: 3464.5878589864906,
      10: 3575.3981609613693,
      11: 3617.9433332156923,
      12: 3630.7516137575535,
      13: 3632.7823295462845,
      14: 3754.4036203105343,
      15: 3837.677054164043,
      16: 3841.760205882436,
      17: 3858.9773285906977,
      18: 3844.3626311824146,
      19: 3875.9053428113702,
      20: 3869.5372272718314,
      21: 3901.5634407576663,
      22: 3926.0570825223363,
      23: 4019.34604419612,
      24: 4048.5637998085253}}




```python
plt.figure(figsize=(6, 6))

for key, value in feature_k_rmse2.items():
    
    x = list(value.keys())
    y = list(value.values())
    plt.plot(x, y, label=key)
    plt.xlabel("k_value")
    plt.ylabel("RMSE")
    
plt.legend()
plt.show()
```


    
![png](/posts_images/2018-02-13-DataQuestGuidedProjectPredictingCarPrices/output_36_0.png)
    


From the chart above, we can see that choosing the best three features with a K value of 2 will give us the RMSE of 2824. That is it for now though, the goal of this project is to explore the fundamentals of machine learning.

---

#### Learning Summary

Concepts explored: pandas, data cleaning, features engineering, k-nearest neighbors, hyperparameter tuning, RMSE

Functions and methods used: .read_csv(), .replace(), .drop(), .astype(), isnull().sum(), .min(), .max(), .mean(), .permutation(), .reindex(), .iloc[], .fit(), .predict(), mean_squared_error(), .Series(), .sort_values(), .plot(), .legend()


The files used for this project can be found in my [GitHub repository](https://github.com/sengkchu/Dataquest-Guided-Projects-Solutions/tree/master/Guided%20Project_%20Predicting%20Car%20Prices).

