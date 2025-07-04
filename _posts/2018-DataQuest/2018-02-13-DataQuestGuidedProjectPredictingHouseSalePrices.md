---
title: Dataquest Guided Project - Predicting House Sale Prices
date: 2018-02-13 00:00 +/-0000
categories: [DataQuest]
tags: [dataquest, basics]
image:
  path: /posts_images/2018-02-13-DataQuestGuidedProjectPredictingHouseSalePrices/cover.PNG
---

In this project, we are going to apply machine learning algorithms to predict the price of a house using 'AmesHousing.tsv'. In order to do so, we'll have to transform the data and apply various feature engineering techniques.

We will be focusing on the linear regression model, and use RMSE as the error metric. First let's explore the data.


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
%matplotlib inline

pd.set_option('display.max_columns', 500)
data = pd.read_csv("AmesHousing.tsv", delimiter='\t')
```


```python
print(data.shape)
print(len(str(data.shape))*'-')
print(data.dtypes.value_counts())
data.head()
```

    (2930, 82)
    ----------
    object     43
    int64      28
    float64    11
    dtype: int64
    




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
      <th>Order</th>
      <th>PID</th>
      <th>MS SubClass</th>
      <th>MS Zoning</th>
      <th>Lot Frontage</th>
      <th>Lot Area</th>
      <th>Street</th>
      <th>Alley</th>
      <th>Lot Shape</th>
      <th>Land Contour</th>
      <th>Utilities</th>
      <th>Lot Config</th>
      <th>Land Slope</th>
      <th>Neighborhood</th>
      <th>Condition 1</th>
      <th>Condition 2</th>
      <th>Bldg Type</th>
      <th>House Style</th>
      <th>Overall Qual</th>
      <th>Overall Cond</th>
      <th>Year Built</th>
      <th>Year Remod/Add</th>
      <th>Roof Style</th>
      <th>Roof Matl</th>
      <th>Exterior 1st</th>
      <th>Exterior 2nd</th>
      <th>Mas Vnr Type</th>
      <th>Mas Vnr Area</th>
      <th>Exter Qual</th>
      <th>Exter Cond</th>
      <th>Foundation</th>
      <th>Bsmt Qual</th>
      <th>Bsmt Cond</th>
      <th>Bsmt Exposure</th>
      <th>BsmtFin Type 1</th>
      <th>BsmtFin SF 1</th>
      <th>BsmtFin Type 2</th>
      <th>BsmtFin SF 2</th>
      <th>Bsmt Unf SF</th>
      <th>Total Bsmt SF</th>
      <th>Heating</th>
      <th>Heating QC</th>
      <th>Central Air</th>
      <th>Electrical</th>
      <th>1st Flr SF</th>
      <th>2nd Flr SF</th>
      <th>Low Qual Fin SF</th>
      <th>Gr Liv Area</th>
      <th>Bsmt Full Bath</th>
      <th>Bsmt Half Bath</th>
      <th>Full Bath</th>
      <th>Half Bath</th>
      <th>Bedroom AbvGr</th>
      <th>Kitchen AbvGr</th>
      <th>Kitchen Qual</th>
      <th>TotRms AbvGrd</th>
      <th>Functional</th>
      <th>Fireplaces</th>
      <th>Fireplace Qu</th>
      <th>Garage Type</th>
      <th>Garage Yr Blt</th>
      <th>Garage Finish</th>
      <th>Garage Cars</th>
      <th>Garage Area</th>
      <th>Garage Qual</th>
      <th>Garage Cond</th>
      <th>Paved Drive</th>
      <th>Wood Deck SF</th>
      <th>Open Porch SF</th>
      <th>Enclosed Porch</th>
      <th>3Ssn Porch</th>
      <th>Screen Porch</th>
      <th>Pool Area</th>
      <th>Pool QC</th>
      <th>Fence</th>
      <th>Misc Feature</th>
      <th>Misc Val</th>
      <th>Mo Sold</th>
      <th>Yr Sold</th>
      <th>Sale Type</th>
      <th>Sale Condition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>526301100</td>
      <td>20</td>
      <td>RL</td>
      <td>141.0</td>
      <td>31770</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Corner</td>
      <td>Gtl</td>
      <td>NAmes</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>6</td>
      <td>5</td>
      <td>1960</td>
      <td>1960</td>
      <td>Hip</td>
      <td>CompShg</td>
      <td>BrkFace</td>
      <td>Plywood</td>
      <td>Stone</td>
      <td>112.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>TA</td>
      <td>Gd</td>
      <td>Gd</td>
      <td>BLQ</td>
      <td>639.0</td>
      <td>Unf</td>
      <td>0.0</td>
      <td>441.0</td>
      <td>1080.0</td>
      <td>GasA</td>
      <td>Fa</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>1656</td>
      <td>0</td>
      <td>0</td>
      <td>1656</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>TA</td>
      <td>7</td>
      <td>Typ</td>
      <td>2</td>
      <td>Gd</td>
      <td>Attchd</td>
      <td>1960.0</td>
      <td>Fin</td>
      <td>2.0</td>
      <td>528.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>P</td>
      <td>210</td>
      <td>62</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>215000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>526350040</td>
      <td>20</td>
      <td>RH</td>
      <td>80.0</td>
      <td>11622</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>NAmes</td>
      <td>Feedr</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>5</td>
      <td>6</td>
      <td>1961</td>
      <td>1961</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>None</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>TA</td>
      <td>TA</td>
      <td>No</td>
      <td>Rec</td>
      <td>468.0</td>
      <td>LwQ</td>
      <td>144.0</td>
      <td>270.0</td>
      <td>882.0</td>
      <td>GasA</td>
      <td>TA</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>896</td>
      <td>0</td>
      <td>0</td>
      <td>896</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>TA</td>
      <td>5</td>
      <td>Typ</td>
      <td>0</td>
      <td>NaN</td>
      <td>Attchd</td>
      <td>1961.0</td>
      <td>Unf</td>
      <td>1.0</td>
      <td>730.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>140</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>120</td>
      <td>0</td>
      <td>NaN</td>
      <td>MnPrv</td>
      <td>NaN</td>
      <td>0</td>
      <td>6</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>105000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>526351010</td>
      <td>20</td>
      <td>RL</td>
      <td>81.0</td>
      <td>14267</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Corner</td>
      <td>Gtl</td>
      <td>NAmes</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>6</td>
      <td>6</td>
      <td>1958</td>
      <td>1958</td>
      <td>Hip</td>
      <td>CompShg</td>
      <td>Wd Sdng</td>
      <td>Wd Sdng</td>
      <td>BrkFace</td>
      <td>108.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>TA</td>
      <td>TA</td>
      <td>No</td>
      <td>ALQ</td>
      <td>923.0</td>
      <td>Unf</td>
      <td>0.0</td>
      <td>406.0</td>
      <td>1329.0</td>
      <td>GasA</td>
      <td>TA</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>1329</td>
      <td>0</td>
      <td>0</td>
      <td>1329</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>Gd</td>
      <td>6</td>
      <td>Typ</td>
      <td>0</td>
      <td>NaN</td>
      <td>Attchd</td>
      <td>1958.0</td>
      <td>Unf</td>
      <td>1.0</td>
      <td>312.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>393</td>
      <td>36</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Gar2</td>
      <td>12500</td>
      <td>6</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>172000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>526353030</td>
      <td>20</td>
      <td>RL</td>
      <td>93.0</td>
      <td>11160</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Corner</td>
      <td>Gtl</td>
      <td>NAmes</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>7</td>
      <td>5</td>
      <td>1968</td>
      <td>1968</td>
      <td>Hip</td>
      <td>CompShg</td>
      <td>BrkFace</td>
      <td>BrkFace</td>
      <td>None</td>
      <td>0.0</td>
      <td>Gd</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>TA</td>
      <td>TA</td>
      <td>No</td>
      <td>ALQ</td>
      <td>1065.0</td>
      <td>Unf</td>
      <td>0.0</td>
      <td>1045.0</td>
      <td>2110.0</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>2110</td>
      <td>0</td>
      <td>0</td>
      <td>2110</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>Ex</td>
      <td>8</td>
      <td>Typ</td>
      <td>2</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>1968.0</td>
      <td>Fin</td>
      <td>2.0</td>
      <td>522.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>4</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>244000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>527105010</td>
      <td>60</td>
      <td>RL</td>
      <td>74.0</td>
      <td>13830</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>Gilbert</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>5</td>
      <td>5</td>
      <td>1997</td>
      <td>1998</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>None</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>PConc</td>
      <td>Gd</td>
      <td>TA</td>
      <td>No</td>
      <td>GLQ</td>
      <td>791.0</td>
      <td>Unf</td>
      <td>0.0</td>
      <td>137.0</td>
      <td>928.0</td>
      <td>GasA</td>
      <td>Gd</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>928</td>
      <td>701</td>
      <td>0</td>
      <td>1629</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>TA</td>
      <td>6</td>
      <td>Typ</td>
      <td>1</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>1997.0</td>
      <td>Fin</td>
      <td>2.0</td>
      <td>482.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>212</td>
      <td>34</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>MnPrv</td>
      <td>NaN</td>
      <td>0</td>
      <td>3</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>189900</td>
    </tr>
  </tbody>
</table>
</div>



### Data Cleaning and Features Engineering

---

This dataset has a total of 82 columns and 2930 rows. Since we'll be using the linear regression model, we can only use numerical values in our model. One of the most important aspects of machine learning is knowing the features. Here are a couple things we can do to clean up the data:

+ The 'Order' and 'PID' columns are not useful for machine learning as they are simply identification numbers.  

+ It doesn't make much sense to use 'Year built' and 'Year Remod/Add' in our model. We should generate a new column to determine how old the house is since the last remodelling.  

+ We want to drop columns with too many missing values, let's start with 5% for now.  

+ We don't want to leak sales information to our model. Sales information will not be available to us when we actually use the model to estimate the price of a house.


```python
#Create a new feature, 'years_to_sell'.
data['years_to_sell'] = data['Yr Sold'] - data['Year Remod/Add'] 
data = data[data['years_to_sell'] >= 0]

#Remove features that are not useful for machine learning.
data = data.drop(['Order', 'PID'], axis=1)

#Remove features that leak sales data.
data = data.drop(['Mo Sold', 'Yr Sold', 'Sale Type', 'Sale Condition'], axis=1)

#Drop columns with more than 5% missing values
is_null_counts = data.isnull().sum()
features_col = is_null_counts[is_null_counts < 2930*0.05].index


data = data[features_col]
data.head()
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
      <th>MS SubClass</th>
      <th>MS Zoning</th>
      <th>Lot Area</th>
      <th>Street</th>
      <th>Lot Shape</th>
      <th>Land Contour</th>
      <th>Utilities</th>
      <th>Lot Config</th>
      <th>Land Slope</th>
      <th>Neighborhood</th>
      <th>Condition 1</th>
      <th>Condition 2</th>
      <th>Bldg Type</th>
      <th>House Style</th>
      <th>Overall Qual</th>
      <th>Overall Cond</th>
      <th>Year Built</th>
      <th>Year Remod/Add</th>
      <th>Roof Style</th>
      <th>Roof Matl</th>
      <th>Exterior 1st</th>
      <th>Exterior 2nd</th>
      <th>Mas Vnr Type</th>
      <th>Mas Vnr Area</th>
      <th>Exter Qual</th>
      <th>Exter Cond</th>
      <th>Foundation</th>
      <th>Bsmt Qual</th>
      <th>Bsmt Cond</th>
      <th>Bsmt Exposure</th>
      <th>BsmtFin Type 1</th>
      <th>BsmtFin SF 1</th>
      <th>BsmtFin Type 2</th>
      <th>BsmtFin SF 2</th>
      <th>Bsmt Unf SF</th>
      <th>Total Bsmt SF</th>
      <th>Heating</th>
      <th>Heating QC</th>
      <th>Central Air</th>
      <th>Electrical</th>
      <th>1st Flr SF</th>
      <th>2nd Flr SF</th>
      <th>Low Qual Fin SF</th>
      <th>Gr Liv Area</th>
      <th>Bsmt Full Bath</th>
      <th>Bsmt Half Bath</th>
      <th>Full Bath</th>
      <th>Half Bath</th>
      <th>Bedroom AbvGr</th>
      <th>Kitchen AbvGr</th>
      <th>Kitchen Qual</th>
      <th>TotRms AbvGrd</th>
      <th>Functional</th>
      <th>Fireplaces</th>
      <th>Garage Cars</th>
      <th>Garage Area</th>
      <th>Paved Drive</th>
      <th>Wood Deck SF</th>
      <th>Open Porch SF</th>
      <th>Enclosed Porch</th>
      <th>3Ssn Porch</th>
      <th>Screen Porch</th>
      <th>Pool Area</th>
      <th>Misc Val</th>
      <th>SalePrice</th>
      <th>years_to_sell</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>20</td>
      <td>RL</td>
      <td>31770</td>
      <td>Pave</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Corner</td>
      <td>Gtl</td>
      <td>NAmes</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>6</td>
      <td>5</td>
      <td>1960</td>
      <td>1960</td>
      <td>Hip</td>
      <td>CompShg</td>
      <td>BrkFace</td>
      <td>Plywood</td>
      <td>Stone</td>
      <td>112.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>TA</td>
      <td>Gd</td>
      <td>Gd</td>
      <td>BLQ</td>
      <td>639.0</td>
      <td>Unf</td>
      <td>0.0</td>
      <td>441.0</td>
      <td>1080.0</td>
      <td>GasA</td>
      <td>Fa</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>1656</td>
      <td>0</td>
      <td>0</td>
      <td>1656</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>TA</td>
      <td>7</td>
      <td>Typ</td>
      <td>2</td>
      <td>2.0</td>
      <td>528.0</td>
      <td>P</td>
      <td>210</td>
      <td>62</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>215000</td>
      <td>50</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20</td>
      <td>RH</td>
      <td>11622</td>
      <td>Pave</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>NAmes</td>
      <td>Feedr</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>5</td>
      <td>6</td>
      <td>1961</td>
      <td>1961</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>None</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>TA</td>
      <td>TA</td>
      <td>No</td>
      <td>Rec</td>
      <td>468.0</td>
      <td>LwQ</td>
      <td>144.0</td>
      <td>270.0</td>
      <td>882.0</td>
      <td>GasA</td>
      <td>TA</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>896</td>
      <td>0</td>
      <td>0</td>
      <td>896</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>TA</td>
      <td>5</td>
      <td>Typ</td>
      <td>0</td>
      <td>1.0</td>
      <td>730.0</td>
      <td>Y</td>
      <td>140</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>120</td>
      <td>0</td>
      <td>0</td>
      <td>105000</td>
      <td>49</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20</td>
      <td>RL</td>
      <td>14267</td>
      <td>Pave</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Corner</td>
      <td>Gtl</td>
      <td>NAmes</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>6</td>
      <td>6</td>
      <td>1958</td>
      <td>1958</td>
      <td>Hip</td>
      <td>CompShg</td>
      <td>Wd Sdng</td>
      <td>Wd Sdng</td>
      <td>BrkFace</td>
      <td>108.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>TA</td>
      <td>TA</td>
      <td>No</td>
      <td>ALQ</td>
      <td>923.0</td>
      <td>Unf</td>
      <td>0.0</td>
      <td>406.0</td>
      <td>1329.0</td>
      <td>GasA</td>
      <td>TA</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>1329</td>
      <td>0</td>
      <td>0</td>
      <td>1329</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>Gd</td>
      <td>6</td>
      <td>Typ</td>
      <td>0</td>
      <td>1.0</td>
      <td>312.0</td>
      <td>Y</td>
      <td>393</td>
      <td>36</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>12500</td>
      <td>172000</td>
      <td>52</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20</td>
      <td>RL</td>
      <td>11160</td>
      <td>Pave</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Corner</td>
      <td>Gtl</td>
      <td>NAmes</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>7</td>
      <td>5</td>
      <td>1968</td>
      <td>1968</td>
      <td>Hip</td>
      <td>CompShg</td>
      <td>BrkFace</td>
      <td>BrkFace</td>
      <td>None</td>
      <td>0.0</td>
      <td>Gd</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>TA</td>
      <td>TA</td>
      <td>No</td>
      <td>ALQ</td>
      <td>1065.0</td>
      <td>Unf</td>
      <td>0.0</td>
      <td>1045.0</td>
      <td>2110.0</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>2110</td>
      <td>0</td>
      <td>0</td>
      <td>2110</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>Ex</td>
      <td>8</td>
      <td>Typ</td>
      <td>2</td>
      <td>2.0</td>
      <td>522.0</td>
      <td>Y</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>244000</td>
      <td>42</td>
    </tr>
    <tr>
      <th>4</th>
      <td>60</td>
      <td>RL</td>
      <td>13830</td>
      <td>Pave</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>Gilbert</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>5</td>
      <td>5</td>
      <td>1997</td>
      <td>1998</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>None</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>PConc</td>
      <td>Gd</td>
      <td>TA</td>
      <td>No</td>
      <td>GLQ</td>
      <td>791.0</td>
      <td>Unf</td>
      <td>0.0</td>
      <td>137.0</td>
      <td>928.0</td>
      <td>GasA</td>
      <td>Gd</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>928</td>
      <td>701</td>
      <td>0</td>
      <td>1629</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>TA</td>
      <td>6</td>
      <td>Typ</td>
      <td>1</td>
      <td>2.0</td>
      <td>482.0</td>
      <td>Y</td>
      <td>212</td>
      <td>34</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>189900</td>
      <td>12</td>
    </tr>
  </tbody>
</table>
</div>



Since we are dealing with a dataset with a large number a columns, it is a good idea to split the data up into two dataframes. We'll first work with the 'float' and 'int' columns. Then we'll set 'object' columns to a new dataframe. Once both dataframes contain only numerical values, we can combine them again and use the features for our linear regression model.

There are qutie a bit of NA values in the numerical columns, so we'll fill them up with the mode. Some of the columns are categorical, so it wouldn't make sense to use median or mean for this.


```python
numerical_cols = data.dtypes[data.dtypes != 'object'].index
numerical_data = data[numerical_cols]

numerical_data = numerical_data.fillna(data.mode().iloc[0])
numerical_data.isnull().sum().sort_values(ascending = False)
```




    years_to_sell      0
    BsmtFin SF 2       0
    Gr Liv Area        0
    Low Qual Fin SF    0
    2nd Flr SF         0
    1st Flr SF         0
    Total Bsmt SF      0
    Bsmt Unf SF        0
    BsmtFin SF 1       0
    SalePrice          0
    Mas Vnr Area       0
    Year Remod/Add     0
    Year Built         0
    Overall Cond       0
    Overall Qual       0
    Lot Area           0
    Bsmt Full Bath     0
    Bsmt Half Bath     0
    Full Bath          0
    Half Bath          0
    Bedroom AbvGr      0
    Kitchen AbvGr      0
    TotRms AbvGrd      0
    Fireplaces         0
    Garage Cars        0
    Garage Area        0
    Wood Deck SF       0
    Open Porch SF      0
    Enclosed Porch     0
    3Ssn Porch         0
    Screen Porch       0
    Pool Area          0
    Misc Val           0
    MS SubClass        0
    dtype: int64



Next, let's check the correlations of all the numerical columns with respect to 'SalePrice'


```python
num_corr = numerical_data.corr()['SalePrice'].abs().sort_values(ascending = False)
num_corr
```




    SalePrice          1.000000
    Overall Qual       0.801206
    Gr Liv Area        0.717596
    Garage Cars        0.648361
    Total Bsmt SF      0.644012
    Garage Area        0.641425
    1st Flr SF         0.635185
    Year Built         0.558490
    Full Bath          0.546118
    years_to_sell      0.534985
    Year Remod/Add     0.533007
    Mas Vnr Area       0.506983
    TotRms AbvGrd      0.498574
    Fireplaces         0.474831
    BsmtFin SF 1       0.439284
    Wood Deck SF       0.328183
    Open Porch SF      0.316262
    Half Bath          0.284871
    Bsmt Full Bath     0.276258
    2nd Flr SF         0.269601
    Lot Area           0.267520
    Bsmt Unf SF        0.182751
    Bedroom AbvGr      0.143916
    Enclosed Porch     0.128685
    Kitchen AbvGr      0.119760
    Screen Porch       0.112280
    Overall Cond       0.101540
    MS SubClass        0.085128
    Pool Area          0.068438
    Low Qual Fin SF    0.037629
    Bsmt Half Bath     0.035875
    3Ssn Porch         0.032268
    Misc Val           0.019273
    BsmtFin SF 2       0.006127
    Name: SalePrice, dtype: float64



We can drop values with less than 0.4 correlation for now. Later, we'll make this value an adjustable parameter in a function.


```python
num_corr = num_corr[num_corr > 0.4]
high_corr_cols = num_corr.index

hi_corr_numerical_data = numerical_data[high_corr_cols]
```

For the 'object' or text columns, we'll drop any column with more than 1 missing value.


```python
text_cols = data.dtypes[data.dtypes == 'object'].index
text_data = data[text_cols]

text_null_counts = text_data.isnull().sum()
text_not_null_cols = text_null_counts[text_null_counts < 1].index

text_data = text_data[text_not_null_cols]
```

From the documatation we want to convert any columns that are nominal into categories. 'MS subclass' is a numerical column but it should be categorical.

For the text columns, we'll take the list of nominal columns from the documentation and use a for loop to search for matches.


```python
nominal_cols = ['MS Zoning', 'Street', 'Alley', 'Land Contour', 'Lot Config', 'Neighborhood', 'Condition 1', 'Condition 2', 'Bldg Type', 'House Style', 'Overall Qual', 'Roof Style', 'Roof Mat1', 'Exterior 1st',  'Exterior 2nd', 'Mas Vnr Type', 'Foundation', 'Heating', 'Central Air'] 
nominal_num_col = ['MS SubClass']
```


```python
#Finds nominal columns in text_data
nominal_text_col = []
for col in nominal_cols:
    if col in text_data.columns:
        nominal_text_col.append(col)
nominal_text_col
```




    ['MS Zoning',
     'Street',
     'Land Contour',
     'Lot Config',
     'Neighborhood',
     'Condition 1',
     'Condition 2',
     'Bldg Type',
     'House Style',
     'Roof Style',
     'Exterior 1st',
     'Exterior 2nd',
     'Foundation',
     'Heating',
     'Central Air']



Simply use boolean filtering to keep the relevant columns in our text dataframe.


```python
text_data = text_data[nominal_text_col]
```


```python
for col in nominal_text_col:
    print(col)
    print(text_data[col].value_counts())
    print("-"*10)
```

    MS Zoning
    RL         2270
    RM          462
    FV          139
    RH           27
    C (all)      25
    A (agr)       2
    I (all)       2
    Name: MS Zoning, dtype: int64
    ----------
    Street
    Pave    2915
    Grvl      12
    Name: Street, dtype: int64
    ----------
    Land Contour
    Lvl    2632
    HLS     120
    Bnk     115
    Low      60
    Name: Land Contour, dtype: int64
    ----------
    Lot Config
    Inside     2138
    Corner      510
    CulDSac     180
    FR2          85
    FR3          14
    Name: Lot Config, dtype: int64
    ----------
    Neighborhood
    NAmes      443
    CollgCr    267
    OldTown    239
    Edwards    192
    Somerst    182
    NridgHt    165
    Gilbert    165
    Sawyer     151
    NWAmes     131
    SawyerW    125
    Mitchel    114
    BrkSide    108
    Crawfor    103
    IDOTRR      93
    Timber      72
    NoRidge     71
    StoneBr     51
    SWISU       48
    ClearCr     44
    MeadowV     37
    BrDale      30
    Blmngtn     28
    Veenker     24
    NPkVill     23
    Blueste     10
    Greens       8
    GrnHill      2
    Landmrk      1
    Name: Neighborhood, dtype: int64
    ----------
    Condition 1
    Norm      2520
    Feedr      164
    Artery      92
    RRAn        50
    PosN        38
    RRAe        28
    PosA        20
    RRNn         9
    RRNe         6
    Name: Condition 1, dtype: int64
    ----------
    Condition 2
    Norm      2898
    Feedr       13
    Artery       5
    PosA         4
    PosN         3
    RRNn         2
    RRAn         1
    RRAe         1
    Name: Condition 2, dtype: int64
    ----------
    Bldg Type
    1Fam      2422
    TwnhsE     233
    Duplex     109
    Twnhs      101
    2fmCon      62
    Name: Bldg Type, dtype: int64
    ----------
    House Style
    1Story    1480
    2Story     871
    1.5Fin     314
    SLvl       128
    SFoyer      83
    2.5Unf      24
    1.5Unf      19
    2.5Fin       8
    Name: House Style, dtype: int64
    ----------
    Roof Style
    Gable      2320
    Hip         549
    Gambrel      22
    Flat         20
    Mansard      11
    Shed          5
    Name: Roof Style, dtype: int64
    ----------
    Exterior 1st
    VinylSd    1025
    MetalSd     450
    HdBoard     442
    Wd Sdng     420
    Plywood     221
    CemntBd     124
    BrkFace      88
    WdShing      56
    AsbShng      44
    Stucco       43
    BrkComm       6
    Stone         2
    CBlock        2
    AsphShn       2
    ImStucc       1
    PreCast       1
    Name: Exterior 1st, dtype: int64
    ----------
    Exterior 2nd
    VinylSd    1014
    MetalSd     447
    HdBoard     406
    Wd Sdng     397
    Plywood     274
    CmentBd     124
    Wd Shng      81
    Stucco       47
    BrkFace      47
    AsbShng      38
    Brk Cmn      22
    ImStucc      15
    Stone         6
    AsphShn       4
    CBlock        3
    PreCast       1
    Other         1
    Name: Exterior 2nd, dtype: int64
    ----------
    Foundation
    PConc     1307
    CBlock    1244
    BrkTil     311
    Slab        49
    Stone       11
    Wood         5
    Name: Foundation, dtype: int64
    ----------
    Heating
    GasA     2882
    GasW       27
    Grav        9
    Wall        6
    OthW        2
    Floor       1
    Name: Heating, dtype: int64
    ----------
    Central Air
    Y    2731
    N     196
    Name: Central Air, dtype: int64
    ----------
    

Columns with too many categories can cause overfitting. We'll remove any columns with more than 10 categories. We'll write a function later to adjust this as a parameter in our feature selection.


```python
nominal_text_col_unique = []
for col in nominal_text_col:
    if len(text_data[col].value_counts()) <= 10:
        nominal_text_col_unique.append(col)
               
text_data = text_data[nominal_text_col_unique]
```

Finally, we can use the pd.get_dummies function to create dummy columns for all the categorical columns. 


```python
#Create dummy columns for nominal text columns, then create a dataframe.
for col in text_data.columns:
    text_data[col] = text_data[col].astype('category')   
categorical_text_data = pd.get_dummies(text_data)    
categorical_text_data.head()
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
      <th>MS Zoning_A (agr)</th>
      <th>MS Zoning_C (all)</th>
      <th>MS Zoning_FV</th>
      <th>MS Zoning_I (all)</th>
      <th>MS Zoning_RH</th>
      <th>MS Zoning_RL</th>
      <th>MS Zoning_RM</th>
      <th>Street_Grvl</th>
      <th>Street_Pave</th>
      <th>Land Contour_Bnk</th>
      <th>Land Contour_HLS</th>
      <th>Land Contour_Low</th>
      <th>Land Contour_Lvl</th>
      <th>Lot Config_Corner</th>
      <th>Lot Config_CulDSac</th>
      <th>Lot Config_FR2</th>
      <th>Lot Config_FR3</th>
      <th>Lot Config_Inside</th>
      <th>Condition 1_Artery</th>
      <th>Condition 1_Feedr</th>
      <th>Condition 1_Norm</th>
      <th>Condition 1_PosA</th>
      <th>Condition 1_PosN</th>
      <th>Condition 1_RRAe</th>
      <th>Condition 1_RRAn</th>
      <th>Condition 1_RRNe</th>
      <th>Condition 1_RRNn</th>
      <th>Condition 2_Artery</th>
      <th>Condition 2_Feedr</th>
      <th>Condition 2_Norm</th>
      <th>Condition 2_PosA</th>
      <th>Condition 2_PosN</th>
      <th>Condition 2_RRAe</th>
      <th>Condition 2_RRAn</th>
      <th>Condition 2_RRNn</th>
      <th>Bldg Type_1Fam</th>
      <th>Bldg Type_2fmCon</th>
      <th>Bldg Type_Duplex</th>
      <th>Bldg Type_Twnhs</th>
      <th>Bldg Type_TwnhsE</th>
      <th>House Style_1.5Fin</th>
      <th>House Style_1.5Unf</th>
      <th>House Style_1Story</th>
      <th>House Style_2.5Fin</th>
      <th>House Style_2.5Unf</th>
      <th>House Style_2Story</th>
      <th>House Style_SFoyer</th>
      <th>House Style_SLvl</th>
      <th>Roof Style_Flat</th>
      <th>Roof Style_Gable</th>
      <th>Roof Style_Gambrel</th>
      <th>Roof Style_Hip</th>
      <th>Roof Style_Mansard</th>
      <th>Roof Style_Shed</th>
      <th>Foundation_BrkTil</th>
      <th>Foundation_CBlock</th>
      <th>Foundation_PConc</th>
      <th>Foundation_Slab</th>
      <th>Foundation_Stone</th>
      <th>Foundation_Wood</th>
      <th>Heating_Floor</th>
      <th>Heating_GasA</th>
      <th>Heating_GasW</th>
      <th>Heating_Grav</th>
      <th>Heating_OthW</th>
      <th>Heating_Wall</th>
      <th>Central Air_N</th>
      <th>Central Air_Y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Create dummy columns for nominal numerical columns, then create a dataframe.
for col in numerical_data.columns:
    if col in nominal_num_col:
        numerical_data[col] = numerical_data[col].astype('category')  
              
categorical_numerical_data = pd.get_dummies(numerical_data.select_dtypes(include=['category'])) 
```

Using the pd.concat() function, we can combine the two categorical columns together.


```python
categorical_data = pd.concat([categorical_text_data, categorical_numerical_data], axis=1)
```

We end up with one numerical dataframe, and one categorical dataframe. We can then combine them into one dataframe for machine learning.


```python
hi_corr_numerical_data.head()
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
      <th>SalePrice</th>
      <th>Overall Qual</th>
      <th>Gr Liv Area</th>
      <th>Garage Cars</th>
      <th>Total Bsmt SF</th>
      <th>Garage Area</th>
      <th>1st Flr SF</th>
      <th>Year Built</th>
      <th>Full Bath</th>
      <th>years_to_sell</th>
      <th>Year Remod/Add</th>
      <th>Mas Vnr Area</th>
      <th>TotRms AbvGrd</th>
      <th>Fireplaces</th>
      <th>BsmtFin SF 1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>215000</td>
      <td>6</td>
      <td>1656</td>
      <td>2.0</td>
      <td>1080.0</td>
      <td>528.0</td>
      <td>1656</td>
      <td>1960</td>
      <td>1</td>
      <td>50</td>
      <td>1960</td>
      <td>112.0</td>
      <td>7</td>
      <td>2</td>
      <td>639.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>105000</td>
      <td>5</td>
      <td>896</td>
      <td>1.0</td>
      <td>882.0</td>
      <td>730.0</td>
      <td>896</td>
      <td>1961</td>
      <td>1</td>
      <td>49</td>
      <td>1961</td>
      <td>0.0</td>
      <td>5</td>
      <td>0</td>
      <td>468.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>172000</td>
      <td>6</td>
      <td>1329</td>
      <td>1.0</td>
      <td>1329.0</td>
      <td>312.0</td>
      <td>1329</td>
      <td>1958</td>
      <td>1</td>
      <td>52</td>
      <td>1958</td>
      <td>108.0</td>
      <td>6</td>
      <td>0</td>
      <td>923.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>244000</td>
      <td>7</td>
      <td>2110</td>
      <td>2.0</td>
      <td>2110.0</td>
      <td>522.0</td>
      <td>2110</td>
      <td>1968</td>
      <td>2</td>
      <td>42</td>
      <td>1968</td>
      <td>0.0</td>
      <td>8</td>
      <td>2</td>
      <td>1065.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>189900</td>
      <td>5</td>
      <td>1629</td>
      <td>2.0</td>
      <td>928.0</td>
      <td>482.0</td>
      <td>928</td>
      <td>1997</td>
      <td>2</td>
      <td>12</td>
      <td>1998</td>
      <td>0.0</td>
      <td>6</td>
      <td>1</td>
      <td>791.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
categorical_data.head()
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
      <th>MS Zoning_A (agr)</th>
      <th>MS Zoning_C (all)</th>
      <th>MS Zoning_FV</th>
      <th>MS Zoning_I (all)</th>
      <th>MS Zoning_RH</th>
      <th>MS Zoning_RL</th>
      <th>MS Zoning_RM</th>
      <th>Street_Grvl</th>
      <th>Street_Pave</th>
      <th>Land Contour_Bnk</th>
      <th>Land Contour_HLS</th>
      <th>Land Contour_Low</th>
      <th>Land Contour_Lvl</th>
      <th>Lot Config_Corner</th>
      <th>Lot Config_CulDSac</th>
      <th>Lot Config_FR2</th>
      <th>Lot Config_FR3</th>
      <th>Lot Config_Inside</th>
      <th>Condition 1_Artery</th>
      <th>Condition 1_Feedr</th>
      <th>Condition 1_Norm</th>
      <th>Condition 1_PosA</th>
      <th>Condition 1_PosN</th>
      <th>Condition 1_RRAe</th>
      <th>Condition 1_RRAn</th>
      <th>Condition 1_RRNe</th>
      <th>Condition 1_RRNn</th>
      <th>Condition 2_Artery</th>
      <th>Condition 2_Feedr</th>
      <th>Condition 2_Norm</th>
      <th>Condition 2_PosA</th>
      <th>Condition 2_PosN</th>
      <th>Condition 2_RRAe</th>
      <th>Condition 2_RRAn</th>
      <th>Condition 2_RRNn</th>
      <th>Bldg Type_1Fam</th>
      <th>Bldg Type_2fmCon</th>
      <th>Bldg Type_Duplex</th>
      <th>Bldg Type_Twnhs</th>
      <th>Bldg Type_TwnhsE</th>
      <th>House Style_1.5Fin</th>
      <th>House Style_1.5Unf</th>
      <th>House Style_1Story</th>
      <th>House Style_2.5Fin</th>
      <th>House Style_2.5Unf</th>
      <th>House Style_2Story</th>
      <th>House Style_SFoyer</th>
      <th>House Style_SLvl</th>
      <th>Roof Style_Flat</th>
      <th>Roof Style_Gable</th>
      <th>Roof Style_Gambrel</th>
      <th>Roof Style_Hip</th>
      <th>Roof Style_Mansard</th>
      <th>Roof Style_Shed</th>
      <th>Foundation_BrkTil</th>
      <th>Foundation_CBlock</th>
      <th>Foundation_PConc</th>
      <th>Foundation_Slab</th>
      <th>Foundation_Stone</th>
      <th>Foundation_Wood</th>
      <th>Heating_Floor</th>
      <th>Heating_GasA</th>
      <th>Heating_GasW</th>
      <th>Heating_Grav</th>
      <th>Heating_OthW</th>
      <th>Heating_Wall</th>
      <th>Central Air_N</th>
      <th>Central Air_Y</th>
      <th>MS SubClass_20</th>
      <th>MS SubClass_30</th>
      <th>MS SubClass_40</th>
      <th>MS SubClass_45</th>
      <th>MS SubClass_50</th>
      <th>MS SubClass_60</th>
      <th>MS SubClass_70</th>
      <th>MS SubClass_75</th>
      <th>MS SubClass_80</th>
      <th>MS SubClass_85</th>
      <th>MS SubClass_90</th>
      <th>MS SubClass_120</th>
      <th>MS SubClass_150</th>
      <th>MS SubClass_160</th>
      <th>MS SubClass_180</th>
      <th>MS SubClass_190</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
final_data = pd.concat([hi_corr_numerical_data, categorical_data], axis=1)
```

### Creating Functions with Adjustable Parameters

---

When we did our data cleaning, we decided to remove columns that had more than 5% missing values. We can incorporate our this into a function as an adjustable parameter. In addition, this function will perform all the data cleaning operations I've explained above.


```python
def transform_features(data, percent_missing=0.05):
    
    #Adding relevant features:
    data['years_since_remod'] = data['Year Built'] - data['Year Remod/Add']
    data['years_to_sell'] = data['Yr Sold'] - data['Year Built']
    data = data[data['years_since_remod'] >= 0]
    data = data[data['years_to_sell'] >= 0]
    
    #Remove columns not useful for machine learning
    data = data.drop(['Order', 'PID', 'Year Built', 'Year Remod/Add'], axis=1)
    
    #Remove columns that leaks sale data
    data = data.drop(['Mo Sold', 'Yr Sold', 'Sale Type', 'Sale Condition'], axis=1)
    
    #Drop columns with too many missing values defined by the function
    is_null_counts = data.isnull().sum()
    low_NaN_cols = is_null_counts[is_null_counts < len(data)*percent_missing].index
    
    transformed_data = data[low_NaN_cols]    
    return transformed_data
```

For the feature engineering and selection step, we chose columns that had more than 0.4 correlation with 'SalePrice' and removed any columns with more than 10 categories.

Once again, I've combined all the work we've done previously into a function with adjustable parameters.


```python
def select_features(data, corr_threshold=0.4, unique_threshold=10):  
    
    #Fill missing numerical columns with the mode.
    numerical_cols = data.dtypes[data.dtypes != 'object'].index
    numerical_data = data[numerical_cols]

    numerical_data = numerical_data.fillna(data.mode().iloc[0])
    numerical_data.isnull().sum().sort_values(ascending = False)

    #Drop text columns with more than 1 missing value.
    text_cols = data.dtypes[data.dtypes == 'object'].index
    text_data = data[text_cols]

    text_null_counts = text_data.isnull().sum()
    text_not_null_cols = text_null_counts[text_null_counts < 1].index

    text_data = text_data[text_not_null_cols]

    num_corr = numerical_data.corr()['SalePrice'].abs().sort_values(ascending = False)

    num_corr = num_corr[num_corr > corr_threshold]
    high_corr_cols = num_corr.index

    #Apply the correlation threshold parameter
    hi_corr_numerical_data = numerical_data[high_corr_cols]
    

    #Nominal columns from the documentation
    nominal_cols = ['MS Zoning', 'Street', 'Alley', 'Land Contour', 'Lot Config', 'Neighborhood', 'Condition 1', 'Condition 2', 'Bldg Type', 'House Style', 'Overall Qual', 'Roof Style', 'Roof Mat1', 'Exterior 1st',  'Exterior 2nd', 'Mas Vnr Type', 'Foundation', 'Heating', 'Central Air'] 
    nominal_num_col = ['MS SubClass']

    #Finds nominal columns in text_data
    nominal_text_col = []
    for col in nominal_cols:
        if col in text_data.columns:
            nominal_text_col.append(col)
    nominal_text_col

    text_data = text_data[nominal_text_col]

    nominal_text_col_unique = []
    for col in nominal_text_col:
        if len(text_data[col].value_counts()) <= unique_threshold:
            nominal_text_col_unique.append(col)
        
        
    text_data = text_data[nominal_text_col_unique]
    text_data.head()

    #Set all these columns to categorical
    for col in text_data.columns:
        text_data[col] = text_data[col].astype('category')   
    categorical_text_data = pd.get_dummies(text_data)    

    #Change any nominal numerical columns to categorical, then returns a dataframe
    for col in numerical_data.columns:
        if col in nominal_num_col:
            numerical_data[col] = numerical_data[col].astype('category')  
           
    
    categorical_numerical_data = pd.get_dummies(numerical_data.select_dtypes(include=['category'])) 
    final_data = pd.concat([hi_corr_numerical_data, categorical_text_data, categorical_numerical_data], axis=1)

    return final_data
```

### Applying Machine Learning

---

Now we are ready to apply machine learning, we'll use the linear regression model from scikit-learn. Linear regression should work well here since our target column 'SalePrice' is a continuous value. We'll evaluate this model with RMSE as an error metric.


```python
def train_and_test(data):

    train = data[0:1460]
    test = data[1460:]
    features = data.columns.drop(['SalePrice'])
    
    #train
    lr = LinearRegression()
    lr.fit(train[features], train['SalePrice'])
    #predict
    
    predictions = lr.predict(test[features])
    rmse = mean_squared_error(test['SalePrice'], predictions)**0.5
    return rmse
```


```python
data = pd.read_csv("AmesHousing.tsv", delimiter='\t')

transformed_data = transform_features(data, percent_missing=0.05)
final_data = select_features(transformed_data, 0.4, 10)
result = train_and_test(final_data)
result
```




    28749.561761556044



We've selected the first 1460 rows as the training set, and the remaining data as the testing set. This is not really a good way to evaluate a model's performance because the error will change as soon as we shuffle the data. 

We can use KFold cross validation to split the data in K number of folds. Using the KFold function from scikit learn, we can get the indices for the testing and training sets.


```python
from sklearn.model_selection import KFold

def train_and_test2(data, k=2):  
    rf = LinearRegression()
    if k == 0:
        train = data[0:1460]
        test = data[1460:]
        features = data.columns.drop(['SalePrice'])
    
        #train
        rf.fit(train[features], train['SalePrice'])
        
        #predict    
        predictions = rf.predict(test[features])
        rmse = mean_squared_error(test['SalePrice'], predictions)**0.5
        return rmse
    
    elif k == 1:
        train = data[:1460]
        test = data[1460:]
        features = data.columns.drop(['SalePrice'])
        
        rf.fit(train[features], train["SalePrice"])
        predictions_one = rf.predict(test[features])        
        
        mse_one = mean_squared_error(test["SalePrice"], predictions_one)
        rmse_one = np.sqrt(mse_one)
        
        rf.fit(test[features], test["SalePrice"])
        predictions_two = rf.predict(train[features])        
       
        mse_two = mean_squared_error(train["SalePrice"], predictions_two)
        rmse_two = np.sqrt(mse_two)
        return np.mean([rmse_one, rmse_two])   
    
    else:
        kf = KFold(n_splits=k, shuffle=True, random_state = 2)
        rmse_list = []
        for train_index, test_index in kf.split(data):
            train = data.iloc[train_index]
            test = data.iloc[test_index]
            features = data.columns.drop(['SalePrice'])
    
            #train
            rf.fit(train[features], train['SalePrice'])
        
            #predict    
            predictions = rf.predict(test[features])
        
            rmse = mean_squared_error(test['SalePrice'], predictions)**0.5
            rmse_list.append(rmse)
        return np.mean(rmse_list)
```


```python
data = pd.read_csv("AmesHousing.tsv", delimiter='\t')

transformed_data = transform_features(data, percent_missing=0.05)
final_data = select_features(transformed_data, 0.4, 10)

results = []
for i in range(100):
    result = train_and_test2(final_data, k=i)
    results.append(result)
    
x = [i for i in range(100)]
y = results 
plt.plot(x, y)
plt.xlabel('Kfolds')
plt.ylabel('RMSE')

print(results[99])
```

    29830.6836474
    


    
![png](/posts_images/2018-02-13-DataQuestGuidedProjectPredictingHouseSalePrices/output_39_1.png)
    


Our error is actually the lowest, when k = 0. This is acutally not very useful because it means the model is only useful for the indices we've picked out. Without validation there is no way to be sure that the model works well for any set of data.

This is when cross validation is useful for evaluating model performance. We can see the average RMSE goes down as we increase the number of folds. This makes sense as the RMSE shown on the graph above is an average of the cross validation tests. A larger K means we have less bias towards overestimating the model's true error. As a trade off, this requires a lot more computation time. 

---

#### Learning Summary

Concepts explored: pandas, data cleaning, features engineering, linear regression, hyperparameter tuning, RMSE, KFold validation

Functions and methods used: .dtypes, .value_counts(), .drop, .isnull(), sum(), .fillna(), .sort_values(), . corr(), .index, .append(), .get_dummies(), .astype(), predict(), .fit(), KFold(), mean_squared_error()


The files used for this project can be found in my [GitHub repository](https://github.com/sengkchu/Dataquest-Guided-Projects-Solutions/tree/master/Guided%20Project_%20Predicting%20House%20Sale%20Prices).

