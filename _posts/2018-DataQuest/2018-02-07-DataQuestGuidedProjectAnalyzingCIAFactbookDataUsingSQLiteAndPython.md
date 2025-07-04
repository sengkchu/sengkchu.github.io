---
title: Dataquest Guided Project - Analyzing CIA Factbook Data Using SQLite And Python
date: 2018-02-07 00:00 +/-0000
categories: [DataQuest]
tags: [dataquest]
image:
  path: /posts_images/2018-02-07-DataQuestGuidedProjectAnalyzingCIAFactbookDataUsingSQLiteAndPython/cover.PNG
---


In this project we'll be working with SQL in combination with Python. Specifically we'll use sqlite3. We will analyze the database file "factbook.db" which is the CIA World Factbook. We will write queries to look at the data and see if we can draw any interesting insights.


```python
#import sql3, pandas and connect to the databse.
import sqlite3
import pandas as pd
conn = sqlite3.connect("factbook.db")

#activates the cursor
cursor = conn.cursor()

#the SQL query to look at the tables in the databse
q1 = "SELECT * FROM sqlite_master WHERE type='table';"

#execute the query and read it in pandas, this returns a table in pandas form
database_info = pd.read_sql_query(q1, conn)
database_info
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
      <th>type</th>
      <th>name</th>
      <th>tbl_name</th>
      <th>rootpage</th>
      <th>sql</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>table</td>
      <td>facts</td>
      <td>facts</td>
      <td>2</td>
      <td>CREATE TABLE "facts" ("id" INTEGER PRIMARY KEY...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>table</td>
      <td>sqlite_sequence</td>
      <td>sqlite_sequence</td>
      <td>3</td>
      <td>CREATE TABLE sqlite_sequence(name,seq)</td>
    </tr>
  </tbody>
</table>
</div>



Let's begin exploring the data, we can use pd.read_sql_query to see what the first table looks like


```python
q2 = "SELECT * FROM facts"

data = pd.read_sql_query(q2, conn)
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
      <th>id</th>
      <th>code</th>
      <th>name</th>
      <th>area</th>
      <th>area_land</th>
      <th>area_water</th>
      <th>population</th>
      <th>population_growth</th>
      <th>birth_rate</th>
      <th>death_rate</th>
      <th>migration_rate</th>
      <th>created_at</th>
      <th>updated_at</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>af</td>
      <td>Afghanistan</td>
      <td>652230.0</td>
      <td>652230.0</td>
      <td>0.0</td>
      <td>32564342.0</td>
      <td>2.32</td>
      <td>38.57</td>
      <td>13.89</td>
      <td>1.51</td>
      <td>2015-11-01 13:19:49.461734</td>
      <td>2015-11-01 13:19:49.461734</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>al</td>
      <td>Albania</td>
      <td>28748.0</td>
      <td>27398.0</td>
      <td>1350.0</td>
      <td>3029278.0</td>
      <td>0.30</td>
      <td>12.92</td>
      <td>6.58</td>
      <td>3.30</td>
      <td>2015-11-01 13:19:54.431082</td>
      <td>2015-11-01 13:19:54.431082</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>ag</td>
      <td>Algeria</td>
      <td>2381741.0</td>
      <td>2381741.0</td>
      <td>0.0</td>
      <td>39542166.0</td>
      <td>1.84</td>
      <td>23.67</td>
      <td>4.31</td>
      <td>0.92</td>
      <td>2015-11-01 13:19:59.961286</td>
      <td>2015-11-01 13:19:59.961286</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>an</td>
      <td>Andorra</td>
      <td>468.0</td>
      <td>468.0</td>
      <td>0.0</td>
      <td>85580.0</td>
      <td>0.12</td>
      <td>8.13</td>
      <td>6.96</td>
      <td>0.00</td>
      <td>2015-11-01 13:20:03.659945</td>
      <td>2015-11-01 13:20:03.659945</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>ao</td>
      <td>Angola</td>
      <td>1246700.0</td>
      <td>1246700.0</td>
      <td>0.0</td>
      <td>19625353.0</td>
      <td>2.78</td>
      <td>38.78</td>
      <td>11.49</td>
      <td>0.46</td>
      <td>2015-11-01 13:20:08.625072</td>
      <td>2015-11-01 13:20:08.625072</td>
    </tr>
  </tbody>
</table>
</div>



Let's see what the maximum and the minimum population is and then we'll identify the country name. If they are outliers, we should probably remove it from the table.


```python
q3 = "SELECT MIN(population), MAX(population), MIN(population_growth), MAX(population_growth) FROM facts"
data = pd.read_sql_query(q3, conn)
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
      <th>MIN(population)</th>
      <th>MAX(population)</th>
      <th>MIN(population_growth)</th>
      <th>MAX(population_growth)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>7256490011</td>
      <td>0.0</td>
      <td>4.02</td>
    </tr>
  </tbody>
</table>
</div>




```python
q4 = '''
SELECT * FROM facts 
WHERE population == (SELECT MIN(population) from facts);
'''
data = pd.read_sql_query(q4, conn)
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
      <th>id</th>
      <th>code</th>
      <th>name</th>
      <th>area</th>
      <th>area_land</th>
      <th>area_water</th>
      <th>population</th>
      <th>population_growth</th>
      <th>birth_rate</th>
      <th>death_rate</th>
      <th>migration_rate</th>
      <th>created_at</th>
      <th>updated_at</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>250</td>
      <td>ay</td>
      <td>Antarctica</td>
      <td>None</td>
      <td>280000</td>
      <td>None</td>
      <td>0</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>2015-11-01 13:38:44.885746</td>
      <td>2015-11-01 13:38:44.885746</td>
    </tr>
  </tbody>
</table>
</div>




```python
q5 = '''
SELECT * FROM facts 
WHERE population == (SELECT MAX(population) from facts);
'''
data = pd.read_sql_query(q5, conn)
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
      <th>id</th>
      <th>code</th>
      <th>name</th>
      <th>area</th>
      <th>area_land</th>
      <th>area_water</th>
      <th>population</th>
      <th>population_growth</th>
      <th>birth_rate</th>
      <th>death_rate</th>
      <th>migration_rate</th>
      <th>created_at</th>
      <th>updated_at</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>261</td>
      <td>xx</td>
      <td>World</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>7256490011</td>
      <td>1.08</td>
      <td>18.6</td>
      <td>7.8</td>
      <td>None</td>
      <td>2015-11-01 13:39:09.910721</td>
      <td>2015-11-01 13:39:09.910721</td>
    </tr>
  </tbody>
</table>
</div>



It doesn't make much sense to include Antarctica and the entire world as a part of our data analysis, we should definitely exlude this from our analysis.

We can write a SQL query along with subqueries to exlude the min and max population from the data. 


```python
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

q6 = '''
SELECT population, population_growth, birth_rate, death_rate
FROM facts
WHERE population != (SELECT MIN(population) from facts)
AND population != (SELECT MAX(population) from facts)
'''

data = pd.read_sql_query(q6, conn)
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
      <th>population</th>
      <th>population_growth</th>
      <th>birth_rate</th>
      <th>death_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>32564342</td>
      <td>2.32</td>
      <td>38.57</td>
      <td>13.89</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3029278</td>
      <td>0.30</td>
      <td>12.92</td>
      <td>6.58</td>
    </tr>
    <tr>
      <th>2</th>
      <td>39542166</td>
      <td>1.84</td>
      <td>23.67</td>
      <td>4.31</td>
    </tr>
    <tr>
      <th>3</th>
      <td>85580</td>
      <td>0.12</td>
      <td>8.13</td>
      <td>6.96</td>
    </tr>
    <tr>
      <th>4</th>
      <td>19625353</td>
      <td>2.78</td>
      <td>38.78</td>
      <td>11.49</td>
    </tr>
  </tbody>
</table>
</div>



Suppose we are the CIA and we are interested in the future prospects of the countries arround the world. We can plot histograms of the birth rate, death rate, and population growth of the countries.


```python
fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)

data["birth_rate"].hist(ax=ax1)
ax1.set_xlabel("birth_rate")
data["death_rate"].hist(ax=ax2)
ax2.set_xlabel("death_rate")
data["population_growth"].hist(ax=ax3)
ax3.set_xlabel("population_growth")
data["population"].hist(ax=ax4)
ax4.set_xlabel("population")

plt.show()
```


    
![png](/posts_images/2018-02-07-DataQuestGuidedProjectAnalyzingCIAFactbookDataUsingSQLiteAndPython/output_11_0.png)
    


The birth_rate and population growth plot both show a right-skewed distribution, This makes sense as birth rate and population growth are directly related. The death_rate plot shows a normal distribution, almost a double peaked distribution. The population plot is a bit hard to read due to outliers.

Next we are interested to see what city has the highest population density


```python

q7 = '''
SELECT name, CAST(population as float)/CAST(area as float) "density"
FROM facts
WHERE population != (SELECT MIN(population) from facts)
AND population != (SELECT MAX(population) from facts)
ORDER BY density DESC
'''

data = pd.read_sql_query(q7, conn)
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
      <th>name</th>
      <th>density</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Macau</td>
      <td>21168.964286</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Monaco</td>
      <td>15267.500000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Singapore</td>
      <td>8141.279770</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Hong Kong</td>
      <td>6445.041516</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Gaza Strip</td>
      <td>5191.819444</td>
    </tr>
  </tbody>
</table>
</div>



Looks like Macau has the highest population density in the world, not too surprising because Macau is a tourist heavy town with tons of casinos.


```python
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(1,1,1)

data['density'].hist()

plt.show()
```


    
![png](/posts_images/2018-02-07-DataQuestGuidedProjectAnalyzingCIAFactbookDataUsingSQLiteAndPython/output_15_0.png)
    


Again there are several outliers making the data hard to read, let's limit the histogram and increase the number of bins.


```python
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)

data['density'].hist(bins=500)
ax.set_xlim(0, 2000)
plt.show()
```


    
![png](/posts_images/2018-02-07-DataQuestGuidedProjectAnalyzingCIAFactbookDataUsingSQLiteAndPython/output_17_0.png)
    


This table includes cities along with countries. The cities will obviously have way higher density than the countries. So plotting them both together in one histogram doesn't make much sense

This explains why the population histogram we did earlier showed a similar trend.

---

#### Learning Summary

Python/SQL concepts explored: python+sqlite3, pandas, SQL queries, SQL subqueries, matplotlib.plyplot, seaborn, histograms

Python functions and methods used: .cursor(), .read_sql_query(), .set_xlabel(), .set_xlim(), .add_subplot(), .figure()

SQL statements used: SELECT, WHERE, FROM, MIN(), MAX(), ORDER BY, AND

The files used for this project can be found in my [GitHub repository](https://github.com/sengkchu/Dataquest-Guided-Projects-Solutions/tree/master/Guided%20Project_%20Analyzing%20CIA%20Factbook%20Data%20Using%20SQLite%20and%20Python).

