---
title: Dataquest Guided Project - Preparing Data For SQLite
date: 2018-02-07 00:00 +/-0000
categories: [DataQuest]
tags: [dataquest, data science, basics]
---

In this project, we will prepare data in preparation for SQL. We'll have to first clean the data.


```python
import pandas as pd
df = pd.read_csv('academy_awards.csv', encoding='ISO-8859-1')

#Turns off warnings for potentially confusing assignments
pd.options.mode.chained_assignment = None  # default='warn'
df.head()
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
      <th>Year</th>
      <th>Category</th>
      <th>Nominee</th>
      <th>Additional Info</th>
      <th>Won?</th>
      <th>Unnamed: 5</th>
      <th>Unnamed: 6</th>
      <th>Unnamed: 7</th>
      <th>Unnamed: 8</th>
      <th>Unnamed: 9</th>
      <th>Unnamed: 10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2010 (83rd)</td>
      <td>Actor -- Leading Role</td>
      <td>Javier Bardem</td>
      <td>Biutiful {'Uxbal'}</td>
      <td>NO</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2010 (83rd)</td>
      <td>Actor -- Leading Role</td>
      <td>Jeff Bridges</td>
      <td>True Grit {'Rooster Cogburn'}</td>
      <td>NO</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2010 (83rd)</td>
      <td>Actor -- Leading Role</td>
      <td>Jesse Eisenberg</td>
      <td>The Social Network {'Mark Zuckerberg'}</td>
      <td>NO</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2010 (83rd)</td>
      <td>Actor -- Leading Role</td>
      <td>Colin Firth</td>
      <td>The King's Speech {'King George VI'}</td>
      <td>YES</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2010 (83rd)</td>
      <td>Actor -- Leading Role</td>
      <td>James Franco</td>
      <td>127 Hours {'Aron Ralston'}</td>
      <td>NO</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



This is not considered tidy data. Here are the issues:

1. We have string values in the year column, we should change these into integers.

2. Additional Info has the movie name and the role name, this can be split into two columns.

3. The "Won?" column change should be changed to 0's and 1's in.

4. There are a lot of NaN values under the unnammed columns, we should consider dropping them.

First let's take a look at a unnammed columns, we can use .value_counts() to see if there are any significant values in these columns.


```python
print(df['Unnamed: 10'].value_counts())
print(df['Unnamed: 9'].value_counts())
print(df['Unnamed: 8'].value_counts())
print(df['Unnamed: 7'].value_counts())
print(df['Unnamed: 6'].value_counts())
print(df['Unnamed: 5'].value_counts())
```

    *    1
    Name: Unnamed: 10, dtype: int64
    *    1
    Name: Unnamed: 9, dtype: int64
    *                                                 1
     understanding comedy genius - Mack Sennett.""    1
    Name: Unnamed: 8, dtype: int64
    *                                                     1
     while requiring no dangerous solvents. [Systems]"    1
     kindly                                               1
    Name: Unnamed: 7, dtype: int64
    *                                                                   9
     direct radiator bass style cinema loudspeaker systems. [Sound]"    1
     flexibility and water resistance                                   1
     sympathetic                                                        1
    Name: Unnamed: 6, dtype: int64
    *                                                                                                               7
     discoverer of stars                                                                                            1
     D.B. "Don" Keele and Mark E. Engebretson has resulted in the over 20-year dominance of constant-directivity    1
     error-prone measurements on sets. [Digital Imaging Technology]"                                                1
     resilience                                                                                                     1
    Name: Unnamed: 5, dtype: int64
    

It doesn't look like there any significant information under these columns. We can probably drop all the unnamed columns, but first let's convert the year column to integers.


```python
df['Year'] = df['Year'].str[0:4]
df['Year'] = df['Year'].astype(int)
df.head()
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
      <th>Year</th>
      <th>Category</th>
      <th>Nominee</th>
      <th>Additional Info</th>
      <th>Won?</th>
      <th>Unnamed: 5</th>
      <th>Unnamed: 6</th>
      <th>Unnamed: 7</th>
      <th>Unnamed: 8</th>
      <th>Unnamed: 9</th>
      <th>Unnamed: 10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2010</td>
      <td>Actor -- Leading Role</td>
      <td>Javier Bardem</td>
      <td>Biutiful {'Uxbal'}</td>
      <td>NO</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2010</td>
      <td>Actor -- Leading Role</td>
      <td>Jeff Bridges</td>
      <td>True Grit {'Rooster Cogburn'}</td>
      <td>NO</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2010</td>
      <td>Actor -- Leading Role</td>
      <td>Jesse Eisenberg</td>
      <td>The Social Network {'Mark Zuckerberg'}</td>
      <td>NO</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2010</td>
      <td>Actor -- Leading Role</td>
      <td>Colin Firth</td>
      <td>The King's Speech {'King George VI'}</td>
      <td>YES</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2010</td>
      <td>Actor -- Leading Role</td>
      <td>James Franco</td>
      <td>127 Hours {'Aron Ralston'}</td>
      <td>NO</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.describe(include = 'all')
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
      <th>Year</th>
      <th>Category</th>
      <th>Nominee</th>
      <th>Additional Info</th>
      <th>Won?</th>
      <th>Unnamed: 5</th>
      <th>Unnamed: 6</th>
      <th>Unnamed: 7</th>
      <th>Unnamed: 8</th>
      <th>Unnamed: 9</th>
      <th>Unnamed: 10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>10137.000000</td>
      <td>10137</td>
      <td>10137</td>
      <td>9011</td>
      <td>10137</td>
      <td>11</td>
      <td>12</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>NaN</td>
      <td>40</td>
      <td>6001</td>
      <td>6424</td>
      <td>16</td>
      <td>5</td>
      <td>4</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>top</th>
      <td>NaN</td>
      <td>Writing</td>
      <td>Meryl Streep</td>
      <td>Metro-Goldwyn-Mayer</td>
      <td>NO</td>
      <td>*</td>
      <td>*</td>
      <td>*</td>
      <td>*</td>
      <td>*</td>
      <td>*</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>NaN</td>
      <td>888</td>
      <td>16</td>
      <td>60</td>
      <td>7168</td>
      <td>7</td>
      <td>9</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1970.330768</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>std</th>
      <td>23.332917</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1927.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1950.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1970.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1991.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2010.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



We are only interested in data after year 2000, and actors in award winnng categories. We can use boolean filtering and the .isin() method to filter the data out and then create a new dataframe named "nominations".


```python
later_than_2000 = df[df['Year'] > 2000]

award_categories = [
    "Actor -- Leading Role",
    "Actor -- Supporting Role",
    "Actress -- Leading Role",
    "Actress -- Supporting Role"
]


nominations = later_than_2000[later_than_2000['Category'].isin(award_categories)]
nominations.head()
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
      <th>Year</th>
      <th>Category</th>
      <th>Nominee</th>
      <th>Additional Info</th>
      <th>Won?</th>
      <th>Unnamed: 5</th>
      <th>Unnamed: 6</th>
      <th>Unnamed: 7</th>
      <th>Unnamed: 8</th>
      <th>Unnamed: 9</th>
      <th>Unnamed: 10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2010</td>
      <td>Actor -- Leading Role</td>
      <td>Javier Bardem</td>
      <td>Biutiful {'Uxbal'}</td>
      <td>NO</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2010</td>
      <td>Actor -- Leading Role</td>
      <td>Jeff Bridges</td>
      <td>True Grit {'Rooster Cogburn'}</td>
      <td>NO</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2010</td>
      <td>Actor -- Leading Role</td>
      <td>Jesse Eisenberg</td>
      <td>The Social Network {'Mark Zuckerberg'}</td>
      <td>NO</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2010</td>
      <td>Actor -- Leading Role</td>
      <td>Colin Firth</td>
      <td>The King's Speech {'King George VI'}</td>
      <td>YES</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2010</td>
      <td>Actor -- Leading Role</td>
      <td>James Franco</td>
      <td>127 Hours {'Aron Ralston'}</td>
      <td>NO</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



We can also use use the .map() method to convert all the "No" values to 0 and all the "Yes" values to 1.


```python
yes_no = {
    "NO":0,
    "YES":1
}

nominations['Won'] = nominations['Won?'].map(yes_no)
nominations.head()

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
      <th>Year</th>
      <th>Category</th>
      <th>Nominee</th>
      <th>Additional Info</th>
      <th>Won?</th>
      <th>Unnamed: 5</th>
      <th>Unnamed: 6</th>
      <th>Unnamed: 7</th>
      <th>Unnamed: 8</th>
      <th>Unnamed: 9</th>
      <th>Unnamed: 10</th>
      <th>Won</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2010</td>
      <td>Actor -- Leading Role</td>
      <td>Javier Bardem</td>
      <td>Biutiful {'Uxbal'}</td>
      <td>NO</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2010</td>
      <td>Actor -- Leading Role</td>
      <td>Jeff Bridges</td>
      <td>True Grit {'Rooster Cogburn'}</td>
      <td>NO</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2010</td>
      <td>Actor -- Leading Role</td>
      <td>Jesse Eisenberg</td>
      <td>The Social Network {'Mark Zuckerberg'}</td>
      <td>NO</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2010</td>
      <td>Actor -- Leading Role</td>
      <td>Colin Firth</td>
      <td>The King's Speech {'King George VI'}</td>
      <td>YES</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2010</td>
      <td>Actor -- Leading Role</td>
      <td>James Franco</td>
      <td>127 Hours {'Aron Ralston'}</td>
      <td>NO</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Next, we'll drop the unnamed columns


```python
columns_drop = [
    "Won?",
    "Unnamed: 5",
    "Unnamed: 6",
    "Unnamed: 7",
    "Unnamed: 8",
    "Unnamed: 9",
    "Unnamed: 10"
]
final_nominations = nominations.drop(columns_drop, axis = 1)
```


```python
final_nominations.head()
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
      <th>Year</th>
      <th>Category</th>
      <th>Nominee</th>
      <th>Additional Info</th>
      <th>Won</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2010</td>
      <td>Actor -- Leading Role</td>
      <td>Javier Bardem</td>
      <td>Biutiful {'Uxbal'}</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2010</td>
      <td>Actor -- Leading Role</td>
      <td>Jeff Bridges</td>
      <td>True Grit {'Rooster Cogburn'}</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2010</td>
      <td>Actor -- Leading Role</td>
      <td>Jesse Eisenberg</td>
      <td>The Social Network {'Mark Zuckerberg'}</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2010</td>
      <td>Actor -- Leading Role</td>
      <td>Colin Firth</td>
      <td>The King's Speech {'King George VI'}</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2010</td>
      <td>Actor -- Leading Role</td>
      <td>James Franco</td>
      <td>127 Hours {'Aron Ralston'}</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



The last thing we'll have to do is separate the "Additional Info" column into two new columns "Movie" and "Character". We'll have to manipulate the strings and the split the two values in this one column in two new strings.

First we'll use .str.rstrip() to remove the end quotation mark and the bracket from the column series, then we'll split the the string again into a series of lists.


```python
additional_info_one = final_nominations['Additional Info'].str.rstrip(to_strip = "'}")
additional_info_one.head()
```




    0                        Biutiful {'Uxbal
    1             True Grit {'Rooster Cogburn
    2    The Social Network {'Mark Zuckerberg
    3      The King's Speech {'King George VI
    4                127 Hours {'Aron Ralston
    Name: Additional Info, dtype: object




```python
#Split into a series of lists
additional_info_two = additional_info_one.str.split(' {\'')
additional_info_two.head()
```




    0                        [Biutiful, Uxbal]
    1             [True Grit, Rooster Cogburn]
    2    [The Social Network, Mark Zuckerberg]
    3      [The King's Speech, King George VI]
    4                [127 Hours, Aron Ralston]
    Name: Additional Info, dtype: object




```python
#Set a series with the first element to movie names
movie_names = additional_info_two.str[0]


#Set a series with the second element to characters
characters = additional_info_two.str[1]
```


```python
final_nominations["Movie"] = movie_names
final_nominations["Character"] = characters
final_nominations = final_nominations.drop(["Additional Info"], axis=1)
final_nominations.head()
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
      <th>Year</th>
      <th>Category</th>
      <th>Nominee</th>
      <th>Won</th>
      <th>Movie</th>
      <th>Character</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2010</td>
      <td>Actor -- Leading Role</td>
      <td>Javier Bardem</td>
      <td>0</td>
      <td>Biutiful</td>
      <td>Uxbal</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2010</td>
      <td>Actor -- Leading Role</td>
      <td>Jeff Bridges</td>
      <td>0</td>
      <td>True Grit</td>
      <td>Rooster Cogburn</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2010</td>
      <td>Actor -- Leading Role</td>
      <td>Jesse Eisenberg</td>
      <td>0</td>
      <td>The Social Network</td>
      <td>Mark Zuckerberg</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2010</td>
      <td>Actor -- Leading Role</td>
      <td>Colin Firth</td>
      <td>1</td>
      <td>The King's Speech</td>
      <td>King George VI</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2010</td>
      <td>Actor -- Leading Role</td>
      <td>James Franco</td>
      <td>0</td>
      <td>127 Hours</td>
      <td>Aron Ralston</td>
    </tr>
  </tbody>
</table>
</div>



Now that we are done cleaning up the data, we can do some simply analysis using sqlite3


```python
import sqlite3

conn = sqlite3.connect("nominations.db")
cursor = conn.cursor()

#Creates the table "nominations"
final_nominations.to_sql("nominations", conn, index=False)

```


```python
q1 = '''
Pragma table_info(nominations)
'''
result = cursor.execute(q1).fetchall()
result
```




    [(0, 'Year', 'INTEGER', 0, None, 0),
     (1, 'Category', 'TEXT', 0, None, 0),
     (2, 'Nominee', 'TEXT', 0, None, 0),
     (3, 'Won', 'INTEGER', 0, None, 0),
     (4, 'Movie', 'TEXT', 0, None, 0),
     (5, 'Character', 'TEXT', 0, None, 0)]




```python
q2 = '''
SELECT * FROM nominations LIMIT 10
'''

result = cursor.execute(q2).fetchall()
result
```




    [(2010, 'Actor -- Leading Role', 'Javier Bardem', 0, 'Biutiful', 'Uxbal'),
     (2010,
      'Actor -- Leading Role',
      'Jeff Bridges',
      0,
      'True Grit',
      'Rooster Cogburn'),
     (2010,
      'Actor -- Leading Role',
      'Jesse Eisenberg',
      0,
      'The Social Network',
      'Mark Zuckerberg'),
     (2010,
      'Actor -- Leading Role',
      'Colin Firth',
      1,
      "The King's Speech",
      'King George VI'),
     (2010,
      'Actor -- Leading Role',
      'James Franco',
      0,
      '127 Hours',
      'Aron Ralston'),
     (2010,
      'Actor -- Supporting Role',
      'Christian Bale',
      1,
      'The Fighter',
      'Dicky Eklund'),
     (2010,
      'Actor -- Supporting Role',
      'John Hawkes',
      0,
      "Winter's Bone",
      'Teardrop'),
     (2010,
      'Actor -- Supporting Role',
      'Jeremy Renner',
      0,
      'The Town',
      'James Coughlin'),
     (2010,
      'Actor -- Supporting Role',
      'Mark Ruffalo',
      0,
      'The Kids Are All Right',
      'Paul'),
     (2010,
      'Actor -- Supporting Role',
      'Geoffrey Rush',
      0,
      "The King's Speech",
      'Lionel Logue')]




```python
conn.close()
```

---

#### Learning Summary

Python/SQL concepts explored: python+sqlite3, pandas, data cleaning, columns manipulation

Python functions and methods used: .str.rstrip(), .str.split(), .connect(), .cursor(), .drop(), .str[], .map(), .value_counts()

SQL statements used: SELECT, FROM, PRAGMA

The files used for this project can be found in my [GitHub repository](https://github.com/sengkchu/Dataquest-Guided-Projects-Solutions/tree/master/Guided%20Project_%20Preparing%20data%20for%20SQLite).
