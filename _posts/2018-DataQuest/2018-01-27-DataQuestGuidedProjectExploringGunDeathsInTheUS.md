---
title: Dataquest Guided Project - Exploring Gun Deaths In The US
date: 2018-01-27 00:00 +/-0000
categories: [DataQuest]
tags: [dataquest, basics]
image:
  path: /posts_images/2018-01-27-DataQuestGuidedProjectExploringGunDeathsInTheUS/cover.PNG
---


In this project we will look at the total number of gun deaths by race in the US in the file "guns.csv". We will also use the US census ("census.csv") to convert the total number of gun deaths to a perecentage by race. We will accomplish this by using basic functions in python to convert the files into a readable format and explore the data. In addition, we'll also work with list comprehension techniques and the datetime library.


```python
#Converts the csv file into a list of lists
import csv
file = open("guns.csv", "r")
temp = csv.reader(file)
data = list(temp)

data[0:3]
```




    [['',
      'year',
      'month',
      'intent',
      'police',
      'sex',
      'age',
      'race',
      'hispanic',
      'place',
      'education'],
     ['1',
      '2012',
      '01',
      'Suicide',
      '0',
      'M',
      '34',
      'Asian/Pacific Islander',
      '100',
      'Home',
      '4'],
     ['2', '2012', '01', 'Suicide', '0', 'F', '21', 'White', '100', 'Street', '3']]



The csv module didn't remove the headers for us, but it did convert the file into a list of lists which is the format we are looking for. We can use list slicing techniques remove the header and store it as a seperate variable "headers".


```python
headers = data[0:1]
data = data[1:]
print(headers)
print("-------------------------------------")
print(data[0:5])
```

    [['', 'year', 'month', 'intent', 'police', 'sex', 'age', 'race', 'hispanic', 'place', 'education']]
    -------------------------------------
    [['1', '2012', '01', 'Suicide', '0', 'M', '34', 'Asian/Pacific Islander', '100', 'Home', '4'], ['2', '2012', '01', 'Suicide', '0', 'F', '21', 'White', '100', 'Street', '3'], ['3', '2012', '01', 'Suicide', '0', 'M', '60', 'White', '100', 'Other specified', '4'], ['4', '2012', '02', 'Suicide', '0', 'M', '64', 'White', '100', 'Home', '4'], ['5', '2012', '02', 'Suicide', '0', 'M', '31', 'White', '100', 'Other specified', '2']]
    

Now that the data is in a more readable format , we can begin analyzing the data. The first column is unlabeled, but the values in this column is increases by 1 for every list entry. We can assume that this is the ID column.

Suppose we are the government and we are interested to see the total number of gun deaths every year. This might give us an idea of how our current gun regulations are doing. We can accomplish this by creating a dictionary and assigning each year to a key and the number of deaths as the value.


```python
#Extracts the 'year' column from the list of lists
years = [row[1] for row in data]
print(years[0:10])
```

    ['2012', '2012', '2012', '2012', '2012', '2012', '2012', '2012', '2012', '2012']
    

Next we can use a for loop to create a counter in order to populate the dictionary.


```python
year_counts = {}
for i in years:
    if i in year_counts:
        year_counts[i] += 1
    else:
        year_counts[i] = 1
year_counts
```




    {'2012': 33563, '2013': 33636, '2014': 33599}



It looks like the total number of deaths from 2012 to 2014 are relatively close to each other.

Let's break it down even more, we want to look at each month of each year and calculate the total number of gun deaths. Right now the year and month columns are in the string format. We can create a new dictionary and populate it with the month of each year as the key.


```python
import datetime

#The day is not specified in our data, this value will be assignedd as 1.
dates = [datetime.datetime(year=int(row[1]), month=int(row[2]), day=1) for row in data]
date_counts = {}
for i in dates:
    if i in date_counts:
        date_counts[i] += 1
    else:
        date_counts[i] = 1
date_counts
```




    {datetime.datetime(2012, 1, 1, 0, 0): 2758,
     datetime.datetime(2012, 2, 1, 0, 0): 2357,
     datetime.datetime(2012, 3, 1, 0, 0): 2743,
     datetime.datetime(2012, 4, 1, 0, 0): 2795,
     datetime.datetime(2012, 5, 1, 0, 0): 2999,
     datetime.datetime(2012, 6, 1, 0, 0): 2826,
     datetime.datetime(2012, 7, 1, 0, 0): 3026,
     datetime.datetime(2012, 8, 1, 0, 0): 2954,
     datetime.datetime(2012, 9, 1, 0, 0): 2852,
     datetime.datetime(2012, 10, 1, 0, 0): 2733,
     datetime.datetime(2012, 11, 1, 0, 0): 2729,
     datetime.datetime(2012, 12, 1, 0, 0): 2791,
     datetime.datetime(2013, 1, 1, 0, 0): 2864,
     datetime.datetime(2013, 2, 1, 0, 0): 2375,
     datetime.datetime(2013, 3, 1, 0, 0): 2862,
     datetime.datetime(2013, 4, 1, 0, 0): 2798,
     datetime.datetime(2013, 5, 1, 0, 0): 2806,
     datetime.datetime(2013, 6, 1, 0, 0): 2920,
     datetime.datetime(2013, 7, 1, 0, 0): 3079,
     datetime.datetime(2013, 8, 1, 0, 0): 2859,
     datetime.datetime(2013, 9, 1, 0, 0): 2742,
     datetime.datetime(2013, 10, 1, 0, 0): 2808,
     datetime.datetime(2013, 11, 1, 0, 0): 2758,
     datetime.datetime(2013, 12, 1, 0, 0): 2765,
     datetime.datetime(2014, 1, 1, 0, 0): 2651,
     datetime.datetime(2014, 2, 1, 0, 0): 2361,
     datetime.datetime(2014, 3, 1, 0, 0): 2684,
     datetime.datetime(2014, 4, 1, 0, 0): 2862,
     datetime.datetime(2014, 5, 1, 0, 0): 2864,
     datetime.datetime(2014, 6, 1, 0, 0): 2931,
     datetime.datetime(2014, 7, 1, 0, 0): 2884,
     datetime.datetime(2014, 8, 1, 0, 0): 2970,
     datetime.datetime(2014, 9, 1, 0, 0): 2914,
     datetime.datetime(2014, 10, 1, 0, 0): 2865,
     datetime.datetime(2014, 11, 1, 0, 0): 2756,
     datetime.datetime(2014, 12, 1, 0, 0): 2857}



We can repeat the process above and break down the data by sex.


```python
sex = [row[5] for row in data]
sex_counts = {}
for i in sex:
    if i in sex_counts:
        sex_counts[i] += 1
    else:
        sex_counts[i] = 1
print(sex_counts)
```

    {'M': 86349, 'F': 14449}
    

It looks like there are significantly more males gun deaths than females.

We use the similar for loop to break down the data by race.


```python
race = [row[7] for row in data]
race_counts = {}
for i in race:
    if i in race_counts:
        race_counts[i] += 1
    else:
        race_counts[i] = 1
race_counts
```




    {'Asian/Pacific Islander': 1326,
     'Black': 23296,
     'Hispanic': 9022,
     'Native American/Native Alaskan': 917,
     'White': 66237}



It looks like whites have highest total number of gun deaths. However, there are significantly higher number of whites in the US than all the other groups. We need population data that can give us the total population of each race.

We will need to look at census data "census.csv".


```python
f2 = open("census.csv", "r")
temp2 = csv.reader(f2)
census = list(temp2)
census[0:2]
```




    [['Id',
      'Year',
      'Id',
      'Sex',
      'Id',
      'Hispanic Origin',
      'Id',
      'Id2',
      'Geography',
      'Total',
      'Race Alone - White',
      'Race Alone - Hispanic',
      'Race Alone - Black or African American',
      'Race Alone - American Indian and Alaska Native',
      'Race Alone - Asian',
      'Race Alone - Native Hawaiian and Other Pacific Islander',
      'Two or More Races'],
     ['cen42010',
      'April 1, 2010 Census',
      'totsex',
      'Both Sexes',
      'tothisp',
      'Total',
      '0100000US',
      '',
      'United States',
      '308745538',
      '197318956',
      '44618105',
      '40250635',
      '3739506',
      '15159516',
      '674625',
      '6984195']]



We are going to create a dictionary of the total population of each race. We'll have to create it this dictionary manually using the data printed above. Next we'll create a another dictionary with each race as the key and the rate of gun deaths per 100,000 people.


```python
mapping = {}
mapping['Asian/Pacific Islander'] = 674625+6984195
mapping['Black'] = 44618105
mapping['Hispanic'] = 44618105
mapping['Native American/Native Alaskan'] = 15159516
mapping['White'] = 197318956

race_per_hundredk = {}
#We can iterate both the key and the value in a dictionary using .items()
for key, value in (race_counts.items()):    
    race_per_hundredk[key] = value*100000/mapping[key]
race_per_hundredk
```




    {'Asian/Pacific Islander': 17.313372033811998,
     'Black': 52.21198883278436,
     'Hispanic': 20.220491210910907,
     'Native American/Native Alaskan': 6.049005786200562,
     'White': 33.56849303419181}



It looks like the category 'black' have the highest rate of gun deaths. Let's look filter the further and look at the rate of gun deaths by homicide.


```python
intent = [row[3] for row in data]

races = [row[7] for row in data]
homicide_race_counts = {}
#We can iterate over both the index and the element in the list using the enumerate() function
for i, rac in enumerate(races):
    if intent[i] == 'Homicide':
        if rac in homicide_race_counts:
            homicide_race_counts[rac] += 1
        else:
            homicide_race_counts[rac] = 1
            
homicide_race_counts
```




    {'Asian/Pacific Islander': 559,
     'Black': 19510,
     'Hispanic': 5634,
     'Native American/Native Alaskan': 326,
     'White': 9147}




```python
homicide_race_per_hundredk = {}
for key, value in (homicide_race_counts.items()):    
    homicide_race_per_hundredk[key] = value*100000/mapping[key]
homicide_race_per_hundredk
```




    {'Asian/Pacific Islander': 7.298774484842313,
     'Black': 43.7266441503959,
     'Hispanic': 12.627161104219912,
     'Native American/Native Alaskan': 2.1504644343526533,
     'White': 4.6356417981453335}



It appears that the rate of gun deaths by homicide are highest in the 'black' and 'hispanic' racial categories. 

---

#### Learning Summary

Python concepts explored: list comprehension, datetime module, csv module  
Python functions and methods used: csv.reader(), .items(), list(), datetime.datetime()


The files used for this project can be found in my [GitHub repository](https://github.com/sengkchu/Dataquest-Guided-Projects-Solutions/blob/master/Guided%20Project_%20Exploring%20Gun%20Deaths%20in%20the%20US/).

