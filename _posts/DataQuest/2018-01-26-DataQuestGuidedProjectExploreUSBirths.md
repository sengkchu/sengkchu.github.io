---
title: Dataquest Guided Project - Explore US Births
date: 2018-01-26 00:00 +/-0000
categories: [DataQuest]
tags: [dataquest, data science, basics]
image:
  path: /posts_images/2018-01-26-DataQuestGuidedProjectExploreUSBirths/cover.PNG
---


In this project, we will look at the number of births in the United States in the file "US_births_1994-2003_CDC_NCHS.csv". First, we will use basic functions in python to convert the file into a readable format and explore the data.


```python
#Opens the file and read it.
file = open("US_births_1994-2003_CDC_NCHS.csv", "r")
data = file.read()

#Prints the first 400 characters.
data[0:400]
```




    'year,month,date_of_month,day_of_week,births\n1994,1,1,6,8096\n1994,1,2,7,7772\n1994,1,3,1,10142\n1994,1,4,2,11248\n1994,1,5,3,11053\n1994,1,6,4,11406\n1994,1,7,5,11251\n1994,1,8,6,8653\n1994,1,9,7,7910\n1994,1,10,1,10498\n1994,1,11,2,11706\n1994,1,12,3,11567\n1994,1,13,4,11212\n1994,1,14,5,11570\n1994,1,15,6,8660\n1994,1,16,7,8123\n1994,1,17,1,10567\n1994,1,18,2,11541\n1994,1,19,3,11257\n1994,1,20,4,11682\n1994,1,21,5'



It looks like each entry in the data is seperated by "\n". Let's create a list with the variable "births" and seperate each item in the list by \n. We can accomplish this by using the built in .split() function in python.


```python
births = data.split("\n")
births[0:10]
```




    ['year,month,date_of_month,day_of_week,births',
     '1994,1,1,6,8096',
     '1994,1,2,7,7772',
     '1994,1,3,1,10142',
     '1994,1,4,2,11248',
     '1994,1,5,3,11053',
     '1994,1,6,4,11406',
     '1994,1,7,5,11251',
     '1994,1,8,6,8653',
     '1994,1,9,7,7910']



This is still not quite what we want, each element in the list contains information on the year, month, day, day of the week, and births. We want to further break down each element in the list. We can accomplish this creating a list of lists. 

In addition, we want to write a generic function that can convert any csv file into this format.


```python
def read_csv(inputs):
    f = open(inputs, "r")
    data = f.read()
    string_list = data.split("\n")
    #Remove the headers.
    string_list = string_list[1:]
    
    #Splits each list element by ',' and append it to the final list.
    final_list = []   
    for i in string_list:
        int_fields = []
        string_fields = i.split(",")
        #Converts each string to an integer.
        for i in string_fields:
            int_fields.append(int(i))
        final_list.append(int_fields)   
    return final_list

cdc_list = read_csv("US_births_1994-2003_CDC_NCHS.csv")
print(cdc_list[0:10])
```

    [[1994, 1, 1, 6, 8096], [1994, 1, 2, 7, 7772], [1994, 1, 3, 1, 10142], [1994, 1, 4, 2, 11248], [1994, 1, 5, 3, 11053], [1994, 1, 6, 4, 11406], [1994, 1, 7, 5, 11251], [1994, 1, 8, 6, 8653], [1994, 1, 9, 7, 7910], [1994, 1, 10, 1, 10498]]
    

Now that we have the list in the format we want, we can explore the data a little bit. Let's say we are a baby products company, and we want to know if there is a particular month when more babies will be born. If that is the case, we can produce more products in preparation for that month.

We can write a function that takes the list of lists as an argument and then convert it into a dictionary with each month as the dictionary key and the number of births as the value of that key.


```python
def month_births(inputs):
    births_per_month = {}
    #Loops through the list of lists.
    for i in inputs:
        #Sets the month as the first item, and births as the last item.
        month = i[1]
        births = i[-1]
        #Add the number of births to the current value, if the key exists in the dictionary.
        if month in births_per_month:
            births_per_month[month] = births + int(births_per_month[month])
        #Set the key equal to the value, if the key doesn't exist in teh dictionary.
        else:
            births_per_month[month] = births           
    return births_per_month

#The result is a dictionary with the month as the key and the number of births as values.
cdc_month_births = month_births(cdc_list)
cdc_month_births
```




    {1: 3232517,
     2: 3018140,
     3: 3322069,
     4: 3185314,
     5: 3350907,
     6: 3296530,
     7: 3498783,
     8: 3525858,
     9: 3439698,
     10: 3378814,
     11: 3171647,
     12: 3301860}



Looks like the total number births from 1994 - 2003 in August ended up to be the highest. However, the number might not be high enough to justify increased production of baby products.

Let's repeat the process of what we just did above for the day of the week instead of month. We'll write a function that takes in the list of lists as an argument and generate a dictionary with day of the week as the key and number of births as the value. This will give us a count of the total number of births from 1994-2003 for each day of the week


```python
def dow_births(inputs):
    births_dow = {}
    for i in inputs:
        dow = i[-2]
        births = i[-1]
        if dow in births_dow:
            births_dow[dow] = births + int(births_dow[dow])
        else:
            births_dow[dow] = births           
    return births_dow



cdc_day_births = dow_births(cdc_list)
cdc_day_births
```




    {1: 5789166,
     2: 6446196,
     3: 6322855,
     4: 6288429,
     5: 6233657,
     6: 4562111,
     7: 4079723}



According to the international standard ISO-8601, the International Organization for Standardization (ISO) has decreed that Monday shall be the first day of the week. So it looks like the highest total amount of births from 1994 - 2003 is actually a Tuesday.

We can repeat the process again. We'll write a function that takes in the list of lists as an argument and generate a dictionary with year as the key and number of births as the value. This time, the function will be a bit more generic. The function will take in two arguments, the data and the column index. A column index of 0 will refer to the year column, whereas a column index of 1 will refer to the month and so on. That way we can quickly generate dictionary of all the birth counts.


```python
def calc_counts(data, column):
    dictionary = {}
    for i in data:
        column_name = i[column]
        births = i[-1]
        if column_name in dictionary:
            dictionary[column_name] = births + int(dictionary[column_name])
        else:
            dictionary[column_name] = births
            
    return dictionary

cdc_year_births = calc_counts(cdc_list, 0)
cdc_year_births
```




    {1994: 3952767,
     1995: 3899589,
     1996: 3891494,
     1997: 3880894,
     1998: 3941553,
     1999: 3959417,
     2000: 4058814,
     2001: 4025933,
     2002: 4021726,
     2003: 4089950}



It looks like the number of births in the US is increasing every year. 

Lastly, we'll use the same function again to look at the total number of births in each day of the month


```python
cdc_day_births = calc_counts(cdc_list, 2)
cdc_day_births
```




    {1: 1276557,
     2: 1288739,
     3: 1304499,
     4: 1288154,
     5: 1299953,
     6: 1304474,
     7: 1310459,
     8: 1312297,
     9: 1303292,
     10: 1320764,
     11: 1314361,
     12: 1318437,
     13: 1277684,
     14: 1320153,
     15: 1319171,
     16: 1315192,
     17: 1324953,
     18: 1326855,
     19: 1318727,
     20: 1324821,
     21: 1322897,
     22: 1317381,
     23: 1293290,
     24: 1288083,
     25: 1272116,
     26: 1284796,
     27: 1294395,
     28: 1307685,
     29: 1223161,
     30: 1202095,
     31: 746696}



We are not really seeing any patterns here other than the 31st day having signficantly less number of births compared to the rest of the days in the month. This makes sense as not every month have 31 days.

---

#### Learning Summary

Python concepts explored: lists, dictionaries, functions, for loops  
Python functions and methods used: .read(), open(), .split(), .append(), int()


The files used for this project can be found in my [GitHub repository](https://github.com/sengkchu/Dataquest-Guided-Projects-Solutions/tree/master/Guided%20Project_%20Explore%20U.S.%20Births).

