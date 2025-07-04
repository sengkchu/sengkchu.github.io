---
title: Dataquest Guided Project - Winning Jeopardy
date: 2018-02-09 00:00 +/-0000
categories: [DataQuest]
tags: [dataquest, basics]
image:
  path: /posts_images/2018-02-09-DataQuestGuidedProjectWinningJeopardy/cover.PNG
---


In this project, we'll look at 20,000 rows of the jeopardy dataset in "jeopardy.csv". We want to see if there are patterns in the questions asked so we can get a little bit of an edge to win.

First, we'll have to tidy up the data.


```python
import pandas as pd
import matplotlib.pyplot as plt

jeopardy = pd.read_csv('jeopardy.csv')
jeopardy.head(5)
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
      <th>Show Number</th>
      <th>Air Date</th>
      <th>Round</th>
      <th>Category</th>
      <th>Value</th>
      <th>Question</th>
      <th>Answer</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4680</td>
      <td>2004-12-31</td>
      <td>Jeopardy!</td>
      <td>HISTORY</td>
      <td>$200</td>
      <td>For the last 8 years of his life, Galileo was ...</td>
      <td>Copernicus</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4680</td>
      <td>2004-12-31</td>
      <td>Jeopardy!</td>
      <td>ESPN's TOP 10 ALL-TIME ATHLETES</td>
      <td>$200</td>
      <td>No. 2: 1912 Olympian; football star at Carlisl...</td>
      <td>Jim Thorpe</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4680</td>
      <td>2004-12-31</td>
      <td>Jeopardy!</td>
      <td>EVERYBODY TALKS ABOUT IT...</td>
      <td>$200</td>
      <td>The city of Yuma in this state has a record av...</td>
      <td>Arizona</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4680</td>
      <td>2004-12-31</td>
      <td>Jeopardy!</td>
      <td>THE COMPANY LINE</td>
      <td>$200</td>
      <td>In 1963, live on "The Art Linkletter Show", th...</td>
      <td>McDonald's</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4680</td>
      <td>2004-12-31</td>
      <td>Jeopardy!</td>
      <td>EPITAPHS &amp; TRIBUTES</td>
      <td>$200</td>
      <td>Signer of the Dec. of Indep., framer of the Co...</td>
      <td>John Adams</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(jeopardy.columns)
```

    Index(['Show Number', ' Air Date', ' Round', ' Category', ' Value',
           ' Question', ' Answer'],
          dtype='object')
    

Looks like there is a space after each column name, we can fix this pretty easily with the .columns() method.


```python
jeopardy.columns = ['Show Number', 'Air Date', 'Round', 'Category', 'Value',
       'Question', 'Answer']
jeopardy.columns
```




    Index(['Show Number', 'Air Date', 'Round', 'Category', 'Value', 'Question',
           'Answer'],
          dtype='object')



Next, let's make all the strings in the question and answer columns lower case. We can do write a function and then use the .apply() method.

We also want to remove all the punctuations, the goal is to have the "Question" and "Answer" columns down to just words.


```python
import re
def lowercase_no_punct(string):
    lower = string.lower()
    punremoved = re.sub('[^A-Za-z0-9\s]','', lower)
    return punremoved
```


```python
jeopardy['clean_question'] = jeopardy['Question'].apply(lowercase_no_punct)
jeopardy['clean_answer'] = jeopardy['Answer'].apply(lowercase_no_punct)
```

The "Value" column is usually a dollar sign followed by a number. However, this is currently in a string format. We should conver tthis to an integer and remove the dollar sign.


```python
def punremovandtoint(string):
    punremoved = re.sub('[^A-Za-z0-9\s]','', string)
    try:
        integer = int(punremoved)
    except Exception:
        integer = 0
    return integer
```


```python
jeopardy['clean_values'] = jeopardy['Value'].apply(punremovandtoint)
```

We'll have to convert the values in the "Air Date" column into a datetime object


```python
jeopardy['Air Date'] = pd.to_datetime(jeopardy['Air Date'])
```

Let's see what our table currently looks like


```python
jeopardy.head()
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
      <th>Show Number</th>
      <th>Air Date</th>
      <th>Round</th>
      <th>Category</th>
      <th>Value</th>
      <th>Question</th>
      <th>Answer</th>
      <th>clean_question</th>
      <th>clean_answer</th>
      <th>clean_values</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4680</td>
      <td>2004-12-31</td>
      <td>Jeopardy!</td>
      <td>HISTORY</td>
      <td>$200</td>
      <td>For the last 8 years of his life, Galileo was ...</td>
      <td>Copernicus</td>
      <td>for the last 8 years of his life galileo was u...</td>
      <td>copernicus</td>
      <td>200</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4680</td>
      <td>2004-12-31</td>
      <td>Jeopardy!</td>
      <td>ESPN's TOP 10 ALL-TIME ATHLETES</td>
      <td>$200</td>
      <td>No. 2: 1912 Olympian; football star at Carlisl...</td>
      <td>Jim Thorpe</td>
      <td>no 2 1912 olympian football star at carlisle i...</td>
      <td>jim thorpe</td>
      <td>200</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4680</td>
      <td>2004-12-31</td>
      <td>Jeopardy!</td>
      <td>EVERYBODY TALKS ABOUT IT...</td>
      <td>$200</td>
      <td>The city of Yuma in this state has a record av...</td>
      <td>Arizona</td>
      <td>the city of yuma in this state has a record av...</td>
      <td>arizona</td>
      <td>200</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4680</td>
      <td>2004-12-31</td>
      <td>Jeopardy!</td>
      <td>THE COMPANY LINE</td>
      <td>$200</td>
      <td>In 1963, live on "The Art Linkletter Show", th...</td>
      <td>McDonald's</td>
      <td>in 1963 live on the art linkletter show this c...</td>
      <td>mcdonalds</td>
      <td>200</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4680</td>
      <td>2004-12-31</td>
      <td>Jeopardy!</td>
      <td>EPITAPHS &amp; TRIBUTES</td>
      <td>$200</td>
      <td>Signer of the Dec. of Indep., framer of the Co...</td>
      <td>John Adams</td>
      <td>signer of the dec of indep framer of the const...</td>
      <td>john adams</td>
      <td>200</td>
    </tr>
  </tbody>
</table>
</div>



Now that the data is cleaned, we can start analyzing it. 

Suppose we are interested in the number of words in the answer that occurs in the question. We'll create a function and use the .apply() method to create a new column. This column will have ratio of matching question words to total answer words.


```python
def cleaner(series):
    split_answer = series['clean_answer'].split(' ')
    split_question = series['clean_question'].split(' ')
    match_count = 0
    if "the" in split_answer:
        split_answer.remove('the')
    if len(split_answer) == 0:
        return 0
    for item in split_answer:
        if item in split_question:
            match_count +=1
    return match_count/len(split_answer)
```


```python
jeopardy['answer_in_question'] = jeopardy.apply(cleaner, axis=1)
jeopardy['answer_in_question'].mean()
```




    0.060493257069335872



It looks like the answer only appears in the question 6% of the time, so this is not a super reliable strategy.

Next, we'll look at words used in the questions column. We can write a function to see how often they repeat


```python
question_overlap = []
#a python set is an unordered list of items
terms_used = set()
for idx, row in jeopardy.iterrows():
    split_question = row['clean_question'].split(" ")     
    match_count = 0
    newlist = []
    for word in split_question:
        if len(word) >= 6:
            newlist.append(word)
    for word in newlist:
        if word in terms_used:
            match_count += 1
    for word in newlist:
        terms_used.add(word)
    if len(newlist) > 0:
        match_count = match_count/len(newlist)
    question_overlap.append(match_count)

jeopardy['question_overlap'] = question_overlap
```


```python
jeopardy['question_overlap'].mean()
```




    0.69087373156719623



There is a 69% overlap of words between new questions and old ones. However words can be put together as different phases with a big difference in meaning. So this  huge overlap is not super significant.

Let's take a look at the number of questions that are > 800 dollars. Maybe it is a good idea to only study high value questions.


```python
def highvalue(row):
    value = 0
    if row['clean_values'] > 800:
        value = 1
    return value

jeopardy['high_value'] = jeopardy.apply(highvalue, axis =1)
```


```python
high_value_count = jeopardy[jeopardy['high_value'] == 1].shape[0]
low_value_count = jeopardy[jeopardy['high_value'] == 0].shape[0]
```


```python
print(high_value_count)
low_value_count
```

    5734
    




    14265



It doesnt look like there are that many high value questions in the dataset. 

We can create a function that takes in a word, then return the # of high/low values questions this word showed up in. Maybe this will help us study.


```python
def highlowcounts(word):
    low_count = 0
    high_count = 0 
    for idx, row in jeopardy.iterrows():
        if word in row['clean_question'].split(' '):
            if row["high_value"] == 1:
                high_count += 1
            else:
                low_count += 1   
    return high_count, low_count
```


```python
observed_expected = []
comparison_terms = list(terms_used)[:5]
comparison_terms
```




    ['emigrated', 'ruffles', 'waterworld', 'mussorgsky', 'appendages']




```python
for term in comparison_terms:
    observed_expected.append(highlowcounts(term))

observed_expected
```




    [(1, 0), (0, 2), (1, 0), (1, 1), (1, 2)]



We can use the chi squared test to see if the values of the terms in "comparsion_terms" are statiscally significant.


```python
chi_squared =[]
from scipy.stats import chisquare
import numpy as np
for lists in observed_expected:
    total = sum(lists)
    total_prop = total/jeopardy.shape[0]
    expected_high = total_prop * high_value_count
    expected_low = total_prop * low_value_count
    observed = np.array([lists[0], lists[1]])
    expected = np.array([expected_high, expected_low])
    chi_squared.append(chisquare(observed, expected))
```


```python
chi_squared
```




    [Power_divergenceResult(statistic=2.4877921171956752, pvalue=0.11473257634454047),
     Power_divergenceResult(statistic=0.80392569225376798, pvalue=0.36992223780795708),
     Power_divergenceResult(statistic=2.4877921171956752, pvalue=0.11473257634454047),
     Power_divergenceResult(statistic=0.44487748166127949, pvalue=0.50477764875459963),
     Power_divergenceResult(statistic=0.031881167234403623, pvalue=0.85828871632352932)]



None of the p values are less than 0.05 so this is not statiscally significant.

---

#### Learning Summary

Python concepts explored: pandas, matplotlib, data cleaning, string manipulation, chi squared test, regex, try/except

Python functions and methods used: .columns, .lower(), .sub(), .apply(), sum(), .array(), .split(), .shape, .mean(), .iterrows(), .remove(), .add(), .append()

The files used for this project can be found in my [GitHub repository](https://github.com/sengkchu/Dataquest-Guided-Projects-Solutions/tree/master/Guided%20Project_%20Winning%20Jeopardy).
