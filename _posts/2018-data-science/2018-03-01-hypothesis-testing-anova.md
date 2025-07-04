---
title: Hypothesis Testing ANOVA
date: 2018-03-01 00:00:00 +/-0000
categories: [Data Science, Statistics]
tags: [statistics]
image:
  path: /posts_images/2018-03-01-hypothesis-testing-anova/cover.png
---

In the previous article, we talked about hypothesis testing using the Welch's t-test on two independent samples of data. So what happens if we want know the statiscal significance for $k$ groups of data?

This is where the analysis of variance technique, or ANOVA is useful.

### ANOVA Assumptions

We'll be looking at SAT scores for five different districts in New York City. Specifically, we'll be using "scores.csv" from [Kaggle](https://www.kaggle.com/nycopendata/high-schools). First let's get the assumptions out of the way:

+ The dependent variable (SAT scores) should be continuous. 
+ The independent variables (districts) should be two or more categorical groups.
+ There must be different participants in each group with no participant being in more than one group. In our case, each school cannot be in more than one district.
+ The dependent variable should be approximately normally distributed for each category.
+ Variances of each group are approximately equal.



### Data Exploration

Let's begin by taking a look at what our data looks like.


```python
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
%matplotlib inline

data = pd.read_csv("scores.csv")
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
      <th>School ID</th>
      <th>School Name</th>
      <th>Borough</th>
      <th>Building Code</th>
      <th>Street Address</th>
      <th>City</th>
      <th>State</th>
      <th>Zip Code</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>...</th>
      <th>End Time</th>
      <th>Student Enrollment</th>
      <th>Percent White</th>
      <th>Percent Black</th>
      <th>Percent Hispanic</th>
      <th>Percent Asian</th>
      <th>Average Score (SAT Math)</th>
      <th>Average Score (SAT Reading)</th>
      <th>Average Score (SAT Writing)</th>
      <th>Percent Tested</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>02M260</td>
      <td>Clinton School Writers and Artists</td>
      <td>Manhattan</td>
      <td>M933</td>
      <td>425 West 33rd Street</td>
      <td>Manhattan</td>
      <td>NY</td>
      <td>10001</td>
      <td>40.75321</td>
      <td>-73.99786</td>
      <td>...</td>
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
      <th>1</th>
      <td>06M211</td>
      <td>Inwood Early College for Health and Informatio...</td>
      <td>Manhattan</td>
      <td>M052</td>
      <td>650 Academy Street</td>
      <td>Manhattan</td>
      <td>NY</td>
      <td>10002</td>
      <td>40.86605</td>
      <td>-73.92486</td>
      <td>...</td>
      <td>3:00 PM</td>
      <td>87.0</td>
      <td>3.4%</td>
      <td>21.8%</td>
      <td>67.8%</td>
      <td>4.6%</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>01M539</td>
      <td>New Explorations into Science, Technology and ...</td>
      <td>Manhattan</td>
      <td>M022</td>
      <td>111 Columbia Street</td>
      <td>Manhattan</td>
      <td>NY</td>
      <td>10002</td>
      <td>40.71873</td>
      <td>-73.97943</td>
      <td>...</td>
      <td>4:00 PM</td>
      <td>1735.0</td>
      <td>28.6%</td>
      <td>13.3%</td>
      <td>18.0%</td>
      <td>38.5%</td>
      <td>657.0</td>
      <td>601.0</td>
      <td>601.0</td>
      <td>91.0%</td>
    </tr>
    <tr>
      <th>3</th>
      <td>02M294</td>
      <td>Essex Street Academy</td>
      <td>Manhattan</td>
      <td>M445</td>
      <td>350 Grand Street</td>
      <td>Manhattan</td>
      <td>NY</td>
      <td>10002</td>
      <td>40.71687</td>
      <td>-73.98953</td>
      <td>...</td>
      <td>2:45 PM</td>
      <td>358.0</td>
      <td>11.7%</td>
      <td>38.5%</td>
      <td>41.3%</td>
      <td>5.9%</td>
      <td>395.0</td>
      <td>411.0</td>
      <td>387.0</td>
      <td>78.9%</td>
    </tr>
    <tr>
      <th>4</th>
      <td>02M308</td>
      <td>Lower Manhattan Arts Academy</td>
      <td>Manhattan</td>
      <td>M445</td>
      <td>350 Grand Street</td>
      <td>Manhattan</td>
      <td>NY</td>
      <td>10002</td>
      <td>40.71687</td>
      <td>-73.98953</td>
      <td>...</td>
      <td>3:00 PM</td>
      <td>383.0</td>
      <td>3.1%</td>
      <td>28.2%</td>
      <td>56.9%</td>
      <td>8.6%</td>
      <td>418.0</td>
      <td>428.0</td>
      <td>415.0</td>
      <td>65.1%</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>




```python
data['Borough'].value_counts()
```




    Brooklyn         121
    Bronx            118
    Manhattan        106
    Queens            80
    Staten Island     10
    Name: Borough, dtype: int64



### Creating New Columns

There is no total score column, so we'll have to create it. In addition, we'll have to find the mean score of the each district across all schools.


```python
data['total_score'] = data['Average Score (SAT Reading)'] +  \
                      data['Average Score (SAT Math)']    +  \
                      data['Average Score (SAT Writing)']
data = data[['Borough', 'total_score']].dropna()        
x = ['Brooklyn', 'Bronx', 'Manhattan', 'Queens', 'Staten Island']
district_dict = {}

#Assigns each test score series to a dictionary key
for district in x:
    district_dict[district] = data[data['Borough'] == district]['total_score']


y = []
yerror = []
#Assigns the mean score and 95% confidence limit to each district
for district in x:
    y.append(district_dict[district].mean())
    yerror.append(1.96*district_dict[district].std()/np.sqrt(district_dict[district].shape[0]))    
    print(district + '_std : {}'.format(district_dict[district].std()))
    
sns.set(font_scale=1.8)
fig = plt.figure(figsize=(10,5))
ax = sns.barplot(x, y, yerr=yerror)
ax.set_ylabel('Average Total SAT Score')
plt.show()
```

    Brooklyn_std : 154.8684270520867
    Bronx_std : 150.39390071890665
    Manhattan_std : 230.29413953637814
    Queens_std : 195.25289850192115
    Staten Island_std : 222.30359621222706
    


    
![png](/posts_images/2018-03-01-hypothesis-testing-anova/output_6_1.png)
    


From our data exploration, we can see that the average SAT scores are quite different for each district. We are interested in knowing if this is caused by random variation in data, or if there is an underlying cause. Since we have five different groups, we cannot use the t-test. Also note that the standard deviation of each group are also very different, so we've violated one of our assumpions. However, we are going to use the 1-way ANOVA test anyway just to understand the concepts.

### The Null and Alternative Hypothesis

There are no significant differences between the groups' mean SAT scores.

$ H_0 : \mu_1 = \mu_2 = \mu_3 = \mu_4 = \mu_5 $

There is a significant difference between the groups' mean SAT scores.

$ H_a : \mu_i \ne \mu_j $

Where  $\mu_i$ and $\mu_j$ can be the mean of any group. If there is at least one group with a significant difference with another group, the null hypothesis will be rejected.

### 1-way ANOVA

Similar to the t-test, we can calculate a score for the ANOVA. Then we can look up the score in the F-distribution and obtain a p-value.

The F-statistic is defined as follows:

$$ F = \frac{MS_{b}} {MS_w} $$

$$ MS_{b} = \frac{SS_{b}} {K-1}$$

$$ MS_{w} = \frac{SS_{w}} {N-K}$$

$$ SS_{b} = {n_k\sum(\bar{x_{k}}-\bar{x_{G}})^2} $$

$$ SS_{w} = \sum(x_{i}-\bar{x_{k}})^2 $$

Where $MS_{b}$ is the estimated variance between groups and $MS_{w}$ is the estimated variance within groups, $\bar{x_{k}}$ is the mean within each group, $n_k$ is the sample size for each group, ${x_i}$ is the individual data point, and $\bar{x_{G}}$ is the total mean. 

This is quite a lot of math, fortunately scipy has a function that plugs in all the values for us. The documentation for calculating 1-way ANOVA using scipy is [here](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f_oneway.html). 


```python
stats.f_oneway(
             district_dict['Brooklyn'], district_dict['Bronx'], \
             district_dict['Manhattan'], district_dict['Queens'], \
             district_dict['Staten Island']
)
```




    F_onewayResult(statistic=12.733085029201668, pvalue=1.0161974965566023e-09)



The resulting pvalue was less than 0.05. We can reject the null hypothesis and conclude that there is a significant difference between the SAT scores for each district. Even though we've obtained a very low p-value, we cannot make any assumptions about the magnitude of the effect. Also scipy does not calculate $SS_b$ and $SS_w$, so it is probably better to write our own code.


```python
districts = ['Brooklyn', 'Bronx', 'Manhattan', 'Queens', 'Staten Island']

ss_b = 0
for d in districts:
    ss_b += district_dict[d].shape[0] * \
            np.sum((district_dict[d].mean() - data['total_score'].mean())**2)

ss_w = 0
for d in districts:
    ss_w += np.sum((district_dict[d] - district_dict[d].mean())**2)

msb = ss_b/4
msw = ss_w/(len(data)-5)
f=msb/msw
print('F_statistic: {}'.format(f))
```

    F_statistic: 12.733085029201687
    

### The Effect Size

We can calculate the magnitude of the effect to determine how large the difference is. One of the measures we can use is Eta-squared.


$$ \eta^2 = \frac{SS_{b}} {SS_{total}}$$

$$ SS_{b} = {n_k\sum(\bar{x_{k}}-\bar{x_{G}})^2} $$

$$ SS_{total} = \sum(x_{i}-\bar{x_{G}})^2 $$


```python
ss_t = np.sum((data['total_score']-data['total_score'].mean())**2)        
eta_squared = ss_b/ss_t
print('eta_squared: {}'.format(eta_squared))
```

    eta_squared: 0.12099887621529214
    

The general rules of thumb given by Cohen and Miles & Shevlin (2001) for analyzing eta-squared, $\eta^2$:

+ Small effect: $ 0.01 $
+ Medium ffect: $ 0.06 $
+ Large effect: $ 0.14 $

From our calculations, the effect size for this ANOVA test would be "Medium". For a full write up on effect sizes click [here](https://imaging.mrc-cbu.cam.ac.uk/statswiki/FAQ/effectSize).
