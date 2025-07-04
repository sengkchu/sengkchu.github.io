---
title: Dataquest Guided Project - Analyzing Movie Reviews
date: 2018-02-08 00:00 +/-0000
categories: [DataQuest]
tags: [dataquest]
image:
  path: /posts_images/2018-02-08-data-quest-guided-project-analyzing-movie-reviews/cover.PNG
---


In this project, we willl analyze various movie review websites using "fandango_score_comparison.csv" We will use descriptive statistics to draw comparisons between fandango and other review websites. In addition, we'll also use linear regression to determine fandango review scores based on other review scores.


```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

movies = pd.read_csv('fandango_score_comparison.csv')
movies.head()
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
      <th>FILM</th>
      <th>RottenTomatoes</th>
      <th>RottenTomatoes_User</th>
      <th>Metacritic</th>
      <th>Metacritic_User</th>
      <th>IMDB</th>
      <th>Fandango_Stars</th>
      <th>Fandango_Ratingvalue</th>
      <th>RT_norm</th>
      <th>RT_user_norm</th>
      <th>...</th>
      <th>IMDB_norm</th>
      <th>RT_norm_round</th>
      <th>RT_user_norm_round</th>
      <th>Metacritic_norm_round</th>
      <th>Metacritic_user_norm_round</th>
      <th>IMDB_norm_round</th>
      <th>Metacritic_user_vote_count</th>
      <th>IMDB_user_vote_count</th>
      <th>Fandango_votes</th>
      <th>Fandango_Difference</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Avengers: Age of Ultron (2015)</td>
      <td>74</td>
      <td>86</td>
      <td>66</td>
      <td>7.1</td>
      <td>7.8</td>
      <td>5.0</td>
      <td>4.5</td>
      <td>3.70</td>
      <td>4.3</td>
      <td>...</td>
      <td>3.90</td>
      <td>3.5</td>
      <td>4.5</td>
      <td>3.5</td>
      <td>3.5</td>
      <td>4.0</td>
      <td>1330</td>
      <td>271107</td>
      <td>14846</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Cinderella (2015)</td>
      <td>85</td>
      <td>80</td>
      <td>67</td>
      <td>7.5</td>
      <td>7.1</td>
      <td>5.0</td>
      <td>4.5</td>
      <td>4.25</td>
      <td>4.0</td>
      <td>...</td>
      <td>3.55</td>
      <td>4.5</td>
      <td>4.0</td>
      <td>3.5</td>
      <td>4.0</td>
      <td>3.5</td>
      <td>249</td>
      <td>65709</td>
      <td>12640</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ant-Man (2015)</td>
      <td>80</td>
      <td>90</td>
      <td>64</td>
      <td>8.1</td>
      <td>7.8</td>
      <td>5.0</td>
      <td>4.5</td>
      <td>4.00</td>
      <td>4.5</td>
      <td>...</td>
      <td>3.90</td>
      <td>4.0</td>
      <td>4.5</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>627</td>
      <td>103660</td>
      <td>12055</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Do You Believe? (2015)</td>
      <td>18</td>
      <td>84</td>
      <td>22</td>
      <td>4.7</td>
      <td>5.4</td>
      <td>5.0</td>
      <td>4.5</td>
      <td>0.90</td>
      <td>4.2</td>
      <td>...</td>
      <td>2.70</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>2.5</td>
      <td>2.5</td>
      <td>31</td>
      <td>3136</td>
      <td>1793</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Hot Tub Time Machine 2 (2015)</td>
      <td>14</td>
      <td>28</td>
      <td>29</td>
      <td>3.4</td>
      <td>5.1</td>
      <td>3.5</td>
      <td>3.0</td>
      <td>0.70</td>
      <td>1.4</td>
      <td>...</td>
      <td>2.55</td>
      <td>0.5</td>
      <td>1.5</td>
      <td>1.5</td>
      <td>1.5</td>
      <td>2.5</td>
      <td>88</td>
      <td>19560</td>
      <td>1021</td>
      <td>0.5</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>



First, we'll use a histogram to see the distribution of ratings for "Fandango_Stars" and "Metacritic_norm_round".


```python
mc = movies['Metacritic_norm_round']
fd = movies['Fandango_Stars']

plt.hist(mc, 5)
plt.show()

plt.hist(fd, 5)
plt.show()
```


    
![png](/posts_images/2018-02-08-data-quest-guided-project-analyzing-movie-reviews/output_3_0.png)
    



    
![png](/posts_images/2018-02-08-data-quest-guided-project-analyzing-movie-reviews/output_3_1.png)
    


It looks like fandango seems to have higher overalll ratings than metacritic, but just looking at histograms isn't enough to prove that. We can calclate the mean, median, and standard deviation of the two websites using numpy functions.  


```python
mean_fd = fd.mean()
mean_mc = mc.mean()
median_fd = fd.median()
median_mc = mc.median()
std_fd = fd.std()
std_mc = mc.std()

print("means", mean_fd, mean_mc)
print("medians",median_fd, median_mc)
print("std_devs",std_fd, std_mc)
```

    means 4.08904109589 2.97260273973
    medians 4.0 3.0
    std_devs 0.540385977979 0.990960561374
    

Couple of things to note here:

+ Fandango rating methods are hidden, where as metacritic takes a weighted average of all the published critic scores.

+ The mean and the median for fandango is way higher, they also got a low std deviation. I'd imagine their scores are influenced by studios and have inflated scores to get people on the website to watch the movies.

+ The standard deviation for fandango is also lower because most of their ratings are clustered on the high side.

+ Metacritic on the other hand has a median of 3.0 and an average of 3 which is basically what you would expect from a normal distribution.

Let's make a scatter plot between fandango and metacritic to see if we can draw any correlations.


```python
plt.scatter(fd, mc)
plt.show()
```


    
![png](/posts_images/2018-02-08-data-quest-guided-project-analyzing-movie-reviews/output_8_0.png)
    



```python
movies['fm_diff'] = fd - mc
movies['fm_diff'] = np.absolute(movies['fm_diff'])
dif_sort = movies['fm_diff'].sort_values(ascending=False)

movies.sort_values(by='fm_diff', ascending = False).head(5)
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
      <th>FILM</th>
      <th>RottenTomatoes</th>
      <th>RottenTomatoes_User</th>
      <th>Metacritic</th>
      <th>Metacritic_User</th>
      <th>IMDB</th>
      <th>Fandango_Stars</th>
      <th>Fandango_Ratingvalue</th>
      <th>RT_norm</th>
      <th>RT_user_norm</th>
      <th>...</th>
      <th>RT_norm_round</th>
      <th>RT_user_norm_round</th>
      <th>Metacritic_norm_round</th>
      <th>Metacritic_user_norm_round</th>
      <th>IMDB_norm_round</th>
      <th>Metacritic_user_vote_count</th>
      <th>IMDB_user_vote_count</th>
      <th>Fandango_votes</th>
      <th>Fandango_Difference</th>
      <th>fm_diff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>Do You Believe? (2015)</td>
      <td>18</td>
      <td>84</td>
      <td>22</td>
      <td>4.7</td>
      <td>5.4</td>
      <td>5.0</td>
      <td>4.5</td>
      <td>0.90</td>
      <td>4.20</td>
      <td>...</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>2.5</td>
      <td>2.5</td>
      <td>31</td>
      <td>3136</td>
      <td>1793</td>
      <td>0.5</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>85</th>
      <td>Little Boy (2015)</td>
      <td>20</td>
      <td>81</td>
      <td>30</td>
      <td>5.9</td>
      <td>7.4</td>
      <td>4.5</td>
      <td>4.3</td>
      <td>1.00</td>
      <td>4.05</td>
      <td>...</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>1.5</td>
      <td>3.0</td>
      <td>3.5</td>
      <td>38</td>
      <td>5927</td>
      <td>811</td>
      <td>0.2</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>47</th>
      <td>Annie (2014)</td>
      <td>27</td>
      <td>61</td>
      <td>33</td>
      <td>4.8</td>
      <td>5.2</td>
      <td>4.5</td>
      <td>4.2</td>
      <td>1.35</td>
      <td>3.05</td>
      <td>...</td>
      <td>1.5</td>
      <td>3.0</td>
      <td>1.5</td>
      <td>2.5</td>
      <td>2.5</td>
      <td>108</td>
      <td>19222</td>
      <td>6835</td>
      <td>0.3</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Pixels (2015)</td>
      <td>17</td>
      <td>54</td>
      <td>27</td>
      <td>5.3</td>
      <td>5.6</td>
      <td>4.5</td>
      <td>4.1</td>
      <td>0.85</td>
      <td>2.70</td>
      <td>...</td>
      <td>1.0</td>
      <td>2.5</td>
      <td>1.5</td>
      <td>2.5</td>
      <td>3.0</td>
      <td>246</td>
      <td>19521</td>
      <td>3886</td>
      <td>0.4</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>134</th>
      <td>The Longest Ride (2015)</td>
      <td>31</td>
      <td>73</td>
      <td>33</td>
      <td>4.8</td>
      <td>7.2</td>
      <td>4.5</td>
      <td>4.5</td>
      <td>1.55</td>
      <td>3.65</td>
      <td>...</td>
      <td>1.5</td>
      <td>3.5</td>
      <td>1.5</td>
      <td>2.5</td>
      <td>3.5</td>
      <td>49</td>
      <td>25214</td>
      <td>2603</td>
      <td>0.0</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>



It looks like the difference can get as high as 4.0 or 3.0. We should try to calculate the correlation between the two websites. We can do this by simply using the .pearsonr() function from scipy.


```python
import scipy.stats as sci

r, pearsonr = sci.pearsonr(mc, fd)
print(r)
print(pearsonr)
```

    0.178449190739
    0.0311615162285
    

If both movie review sites uses the similar methods for rating their movies, we should see a strong correlation. A low correlation tells us that these two websites have very different review methods.

Doing a linear regression wouldn't be very accurate with a low correlation. However, let's do it for the sake of practice anyway.


```python
m, b, r, p, stderr = sci.linregress(mc, fd)

#Fit into a line, y = mx+b where x is 3.
pred_3 = m*3 + b
pred_3
```




    4.0917071528212041




```python
pred_1 = m*1 + b
print(pred_1)
pred_5 = m*5 + b
print(pred_5)
```

    3.89708499687
    4.28632930877
    

We can make predictions of what the fandango score is based on the metacritic score by doing a linear regression. However it is important to keep in mind, if the correlation is low, the model might not be very accurate.


```python
x_pred = [1.0, 5.0]
y_pred = [3.89708499687, 4.28632930877]

plt.scatter(fd, mc)
plt.plot(x_pred, y_pred)



plt.show()
```


    
![png](/posts_images/2018-02-08-data-quest-guided-project-analyzing-movie-reviews/output_17_0.png)
    


---

#### Learning Summary

Concepts explored: pandas, descriptive statistics, numpy, matplotlib, scipy, correlations

Functions and methods used: .sort_values(), sci.linregress(), .hist(), .absolute(), .mean(), .median(), .absolute()

The files used for this project can be found in my [GitHub repository](https://github.com/sengkchu/Dataquest-Guided-Projects-Solutions/tree/master/Guided%20Project_%20Analyzing%20Movie%20Reviews).

