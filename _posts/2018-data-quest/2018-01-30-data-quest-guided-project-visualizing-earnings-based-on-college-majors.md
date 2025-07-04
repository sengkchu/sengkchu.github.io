---
title: Dataquest Guided Project - Visualizing Earnings Based On College Majors
date: 2018-01-30 00:00 +/-0000
categories: [DataQuest]
tags: [dataquest]
image:
  path: /posts_images/2018-01-30-data-quest-guided-project-visualizing-earnings-based-on-college-majors/cover.PNG
---


In this project we will look at earnings from recent college graduates based on each major in 'recent-grads.csv'. We'll visualize the data using histograms, bar charts, and scatter plots and see if we can draw any interesting insights from it. However, the main purpose of this project is to practice some of the data visualization tools.


```python
import pandas as pd
import matplotlib as plt

#jupyter magic so the plots are displayed inline
%matplotlib inline
```


```python
recent_grads = pd.read_csv('recent-grads.csv')
recent_grads.iloc[0]
```




    Rank                                        1
    Major_code                               2419
    Major                   PETROLEUM ENGINEERING
    Total                                    2339
    Men                                      2057
    Women                                     282
    Major_category                    Engineering
    ShareWomen                           0.120564
    Sample_size                                36
    Employed                                 1976
    Full_time                                1849
    Part_time                                 270
    Full_time_year_round                     1207
    Unemployed                                 37
    Unemployment_rate                   0.0183805
    Median                                 110000
    P25th                                   95000
    P75th                                  125000
    College_jobs                             1534
    Non_college_jobs                          364
    Low_wage_jobs                             193
    Name: 0, dtype: object




```python
recent_grads.head(1)
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
      <th>Rank</th>
      <th>Major_code</th>
      <th>Major</th>
      <th>Total</th>
      <th>Men</th>
      <th>Women</th>
      <th>Major_category</th>
      <th>ShareWomen</th>
      <th>Sample_size</th>
      <th>Employed</th>
      <th>...</th>
      <th>Part_time</th>
      <th>Full_time_year_round</th>
      <th>Unemployed</th>
      <th>Unemployment_rate</th>
      <th>Median</th>
      <th>P25th</th>
      <th>P75th</th>
      <th>College_jobs</th>
      <th>Non_college_jobs</th>
      <th>Low_wage_jobs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2419</td>
      <td>PETROLEUM ENGINEERING</td>
      <td>2339.0</td>
      <td>2057.0</td>
      <td>282.0</td>
      <td>Engineering</td>
      <td>0.120564</td>
      <td>36</td>
      <td>1976</td>
      <td>...</td>
      <td>270</td>
      <td>1207</td>
      <td>37</td>
      <td>0.018381</td>
      <td>110000</td>
      <td>95000</td>
      <td>125000</td>
      <td>1534</td>
      <td>364</td>
      <td>193</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 21 columns</p>
</div>




```python
recent_grads.tail(1)
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
      <th>Rank</th>
      <th>Major_code</th>
      <th>Major</th>
      <th>Total</th>
      <th>Men</th>
      <th>Women</th>
      <th>Major_category</th>
      <th>ShareWomen</th>
      <th>Sample_size</th>
      <th>Employed</th>
      <th>...</th>
      <th>Part_time</th>
      <th>Full_time_year_round</th>
      <th>Unemployed</th>
      <th>Unemployment_rate</th>
      <th>Median</th>
      <th>P25th</th>
      <th>P75th</th>
      <th>College_jobs</th>
      <th>Non_college_jobs</th>
      <th>Low_wage_jobs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>172</th>
      <td>173</td>
      <td>3501</td>
      <td>LIBRARY SCIENCE</td>
      <td>1098.0</td>
      <td>134.0</td>
      <td>964.0</td>
      <td>Education</td>
      <td>0.87796</td>
      <td>2</td>
      <td>742</td>
      <td>...</td>
      <td>237</td>
      <td>410</td>
      <td>87</td>
      <td>0.104946</td>
      <td>22000</td>
      <td>20000</td>
      <td>22000</td>
      <td>288</td>
      <td>338</td>
      <td>192</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 21 columns</p>
</div>




```python
recent_grads.describe()
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
      <th>Rank</th>
      <th>Major_code</th>
      <th>Total</th>
      <th>Men</th>
      <th>Women</th>
      <th>ShareWomen</th>
      <th>Sample_size</th>
      <th>Employed</th>
      <th>Full_time</th>
      <th>Part_time</th>
      <th>Full_time_year_round</th>
      <th>Unemployed</th>
      <th>Unemployment_rate</th>
      <th>Median</th>
      <th>P25th</th>
      <th>P75th</th>
      <th>College_jobs</th>
      <th>Non_college_jobs</th>
      <th>Low_wage_jobs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>173.000000</td>
      <td>173.000000</td>
      <td>172.000000</td>
      <td>172.000000</td>
      <td>172.000000</td>
      <td>172.000000</td>
      <td>173.000000</td>
      <td>173.000000</td>
      <td>173.000000</td>
      <td>173.000000</td>
      <td>173.000000</td>
      <td>173.000000</td>
      <td>173.000000</td>
      <td>173.000000</td>
      <td>173.000000</td>
      <td>173.000000</td>
      <td>173.000000</td>
      <td>173.000000</td>
      <td>173.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>87.000000</td>
      <td>3879.815029</td>
      <td>39370.081395</td>
      <td>16723.406977</td>
      <td>22646.674419</td>
      <td>0.522223</td>
      <td>356.080925</td>
      <td>31192.763006</td>
      <td>26029.306358</td>
      <td>8832.398844</td>
      <td>19694.427746</td>
      <td>2416.329480</td>
      <td>0.068191</td>
      <td>40151.445087</td>
      <td>29501.445087</td>
      <td>51494.219653</td>
      <td>12322.635838</td>
      <td>13284.497110</td>
      <td>3859.017341</td>
    </tr>
    <tr>
      <th>std</th>
      <td>50.084928</td>
      <td>1687.753140</td>
      <td>63483.491009</td>
      <td>28122.433474</td>
      <td>41057.330740</td>
      <td>0.231205</td>
      <td>618.361022</td>
      <td>50675.002241</td>
      <td>42869.655092</td>
      <td>14648.179473</td>
      <td>33160.941514</td>
      <td>4112.803148</td>
      <td>0.030331</td>
      <td>11470.181802</td>
      <td>9166.005235</td>
      <td>14906.279740</td>
      <td>21299.868863</td>
      <td>23789.655363</td>
      <td>6944.998579</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>1100.000000</td>
      <td>124.000000</td>
      <td>119.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>111.000000</td>
      <td>0.000000</td>
      <td>111.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>22000.000000</td>
      <td>18500.000000</td>
      <td>22000.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>44.000000</td>
      <td>2403.000000</td>
      <td>4549.750000</td>
      <td>2177.500000</td>
      <td>1778.250000</td>
      <td>0.336026</td>
      <td>39.000000</td>
      <td>3608.000000</td>
      <td>3154.000000</td>
      <td>1030.000000</td>
      <td>2453.000000</td>
      <td>304.000000</td>
      <td>0.050306</td>
      <td>33000.000000</td>
      <td>24000.000000</td>
      <td>42000.000000</td>
      <td>1675.000000</td>
      <td>1591.000000</td>
      <td>340.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>87.000000</td>
      <td>3608.000000</td>
      <td>15104.000000</td>
      <td>5434.000000</td>
      <td>8386.500000</td>
      <td>0.534024</td>
      <td>130.000000</td>
      <td>11797.000000</td>
      <td>10048.000000</td>
      <td>3299.000000</td>
      <td>7413.000000</td>
      <td>893.000000</td>
      <td>0.067961</td>
      <td>36000.000000</td>
      <td>27000.000000</td>
      <td>47000.000000</td>
      <td>4390.000000</td>
      <td>4595.000000</td>
      <td>1231.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>130.000000</td>
      <td>5503.000000</td>
      <td>38909.750000</td>
      <td>14631.000000</td>
      <td>22553.750000</td>
      <td>0.703299</td>
      <td>338.000000</td>
      <td>31433.000000</td>
      <td>25147.000000</td>
      <td>9948.000000</td>
      <td>16891.000000</td>
      <td>2393.000000</td>
      <td>0.087557</td>
      <td>45000.000000</td>
      <td>33000.000000</td>
      <td>60000.000000</td>
      <td>14444.000000</td>
      <td>11783.000000</td>
      <td>3466.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>173.000000</td>
      <td>6403.000000</td>
      <td>393735.000000</td>
      <td>173809.000000</td>
      <td>307087.000000</td>
      <td>0.968954</td>
      <td>4212.000000</td>
      <td>307933.000000</td>
      <td>251540.000000</td>
      <td>115172.000000</td>
      <td>199897.000000</td>
      <td>28169.000000</td>
      <td>0.177226</td>
      <td>110000.000000</td>
      <td>95000.000000</td>
      <td>125000.000000</td>
      <td>151643.000000</td>
      <td>148395.000000</td>
      <td>48207.000000</td>
    </tr>
  </tbody>
</table>
</div>



First, let's clean up the data a bit and drop the rows that have NaN as values.


```python
recent_grads = recent_grads.dropna()
recent_grads
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
      <th>Rank</th>
      <th>Major_code</th>
      <th>Major</th>
      <th>Total</th>
      <th>Men</th>
      <th>Women</th>
      <th>Major_category</th>
      <th>ShareWomen</th>
      <th>Sample_size</th>
      <th>Employed</th>
      <th>...</th>
      <th>Part_time</th>
      <th>Full_time_year_round</th>
      <th>Unemployed</th>
      <th>Unemployment_rate</th>
      <th>Median</th>
      <th>P25th</th>
      <th>P75th</th>
      <th>College_jobs</th>
      <th>Non_college_jobs</th>
      <th>Low_wage_jobs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2419</td>
      <td>PETROLEUM ENGINEERING</td>
      <td>2339.0</td>
      <td>2057.0</td>
      <td>282.0</td>
      <td>Engineering</td>
      <td>0.120564</td>
      <td>36</td>
      <td>1976</td>
      <td>...</td>
      <td>270</td>
      <td>1207</td>
      <td>37</td>
      <td>0.018381</td>
      <td>110000</td>
      <td>95000</td>
      <td>125000</td>
      <td>1534</td>
      <td>364</td>
      <td>193</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2416</td>
      <td>MINING AND MINERAL ENGINEERING</td>
      <td>756.0</td>
      <td>679.0</td>
      <td>77.0</td>
      <td>Engineering</td>
      <td>0.101852</td>
      <td>7</td>
      <td>640</td>
      <td>...</td>
      <td>170</td>
      <td>388</td>
      <td>85</td>
      <td>0.117241</td>
      <td>75000</td>
      <td>55000</td>
      <td>90000</td>
      <td>350</td>
      <td>257</td>
      <td>50</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>2415</td>
      <td>METALLURGICAL ENGINEERING</td>
      <td>856.0</td>
      <td>725.0</td>
      <td>131.0</td>
      <td>Engineering</td>
      <td>0.153037</td>
      <td>3</td>
      <td>648</td>
      <td>...</td>
      <td>133</td>
      <td>340</td>
      <td>16</td>
      <td>0.024096</td>
      <td>73000</td>
      <td>50000</td>
      <td>105000</td>
      <td>456</td>
      <td>176</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>2417</td>
      <td>NAVAL ARCHITECTURE AND MARINE ENGINEERING</td>
      <td>1258.0</td>
      <td>1123.0</td>
      <td>135.0</td>
      <td>Engineering</td>
      <td>0.107313</td>
      <td>16</td>
      <td>758</td>
      <td>...</td>
      <td>150</td>
      <td>692</td>
      <td>40</td>
      <td>0.050125</td>
      <td>70000</td>
      <td>43000</td>
      <td>80000</td>
      <td>529</td>
      <td>102</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>2405</td>
      <td>CHEMICAL ENGINEERING</td>
      <td>32260.0</td>
      <td>21239.0</td>
      <td>11021.0</td>
      <td>Engineering</td>
      <td>0.341631</td>
      <td>289</td>
      <td>25694</td>
      <td>...</td>
      <td>5180</td>
      <td>16697</td>
      <td>1672</td>
      <td>0.061098</td>
      <td>65000</td>
      <td>50000</td>
      <td>75000</td>
      <td>18314</td>
      <td>4440</td>
      <td>972</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>2418</td>
      <td>NUCLEAR ENGINEERING</td>
      <td>2573.0</td>
      <td>2200.0</td>
      <td>373.0</td>
      <td>Engineering</td>
      <td>0.144967</td>
      <td>17</td>
      <td>1857</td>
      <td>...</td>
      <td>264</td>
      <td>1449</td>
      <td>400</td>
      <td>0.177226</td>
      <td>65000</td>
      <td>50000</td>
      <td>102000</td>
      <td>1142</td>
      <td>657</td>
      <td>244</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>6202</td>
      <td>ACTUARIAL SCIENCE</td>
      <td>3777.0</td>
      <td>2110.0</td>
      <td>1667.0</td>
      <td>Business</td>
      <td>0.441356</td>
      <td>51</td>
      <td>2912</td>
      <td>...</td>
      <td>296</td>
      <td>2482</td>
      <td>308</td>
      <td>0.095652</td>
      <td>62000</td>
      <td>53000</td>
      <td>72000</td>
      <td>1768</td>
      <td>314</td>
      <td>259</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>5001</td>
      <td>ASTRONOMY AND ASTROPHYSICS</td>
      <td>1792.0</td>
      <td>832.0</td>
      <td>960.0</td>
      <td>Physical Sciences</td>
      <td>0.535714</td>
      <td>10</td>
      <td>1526</td>
      <td>...</td>
      <td>553</td>
      <td>827</td>
      <td>33</td>
      <td>0.021167</td>
      <td>62000</td>
      <td>31500</td>
      <td>109000</td>
      <td>972</td>
      <td>500</td>
      <td>220</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>2414</td>
      <td>MECHANICAL ENGINEERING</td>
      <td>91227.0</td>
      <td>80320.0</td>
      <td>10907.0</td>
      <td>Engineering</td>
      <td>0.119559</td>
      <td>1029</td>
      <td>76442</td>
      <td>...</td>
      <td>13101</td>
      <td>54639</td>
      <td>4650</td>
      <td>0.057342</td>
      <td>60000</td>
      <td>48000</td>
      <td>70000</td>
      <td>52844</td>
      <td>16384</td>
      <td>3253</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>2408</td>
      <td>ELECTRICAL ENGINEERING</td>
      <td>81527.0</td>
      <td>65511.0</td>
      <td>16016.0</td>
      <td>Engineering</td>
      <td>0.196450</td>
      <td>631</td>
      <td>61928</td>
      <td>...</td>
      <td>12695</td>
      <td>41413</td>
      <td>3895</td>
      <td>0.059174</td>
      <td>60000</td>
      <td>45000</td>
      <td>72000</td>
      <td>45829</td>
      <td>10874</td>
      <td>3170</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>2407</td>
      <td>COMPUTER ENGINEERING</td>
      <td>41542.0</td>
      <td>33258.0</td>
      <td>8284.0</td>
      <td>Engineering</td>
      <td>0.199413</td>
      <td>399</td>
      <td>32506</td>
      <td>...</td>
      <td>5146</td>
      <td>23621</td>
      <td>2275</td>
      <td>0.065409</td>
      <td>60000</td>
      <td>45000</td>
      <td>75000</td>
      <td>23694</td>
      <td>5721</td>
      <td>980</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>2401</td>
      <td>AEROSPACE ENGINEERING</td>
      <td>15058.0</td>
      <td>12953.0</td>
      <td>2105.0</td>
      <td>Engineering</td>
      <td>0.139793</td>
      <td>147</td>
      <td>11391</td>
      <td>...</td>
      <td>2724</td>
      <td>8790</td>
      <td>794</td>
      <td>0.065162</td>
      <td>60000</td>
      <td>42000</td>
      <td>70000</td>
      <td>8184</td>
      <td>2425</td>
      <td>372</td>
    </tr>
    <tr>
      <th>12</th>
      <td>13</td>
      <td>2404</td>
      <td>BIOMEDICAL ENGINEERING</td>
      <td>14955.0</td>
      <td>8407.0</td>
      <td>6548.0</td>
      <td>Engineering</td>
      <td>0.437847</td>
      <td>79</td>
      <td>10047</td>
      <td>...</td>
      <td>2694</td>
      <td>5986</td>
      <td>1019</td>
      <td>0.092084</td>
      <td>60000</td>
      <td>36000</td>
      <td>70000</td>
      <td>6439</td>
      <td>2471</td>
      <td>789</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14</td>
      <td>5008</td>
      <td>MATERIALS SCIENCE</td>
      <td>4279.0</td>
      <td>2949.0</td>
      <td>1330.0</td>
      <td>Engineering</td>
      <td>0.310820</td>
      <td>22</td>
      <td>3307</td>
      <td>...</td>
      <td>878</td>
      <td>1967</td>
      <td>78</td>
      <td>0.023043</td>
      <td>60000</td>
      <td>39000</td>
      <td>65000</td>
      <td>2626</td>
      <td>391</td>
      <td>81</td>
    </tr>
    <tr>
      <th>14</th>
      <td>15</td>
      <td>2409</td>
      <td>ENGINEERING MECHANICS PHYSICS AND SCIENCE</td>
      <td>4321.0</td>
      <td>3526.0</td>
      <td>795.0</td>
      <td>Engineering</td>
      <td>0.183985</td>
      <td>30</td>
      <td>3608</td>
      <td>...</td>
      <td>811</td>
      <td>2004</td>
      <td>23</td>
      <td>0.006334</td>
      <td>58000</td>
      <td>25000</td>
      <td>74000</td>
      <td>2439</td>
      <td>947</td>
      <td>263</td>
    </tr>
    <tr>
      <th>15</th>
      <td>16</td>
      <td>2402</td>
      <td>BIOLOGICAL ENGINEERING</td>
      <td>8925.0</td>
      <td>6062.0</td>
      <td>2863.0</td>
      <td>Engineering</td>
      <td>0.320784</td>
      <td>55</td>
      <td>6170</td>
      <td>...</td>
      <td>1983</td>
      <td>3413</td>
      <td>589</td>
      <td>0.087143</td>
      <td>57100</td>
      <td>40000</td>
      <td>76000</td>
      <td>3603</td>
      <td>1595</td>
      <td>524</td>
    </tr>
    <tr>
      <th>16</th>
      <td>17</td>
      <td>2412</td>
      <td>INDUSTRIAL AND MANUFACTURING ENGINEERING</td>
      <td>18968.0</td>
      <td>12453.0</td>
      <td>6515.0</td>
      <td>Engineering</td>
      <td>0.343473</td>
      <td>183</td>
      <td>15604</td>
      <td>...</td>
      <td>2243</td>
      <td>11326</td>
      <td>699</td>
      <td>0.042876</td>
      <td>57000</td>
      <td>37900</td>
      <td>67000</td>
      <td>8306</td>
      <td>3235</td>
      <td>640</td>
    </tr>
    <tr>
      <th>17</th>
      <td>18</td>
      <td>2400</td>
      <td>GENERAL ENGINEERING</td>
      <td>61152.0</td>
      <td>45683.0</td>
      <td>15469.0</td>
      <td>Engineering</td>
      <td>0.252960</td>
      <td>425</td>
      <td>44931</td>
      <td>...</td>
      <td>7199</td>
      <td>33540</td>
      <td>2859</td>
      <td>0.059824</td>
      <td>56000</td>
      <td>36000</td>
      <td>69000</td>
      <td>26898</td>
      <td>11734</td>
      <td>3192</td>
    </tr>
    <tr>
      <th>18</th>
      <td>19</td>
      <td>2403</td>
      <td>ARCHITECTURAL ENGINEERING</td>
      <td>2825.0</td>
      <td>1835.0</td>
      <td>990.0</td>
      <td>Engineering</td>
      <td>0.350442</td>
      <td>26</td>
      <td>2575</td>
      <td>...</td>
      <td>343</td>
      <td>1848</td>
      <td>170</td>
      <td>0.061931</td>
      <td>54000</td>
      <td>38000</td>
      <td>65000</td>
      <td>1665</td>
      <td>649</td>
      <td>137</td>
    </tr>
    <tr>
      <th>19</th>
      <td>20</td>
      <td>3201</td>
      <td>COURT REPORTING</td>
      <td>1148.0</td>
      <td>877.0</td>
      <td>271.0</td>
      <td>Law &amp; Public Policy</td>
      <td>0.236063</td>
      <td>14</td>
      <td>930</td>
      <td>...</td>
      <td>223</td>
      <td>808</td>
      <td>11</td>
      <td>0.011690</td>
      <td>54000</td>
      <td>50000</td>
      <td>54000</td>
      <td>402</td>
      <td>528</td>
      <td>144</td>
    </tr>
    <tr>
      <th>20</th>
      <td>21</td>
      <td>2102</td>
      <td>COMPUTER SCIENCE</td>
      <td>128319.0</td>
      <td>99743.0</td>
      <td>28576.0</td>
      <td>Computers &amp; Mathematics</td>
      <td>0.222695</td>
      <td>1196</td>
      <td>102087</td>
      <td>...</td>
      <td>18726</td>
      <td>70932</td>
      <td>6884</td>
      <td>0.063173</td>
      <td>53000</td>
      <td>39000</td>
      <td>70000</td>
      <td>68622</td>
      <td>25667</td>
      <td>5144</td>
    </tr>
    <tr>
      <th>22</th>
      <td>23</td>
      <td>2502</td>
      <td>ELECTRICAL ENGINEERING TECHNOLOGY</td>
      <td>11565.0</td>
      <td>8181.0</td>
      <td>3384.0</td>
      <td>Engineering</td>
      <td>0.292607</td>
      <td>97</td>
      <td>8587</td>
      <td>...</td>
      <td>1873</td>
      <td>5681</td>
      <td>824</td>
      <td>0.087557</td>
      <td>52000</td>
      <td>35000</td>
      <td>60000</td>
      <td>5126</td>
      <td>2686</td>
      <td>696</td>
    </tr>
    <tr>
      <th>23</th>
      <td>24</td>
      <td>2413</td>
      <td>MATERIALS ENGINEERING AND MATERIALS SCIENCE</td>
      <td>2993.0</td>
      <td>2020.0</td>
      <td>973.0</td>
      <td>Engineering</td>
      <td>0.325092</td>
      <td>22</td>
      <td>2449</td>
      <td>...</td>
      <td>1040</td>
      <td>1151</td>
      <td>70</td>
      <td>0.027789</td>
      <td>52000</td>
      <td>35000</td>
      <td>62000</td>
      <td>1911</td>
      <td>305</td>
      <td>70</td>
    </tr>
    <tr>
      <th>24</th>
      <td>25</td>
      <td>6212</td>
      <td>MANAGEMENT INFORMATION SYSTEMS AND STATISTICS</td>
      <td>18713.0</td>
      <td>13496.0</td>
      <td>5217.0</td>
      <td>Business</td>
      <td>0.278790</td>
      <td>278</td>
      <td>16413</td>
      <td>...</td>
      <td>2420</td>
      <td>13017</td>
      <td>1015</td>
      <td>0.058240</td>
      <td>51000</td>
      <td>38000</td>
      <td>60000</td>
      <td>6342</td>
      <td>5741</td>
      <td>708</td>
    </tr>
    <tr>
      <th>25</th>
      <td>26</td>
      <td>2406</td>
      <td>CIVIL ENGINEERING</td>
      <td>53153.0</td>
      <td>41081.0</td>
      <td>12072.0</td>
      <td>Engineering</td>
      <td>0.227118</td>
      <td>565</td>
      <td>43041</td>
      <td>...</td>
      <td>10080</td>
      <td>29196</td>
      <td>3270</td>
      <td>0.070610</td>
      <td>50000</td>
      <td>40000</td>
      <td>60000</td>
      <td>28526</td>
      <td>9356</td>
      <td>2899</td>
    </tr>
    <tr>
      <th>26</th>
      <td>27</td>
      <td>5601</td>
      <td>CONSTRUCTION SERVICES</td>
      <td>18498.0</td>
      <td>16820.0</td>
      <td>1678.0</td>
      <td>Industrial Arts &amp; Consumer Services</td>
      <td>0.090713</td>
      <td>295</td>
      <td>16318</td>
      <td>...</td>
      <td>1751</td>
      <td>12313</td>
      <td>1042</td>
      <td>0.060023</td>
      <td>50000</td>
      <td>36000</td>
      <td>60000</td>
      <td>3275</td>
      <td>5351</td>
      <td>703</td>
    </tr>
    <tr>
      <th>27</th>
      <td>28</td>
      <td>6204</td>
      <td>OPERATIONS LOGISTICS AND E-COMMERCE</td>
      <td>11732.0</td>
      <td>7921.0</td>
      <td>3811.0</td>
      <td>Business</td>
      <td>0.324838</td>
      <td>156</td>
      <td>10027</td>
      <td>...</td>
      <td>1183</td>
      <td>7724</td>
      <td>504</td>
      <td>0.047859</td>
      <td>50000</td>
      <td>40000</td>
      <td>60000</td>
      <td>1466</td>
      <td>3629</td>
      <td>285</td>
    </tr>
    <tr>
      <th>28</th>
      <td>29</td>
      <td>2499</td>
      <td>MISCELLANEOUS ENGINEERING</td>
      <td>9133.0</td>
      <td>7398.0</td>
      <td>1735.0</td>
      <td>Engineering</td>
      <td>0.189970</td>
      <td>118</td>
      <td>7428</td>
      <td>...</td>
      <td>1662</td>
      <td>5476</td>
      <td>597</td>
      <td>0.074393</td>
      <td>50000</td>
      <td>39000</td>
      <td>65000</td>
      <td>3445</td>
      <td>2426</td>
      <td>365</td>
    </tr>
    <tr>
      <th>29</th>
      <td>30</td>
      <td>5402</td>
      <td>PUBLIC POLICY</td>
      <td>5978.0</td>
      <td>2639.0</td>
      <td>3339.0</td>
      <td>Law &amp; Public Policy</td>
      <td>0.558548</td>
      <td>55</td>
      <td>4547</td>
      <td>...</td>
      <td>1306</td>
      <td>2776</td>
      <td>670</td>
      <td>0.128426</td>
      <td>50000</td>
      <td>35000</td>
      <td>70000</td>
      <td>1550</td>
      <td>1871</td>
      <td>340</td>
    </tr>
    <tr>
      <th>30</th>
      <td>31</td>
      <td>2410</td>
      <td>ENVIRONMENTAL ENGINEERING</td>
      <td>4047.0</td>
      <td>2662.0</td>
      <td>1385.0</td>
      <td>Engineering</td>
      <td>0.342229</td>
      <td>26</td>
      <td>2983</td>
      <td>...</td>
      <td>930</td>
      <td>1951</td>
      <td>308</td>
      <td>0.093589</td>
      <td>50000</td>
      <td>42000</td>
      <td>56000</td>
      <td>2028</td>
      <td>830</td>
      <td>260</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>143</th>
      <td>144</td>
      <td>1105</td>
      <td>PLANT SCIENCE AND AGRONOMY</td>
      <td>7416.0</td>
      <td>4897.0</td>
      <td>2519.0</td>
      <td>Agriculture &amp; Natural Resources</td>
      <td>0.339671</td>
      <td>110</td>
      <td>6594</td>
      <td>...</td>
      <td>1246</td>
      <td>4522</td>
      <td>314</td>
      <td>0.045455</td>
      <td>32000</td>
      <td>22900</td>
      <td>40000</td>
      <td>2089</td>
      <td>3545</td>
      <td>1231</td>
    </tr>
    <tr>
      <th>144</th>
      <td>145</td>
      <td>2308</td>
      <td>SCIENCE AND COMPUTER TEACHER EDUCATION</td>
      <td>6483.0</td>
      <td>2049.0</td>
      <td>4434.0</td>
      <td>Education</td>
      <td>0.683943</td>
      <td>59</td>
      <td>5362</td>
      <td>...</td>
      <td>1227</td>
      <td>3247</td>
      <td>266</td>
      <td>0.047264</td>
      <td>32000</td>
      <td>28000</td>
      <td>39000</td>
      <td>4214</td>
      <td>1106</td>
      <td>591</td>
    </tr>
    <tr>
      <th>145</th>
      <td>146</td>
      <td>5200</td>
      <td>PSYCHOLOGY</td>
      <td>393735.0</td>
      <td>86648.0</td>
      <td>307087.0</td>
      <td>Psychology &amp; Social Work</td>
      <td>0.779933</td>
      <td>2584</td>
      <td>307933</td>
      <td>...</td>
      <td>115172</td>
      <td>174438</td>
      <td>28169</td>
      <td>0.083811</td>
      <td>31500</td>
      <td>24000</td>
      <td>41000</td>
      <td>125148</td>
      <td>141860</td>
      <td>48207</td>
    </tr>
    <tr>
      <th>146</th>
      <td>147</td>
      <td>6002</td>
      <td>MUSIC</td>
      <td>60633.0</td>
      <td>29909.0</td>
      <td>30724.0</td>
      <td>Arts</td>
      <td>0.506721</td>
      <td>419</td>
      <td>47662</td>
      <td>...</td>
      <td>24943</td>
      <td>21425</td>
      <td>3918</td>
      <td>0.075960</td>
      <td>31000</td>
      <td>22300</td>
      <td>42000</td>
      <td>13752</td>
      <td>28786</td>
      <td>9286</td>
    </tr>
    <tr>
      <th>147</th>
      <td>148</td>
      <td>2306</td>
      <td>PHYSICAL AND HEALTH EDUCATION TEACHING</td>
      <td>28213.0</td>
      <td>15670.0</td>
      <td>12543.0</td>
      <td>Education</td>
      <td>0.444582</td>
      <td>259</td>
      <td>23794</td>
      <td>...</td>
      <td>7230</td>
      <td>13651</td>
      <td>1920</td>
      <td>0.074667</td>
      <td>31000</td>
      <td>24000</td>
      <td>40000</td>
      <td>12777</td>
      <td>9328</td>
      <td>2042</td>
    </tr>
    <tr>
      <th>148</th>
      <td>149</td>
      <td>6006</td>
      <td>ART HISTORY AND CRITICISM</td>
      <td>21030.0</td>
      <td>3240.0</td>
      <td>17790.0</td>
      <td>Humanities &amp; Liberal Arts</td>
      <td>0.845934</td>
      <td>204</td>
      <td>17579</td>
      <td>...</td>
      <td>6140</td>
      <td>9965</td>
      <td>1128</td>
      <td>0.060298</td>
      <td>31000</td>
      <td>23000</td>
      <td>40000</td>
      <td>5139</td>
      <td>9738</td>
      <td>3426</td>
    </tr>
    <tr>
      <th>149</th>
      <td>150</td>
      <td>6000</td>
      <td>FINE ARTS</td>
      <td>74440.0</td>
      <td>24786.0</td>
      <td>49654.0</td>
      <td>Arts</td>
      <td>0.667034</td>
      <td>623</td>
      <td>59679</td>
      <td>...</td>
      <td>23656</td>
      <td>31877</td>
      <td>5486</td>
      <td>0.084186</td>
      <td>30500</td>
      <td>21000</td>
      <td>41000</td>
      <td>20792</td>
      <td>32725</td>
      <td>11880</td>
    </tr>
    <tr>
      <th>150</th>
      <td>151</td>
      <td>2901</td>
      <td>FAMILY AND CONSUMER SCIENCES</td>
      <td>58001.0</td>
      <td>5166.0</td>
      <td>52835.0</td>
      <td>Industrial Arts &amp; Consumer Services</td>
      <td>0.910933</td>
      <td>518</td>
      <td>46624</td>
      <td>...</td>
      <td>15872</td>
      <td>26906</td>
      <td>3355</td>
      <td>0.067128</td>
      <td>30000</td>
      <td>22900</td>
      <td>40000</td>
      <td>20985</td>
      <td>20133</td>
      <td>5248</td>
    </tr>
    <tr>
      <th>151</th>
      <td>152</td>
      <td>5404</td>
      <td>SOCIAL WORK</td>
      <td>53552.0</td>
      <td>5137.0</td>
      <td>48415.0</td>
      <td>Psychology &amp; Social Work</td>
      <td>0.904075</td>
      <td>374</td>
      <td>45038</td>
      <td>...</td>
      <td>13481</td>
      <td>27588</td>
      <td>3329</td>
      <td>0.068828</td>
      <td>30000</td>
      <td>25000</td>
      <td>35000</td>
      <td>27449</td>
      <td>14416</td>
      <td>4344</td>
    </tr>
    <tr>
      <th>152</th>
      <td>153</td>
      <td>1103</td>
      <td>ANIMAL SCIENCES</td>
      <td>21573.0</td>
      <td>5347.0</td>
      <td>16226.0</td>
      <td>Agriculture &amp; Natural Resources</td>
      <td>0.752144</td>
      <td>255</td>
      <td>17112</td>
      <td>...</td>
      <td>5353</td>
      <td>10824</td>
      <td>917</td>
      <td>0.050862</td>
      <td>30000</td>
      <td>22000</td>
      <td>40000</td>
      <td>5443</td>
      <td>9571</td>
      <td>2125</td>
    </tr>
    <tr>
      <th>153</th>
      <td>154</td>
      <td>6003</td>
      <td>VISUAL AND PERFORMING ARTS</td>
      <td>16250.0</td>
      <td>4133.0</td>
      <td>12117.0</td>
      <td>Arts</td>
      <td>0.745662</td>
      <td>132</td>
      <td>12870</td>
      <td>...</td>
      <td>6253</td>
      <td>6322</td>
      <td>1465</td>
      <td>0.102197</td>
      <td>30000</td>
      <td>22000</td>
      <td>40000</td>
      <td>3849</td>
      <td>7635</td>
      <td>2840</td>
    </tr>
    <tr>
      <th>154</th>
      <td>155</td>
      <td>2312</td>
      <td>TEACHER EDUCATION: MULTIPLE LEVELS</td>
      <td>14443.0</td>
      <td>2734.0</td>
      <td>11709.0</td>
      <td>Education</td>
      <td>0.810704</td>
      <td>142</td>
      <td>13076</td>
      <td>...</td>
      <td>2214</td>
      <td>8457</td>
      <td>496</td>
      <td>0.036546</td>
      <td>30000</td>
      <td>24000</td>
      <td>37000</td>
      <td>10766</td>
      <td>1949</td>
      <td>722</td>
    </tr>
    <tr>
      <th>155</th>
      <td>156</td>
      <td>5299</td>
      <td>MISCELLANEOUS PSYCHOLOGY</td>
      <td>9628.0</td>
      <td>1936.0</td>
      <td>7692.0</td>
      <td>Psychology &amp; Social Work</td>
      <td>0.798920</td>
      <td>60</td>
      <td>7653</td>
      <td>...</td>
      <td>3221</td>
      <td>3838</td>
      <td>419</td>
      <td>0.051908</td>
      <td>30000</td>
      <td>20800</td>
      <td>40000</td>
      <td>2960</td>
      <td>3948</td>
      <td>1650</td>
    </tr>
    <tr>
      <th>156</th>
      <td>157</td>
      <td>5403</td>
      <td>HUMAN SERVICES AND COMMUNITY ORGANIZATION</td>
      <td>9374.0</td>
      <td>885.0</td>
      <td>8489.0</td>
      <td>Psychology &amp; Social Work</td>
      <td>0.905590</td>
      <td>89</td>
      <td>8294</td>
      <td>...</td>
      <td>2405</td>
      <td>5061</td>
      <td>326</td>
      <td>0.037819</td>
      <td>30000</td>
      <td>24000</td>
      <td>35000</td>
      <td>2878</td>
      <td>4595</td>
      <td>724</td>
    </tr>
    <tr>
      <th>157</th>
      <td>158</td>
      <td>3402</td>
      <td>HUMANITIES</td>
      <td>6652.0</td>
      <td>2013.0</td>
      <td>4639.0</td>
      <td>Humanities &amp; Liberal Arts</td>
      <td>0.697384</td>
      <td>49</td>
      <td>5052</td>
      <td>...</td>
      <td>2225</td>
      <td>2661</td>
      <td>372</td>
      <td>0.068584</td>
      <td>30000</td>
      <td>20000</td>
      <td>49000</td>
      <td>1168</td>
      <td>3354</td>
      <td>1141</td>
    </tr>
    <tr>
      <th>158</th>
      <td>159</td>
      <td>4901</td>
      <td>THEOLOGY AND RELIGIOUS VOCATIONS</td>
      <td>30207.0</td>
      <td>18616.0</td>
      <td>11591.0</td>
      <td>Humanities &amp; Liberal Arts</td>
      <td>0.383719</td>
      <td>310</td>
      <td>24202</td>
      <td>...</td>
      <td>8767</td>
      <td>13944</td>
      <td>1617</td>
      <td>0.062628</td>
      <td>29000</td>
      <td>22000</td>
      <td>38000</td>
      <td>9927</td>
      <td>12037</td>
      <td>3304</td>
    </tr>
    <tr>
      <th>159</th>
      <td>160</td>
      <td>6007</td>
      <td>STUDIO ARTS</td>
      <td>16977.0</td>
      <td>4754.0</td>
      <td>12223.0</td>
      <td>Arts</td>
      <td>0.719974</td>
      <td>182</td>
      <td>13908</td>
      <td>...</td>
      <td>5673</td>
      <td>7413</td>
      <td>1368</td>
      <td>0.089552</td>
      <td>29000</td>
      <td>19200</td>
      <td>38300</td>
      <td>3948</td>
      <td>8707</td>
      <td>3586</td>
    </tr>
    <tr>
      <th>160</th>
      <td>161</td>
      <td>2201</td>
      <td>COSMETOLOGY SERVICES AND CULINARY ARTS</td>
      <td>10510.0</td>
      <td>4364.0</td>
      <td>6146.0</td>
      <td>Industrial Arts &amp; Consumer Services</td>
      <td>0.584776</td>
      <td>117</td>
      <td>8650</td>
      <td>...</td>
      <td>2064</td>
      <td>5949</td>
      <td>510</td>
      <td>0.055677</td>
      <td>29000</td>
      <td>20000</td>
      <td>36000</td>
      <td>563</td>
      <td>7384</td>
      <td>3163</td>
    </tr>
    <tr>
      <th>161</th>
      <td>162</td>
      <td>1199</td>
      <td>MISCELLANEOUS AGRICULTURE</td>
      <td>1488.0</td>
      <td>404.0</td>
      <td>1084.0</td>
      <td>Agriculture &amp; Natural Resources</td>
      <td>0.728495</td>
      <td>24</td>
      <td>1290</td>
      <td>...</td>
      <td>335</td>
      <td>936</td>
      <td>82</td>
      <td>0.059767</td>
      <td>29000</td>
      <td>23000</td>
      <td>42100</td>
      <td>483</td>
      <td>626</td>
      <td>31</td>
    </tr>
    <tr>
      <th>162</th>
      <td>163</td>
      <td>5502</td>
      <td>ANTHROPOLOGY AND ARCHEOLOGY</td>
      <td>38844.0</td>
      <td>11376.0</td>
      <td>27468.0</td>
      <td>Humanities &amp; Liberal Arts</td>
      <td>0.707136</td>
      <td>247</td>
      <td>29633</td>
      <td>...</td>
      <td>14515</td>
      <td>13232</td>
      <td>3395</td>
      <td>0.102792</td>
      <td>28000</td>
      <td>20000</td>
      <td>38000</td>
      <td>9805</td>
      <td>16693</td>
      <td>6866</td>
    </tr>
    <tr>
      <th>163</th>
      <td>164</td>
      <td>6102</td>
      <td>COMMUNICATION DISORDERS SCIENCES AND SERVICES</td>
      <td>38279.0</td>
      <td>1225.0</td>
      <td>37054.0</td>
      <td>Health</td>
      <td>0.967998</td>
      <td>95</td>
      <td>29763</td>
      <td>...</td>
      <td>13862</td>
      <td>14460</td>
      <td>1487</td>
      <td>0.047584</td>
      <td>28000</td>
      <td>20000</td>
      <td>40000</td>
      <td>19957</td>
      <td>9404</td>
      <td>5125</td>
    </tr>
    <tr>
      <th>164</th>
      <td>165</td>
      <td>2307</td>
      <td>EARLY CHILDHOOD EDUCATION</td>
      <td>37589.0</td>
      <td>1167.0</td>
      <td>36422.0</td>
      <td>Education</td>
      <td>0.968954</td>
      <td>342</td>
      <td>32551</td>
      <td>...</td>
      <td>7001</td>
      <td>20748</td>
      <td>1360</td>
      <td>0.040105</td>
      <td>28000</td>
      <td>21000</td>
      <td>35000</td>
      <td>23515</td>
      <td>7705</td>
      <td>2868</td>
    </tr>
    <tr>
      <th>165</th>
      <td>166</td>
      <td>2603</td>
      <td>OTHER FOREIGN LANGUAGES</td>
      <td>11204.0</td>
      <td>3472.0</td>
      <td>7732.0</td>
      <td>Humanities &amp; Liberal Arts</td>
      <td>0.690111</td>
      <td>56</td>
      <td>7052</td>
      <td>...</td>
      <td>3685</td>
      <td>3214</td>
      <td>846</td>
      <td>0.107116</td>
      <td>27500</td>
      <td>22900</td>
      <td>38000</td>
      <td>2326</td>
      <td>3703</td>
      <td>1115</td>
    </tr>
    <tr>
      <th>166</th>
      <td>167</td>
      <td>6001</td>
      <td>DRAMA AND THEATER ARTS</td>
      <td>43249.0</td>
      <td>14440.0</td>
      <td>28809.0</td>
      <td>Arts</td>
      <td>0.666119</td>
      <td>357</td>
      <td>36165</td>
      <td>...</td>
      <td>15994</td>
      <td>16891</td>
      <td>3040</td>
      <td>0.077541</td>
      <td>27000</td>
      <td>19200</td>
      <td>35000</td>
      <td>6994</td>
      <td>25313</td>
      <td>11068</td>
    </tr>
    <tr>
      <th>167</th>
      <td>168</td>
      <td>3302</td>
      <td>COMPOSITION AND RHETORIC</td>
      <td>18953.0</td>
      <td>7022.0</td>
      <td>11931.0</td>
      <td>Humanities &amp; Liberal Arts</td>
      <td>0.629505</td>
      <td>151</td>
      <td>15053</td>
      <td>...</td>
      <td>6612</td>
      <td>7832</td>
      <td>1340</td>
      <td>0.081742</td>
      <td>27000</td>
      <td>20000</td>
      <td>35000</td>
      <td>4855</td>
      <td>8100</td>
      <td>3466</td>
    </tr>
    <tr>
      <th>168</th>
      <td>169</td>
      <td>3609</td>
      <td>ZOOLOGY</td>
      <td>8409.0</td>
      <td>3050.0</td>
      <td>5359.0</td>
      <td>Biology &amp; Life Science</td>
      <td>0.637293</td>
      <td>47</td>
      <td>6259</td>
      <td>...</td>
      <td>2190</td>
      <td>3602</td>
      <td>304</td>
      <td>0.046320</td>
      <td>26000</td>
      <td>20000</td>
      <td>39000</td>
      <td>2771</td>
      <td>2947</td>
      <td>743</td>
    </tr>
    <tr>
      <th>169</th>
      <td>170</td>
      <td>5201</td>
      <td>EDUCATIONAL PSYCHOLOGY</td>
      <td>2854.0</td>
      <td>522.0</td>
      <td>2332.0</td>
      <td>Psychology &amp; Social Work</td>
      <td>0.817099</td>
      <td>7</td>
      <td>2125</td>
      <td>...</td>
      <td>572</td>
      <td>1211</td>
      <td>148</td>
      <td>0.065112</td>
      <td>25000</td>
      <td>24000</td>
      <td>34000</td>
      <td>1488</td>
      <td>615</td>
      <td>82</td>
    </tr>
    <tr>
      <th>170</th>
      <td>171</td>
      <td>5202</td>
      <td>CLINICAL PSYCHOLOGY</td>
      <td>2838.0</td>
      <td>568.0</td>
      <td>2270.0</td>
      <td>Psychology &amp; Social Work</td>
      <td>0.799859</td>
      <td>13</td>
      <td>2101</td>
      <td>...</td>
      <td>648</td>
      <td>1293</td>
      <td>368</td>
      <td>0.149048</td>
      <td>25000</td>
      <td>25000</td>
      <td>40000</td>
      <td>986</td>
      <td>870</td>
      <td>622</td>
    </tr>
    <tr>
      <th>171</th>
      <td>172</td>
      <td>5203</td>
      <td>COUNSELING PSYCHOLOGY</td>
      <td>4626.0</td>
      <td>931.0</td>
      <td>3695.0</td>
      <td>Psychology &amp; Social Work</td>
      <td>0.798746</td>
      <td>21</td>
      <td>3777</td>
      <td>...</td>
      <td>965</td>
      <td>2738</td>
      <td>214</td>
      <td>0.053621</td>
      <td>23400</td>
      <td>19200</td>
      <td>26000</td>
      <td>2403</td>
      <td>1245</td>
      <td>308</td>
    </tr>
    <tr>
      <th>172</th>
      <td>173</td>
      <td>3501</td>
      <td>LIBRARY SCIENCE</td>
      <td>1098.0</td>
      <td>134.0</td>
      <td>964.0</td>
      <td>Education</td>
      <td>0.877960</td>
      <td>2</td>
      <td>742</td>
      <td>...</td>
      <td>237</td>
      <td>410</td>
      <td>87</td>
      <td>0.104946</td>
      <td>22000</td>
      <td>20000</td>
      <td>22000</td>
      <td>288</td>
      <td>338</td>
      <td>192</td>
    </tr>
  </tbody>
</table>
<p>172 rows × 21 columns</p>
</div>



Let's begin exploring the data using scatter plots and see if we can draw any interesting correlations.


```python
recent_grads.plot(x='Sample_size', y='Median', kind = 'scatter')
recent_grads.plot(x='Sample_size', y='Unemployment_rate', kind = 'scatter')
recent_grads.plot(x='Full_time', y='Median', kind = 'scatter')
recent_grads.plot(x='ShareWomen', y='Unemployment_rate', kind = 'scatter')
recent_grads.plot(x='Men', y='Median', kind = 'scatter')
recent_grads.plot(x='Women', y='Median', kind = 'scatter')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x14dae4fb710>




    
![png](/posts_images/2018-01-30-data-quest-guided-project-visualizing-earnings-based-on-college-majors/output_9_1.png)
    



    
![png](/posts_images/2018-01-30-data-quest-guided-project-visualizing-earnings-based-on-college-majors/output_9_2.png)
    



    
![png](/posts_images/2018-01-30-data-quest-guided-project-visualizing-earnings-based-on-college-majors/output_9_3.png)
    



    
![png](/posts_images/2018-01-30-data-quest-guided-project-visualizing-earnings-based-on-college-majors/output_9_4.png)
    



    
![png](/posts_images/2018-01-30-data-quest-guided-project-visualizing-earnings-based-on-college-majors/output_9_5.png)
    



    
![png](/posts_images/2018-01-30-data-quest-guided-project-visualizing-earnings-based-on-college-majors/output_9_6.png)
    


From the 'Unemployment_rate' vs. 'ShareWomen' plot, it looks like there is no correlation between unemployment rate and the amount of women in the major.

Doesn't look like there is much other useful information from these scatter plots, let's explore the data a bit further using histograms instead.

The y axis shows the frequency of the data and the x axis refers to the column name specified in code.


```python
recent_grads['Median'].hist(bins=25)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x14dae502a90>




    
![png](/posts_images/2018-01-30-data-quest-guided-project-visualizing-earnings-based-on-college-majors/output_11_1.png)
    



```python
recent_grads['Employed'].hist(bins=25)

```




    <matplotlib.axes._subplots.AxesSubplot at 0x14dae4c74e0>




    
![png](/posts_images/2018-01-30-data-quest-guided-project-visualizing-earnings-based-on-college-majors/output_12_1.png)
    



```python
recent_grads['Full_time'].hist(bins=25)

```




    <matplotlib.axes._subplots.AxesSubplot at 0x14dae7e7c50>




    
![png](/posts_images/2018-01-30-data-quest-guided-project-visualizing-earnings-based-on-college-majors/output_13_1.png)
    



```python
recent_grads['ShareWomen'].hist(bins=25)

```




    <matplotlib.axes._subplots.AxesSubplot at 0x14dae843cf8>




    
![png](/posts_images/2018-01-30-data-quest-guided-project-visualizing-earnings-based-on-college-majors/output_14_1.png)
    



```python
recent_grads['Unemployment_rate'].hist(bins=25)

```




    <matplotlib.axes._subplots.AxesSubplot at 0x14dae4abf28>




    
![png](/posts_images/2018-01-30-data-quest-guided-project-visualizing-earnings-based-on-college-majors/output_15_1.png)
    



```python
recent_grads['Men'].hist(bins=25)

```




    <matplotlib.axes._subplots.AxesSubplot at 0x14dae77a978>




    
![png](/posts_images/2018-01-30-data-quest-guided-project-visualizing-earnings-based-on-college-majors/output_16_1.png)
    



```python
recent_grads['Women'].hist(bins=25)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x14dae45f518>




    
![png](/posts_images/2018-01-30-data-quest-guided-project-visualizing-earnings-based-on-college-majors/output_17_1.png)
    


Again, not much correlation from these histograms. We do see a distribution of unemployment rates for various majors. If unemployment rate is not related to major, then we should see a wide plateau on the histogram.

Next we'll use scatter matrix from pandas to see if we can draw more insight. A scatter matrix can plot many different variables together and allow us to quickly see if there are correlations between those variables.


```python
from pandas.plotting import scatter_matrix
```


```python
scatter_matrix(recent_grads[['Sample_size', 'Median']], figsize=(10,10))
```




    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x0000014DAE8F52E8>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000014DAE92AE80>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x0000014DAE94DE80>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000014DAE978400>]], dtype=object)




    
![png](/posts_images/2018-01-30-data-quest-guided-project-visualizing-earnings-based-on-college-majors/output_20_1.png)
    



```python
scatter_matrix(recent_grads[['Men', 'ShareWomen', 'Median']], figsize=(10,10))
```




    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x0000014DAE9E4E48>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000014DAEA354E0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000014DAEA59550>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x0000014DAEA6C860>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000014DAEAA1550>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000014DAEABAE80>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x0000014DAEADFF60>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000014DAEB03F60>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000014DAEB25F60>]], dtype=object)




    
![png](/posts_images/2018-01-30-data-quest-guided-project-visualizing-earnings-based-on-college-majors/output_21_1.png)
    


We are not really seeing much correlations betwen these plots, There is a weak negative correlation between 'ShareWomen' and Median. Majors with less women tend to have higher earnings. It could be due to the fact that high paying majors like engineering tend to have less women. 

The first ten rows in the data are mostly engineering majors, and the last ten rows are non engineering majors. We can generate a bar chart and look at the 'ShareWomen' vs 'Majors' to see if our hypothesis is correct.


```python
recent_grads[:10].plot(kind='bar', x='Major', y='ShareWomen', colormap='winter')
recent_grads[163:].plot(kind='bar', x='Major', y='ShareWomen', colormap='winter')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x14daedf7fd0>




    
![png](/posts_images/2018-01-30-data-quest-guided-project-visualizing-earnings-based-on-college-majors/output_23_1.png)
    



    
![png](/posts_images/2018-01-30-data-quest-guided-project-visualizing-earnings-based-on-college-majors/output_23_2.png)
    


Let's plot the majors we selected above with 'Median' income to see if engineers earn more income.


```python
recent_grads[:10].plot(kind='bar', x='Major', y='Median', colormap='winter')
recent_grads[163:].plot(kind='bar', x='Major', y='Median', colormap='winter')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x14daee985c0>




    
![png](/posts_images/2018-01-30-data-quest-guided-project-visualizing-earnings-based-on-college-majors/output_25_1.png)
    



    
![png](/posts_images/2018-01-30-data-quest-guided-project-visualizing-earnings-based-on-college-majors/output_25_2.png)
    


Our hypothesis appears to be correct, at least for the majors we selected. Majors with less women such as engineering tend to earn higher salaries.

---

#### Learning Summary

Python concepts explored: pandas, matplotlib, histograms, bar charts, scatterplots, scatter matrices

Python functions and methods used: .plot(), scatter_matrix(), hist(), iloc[], .head(), .tail(), .describe()


The files used for this project can be found in my [GitHub repository](https://github.com/sengkchu/Dataquest-Guided-Projects-Solutions/tree/master/Guided%20Project_%20Visualizing%20Earnings%20Based%20On%20College%20Majors).

