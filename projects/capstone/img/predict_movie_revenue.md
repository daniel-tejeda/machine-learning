
# Machine Learning Engineer Nanodegree - 2019
## Capstone Project: Movie Box Office Revenue Prediction with Gradient Boosting Models
Author: Daniel Tejeda

In a world where movies made an estimated 41.7 billion in 2018 and expected to go over 50 billion by 2020, the film industry is more popular than ever. But which movies make the most money at the box office? How much does a director or the budget matter? In this capstone project, I will build a model to answer that question, working with metadata on over 7,000 past films from The Movie Database published as part of TMDB Box Office Prediction Kaggle competition. 


### Project Design

This project will be implemented in Python 3.7. Libraries involved will be numpy, pandas, matplotlib, seaborn, xgboost, lightgbm, catboost, scikit-learn.

The workflow for this project will be in the following order: 
0. Import all libraries
- Exploratory data analysis
- Data cleansing and Feature engineering
- Train the KNN benchmark model based on budget, popularity and runtime
- Stage-1: Boosting Models 
    * Code and train XGBoost model 
    * Code and train CATBoost model
    * Code and train LightGBM model
    * Hyperparameter tuning for the three models
    * Evaluate results against KNN and select new benchmark from the boosting models to be the new benchmark.
    
- Stage 2: Stacked final model
    * Select stacking approach and regression algorithm for the final model
    * Train regression algorithm with the outputs of the base boosting models combined with the original features, according to the stacking approach
    * Hyperparameter tuning for the final model
    * Evaluate and report final results against the benchmark 



## 0. Import all libraries


```python
%load_ext autoreload
%autoreload 2

import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

import visuals as vs
import data_prep as dp
import json
from itertools import cycle, islice
from datetime import datetime

from sklearn.metrics import mean_squared_log_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, SGDRegressor, LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold
from sklearn import cluster

import xgboost as xgb
import lightgbm as lgb
import catboost as cat
import copy


```

## 1. Exploratory data analysis


```python
dataset_names = ['train','test']

# load original datasets
original_datasets = { ds : pd.read_csv("data/{}.csv".format(ds)) for ds in dataset_names }

# complete missing budget/revenue values from B H's Kernel. (https://www.kaggle.com/zero92/tmdb-prediction)
dp.complete_missing_data(original_datasets)

original_datasets['train'].head(3)
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
      <th>belongs_to_collection</th>
      <th>budget</th>
      <th>genres</th>
      <th>homepage</th>
      <th>imdb_id</th>
      <th>original_language</th>
      <th>original_title</th>
      <th>overview</th>
      <th>popularity</th>
      <th>...</th>
      <th>release_date</th>
      <th>runtime</th>
      <th>spoken_languages</th>
      <th>status</th>
      <th>tagline</th>
      <th>title</th>
      <th>Keywords</th>
      <th>cast</th>
      <th>crew</th>
      <th>revenue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>[{'id': 313576, 'name': 'Hot Tub Time Machine ...</td>
      <td>14000000</td>
      <td>[{'id': 35, 'name': 'Comedy'}]</td>
      <td>NaN</td>
      <td>tt2637294</td>
      <td>en</td>
      <td>Hot Tub Time Machine 2</td>
      <td>When Lou, who has become the "father of the In...</td>
      <td>6.575393</td>
      <td>...</td>
      <td>2/20/15</td>
      <td>93.0</td>
      <td>[{'iso_639_1': 'en', 'name': 'English'}]</td>
      <td>Released</td>
      <td>The Laws of Space and Time are About to be Vio...</td>
      <td>Hot Tub Time Machine 2</td>
      <td>[{'id': 4379, 'name': 'time travel'}, {'id': 9...</td>
      <td>[{'cast_id': 4, 'character': 'Lou', 'credit_id...</td>
      <td>[{'credit_id': '59ac067c92514107af02c8c8', 'de...</td>
      <td>12314651</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>[{'id': 107674, 'name': 'The Princess Diaries ...</td>
      <td>40000000</td>
      <td>[{'id': 35, 'name': 'Comedy'}, {'id': 18, 'nam...</td>
      <td>NaN</td>
      <td>tt0368933</td>
      <td>en</td>
      <td>The Princess Diaries 2: Royal Engagement</td>
      <td>Mia Thermopolis is now a college graduate and ...</td>
      <td>8.248895</td>
      <td>...</td>
      <td>8/6/04</td>
      <td>113.0</td>
      <td>[{'iso_639_1': 'en', 'name': 'English'}]</td>
      <td>Released</td>
      <td>It can take a lifetime to find true love; she'...</td>
      <td>The Princess Diaries 2: Royal Engagement</td>
      <td>[{'id': 2505, 'name': 'coronation'}, {'id': 42...</td>
      <td>[{'cast_id': 1, 'character': 'Mia Thermopolis'...</td>
      <td>[{'credit_id': '52fe43fe9251416c7502563d', 'de...</td>
      <td>95149435</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>NaN</td>
      <td>3300000</td>
      <td>[{'id': 18, 'name': 'Drama'}]</td>
      <td>http://sonyclassics.com/whiplash/</td>
      <td>tt2582802</td>
      <td>en</td>
      <td>Whiplash</td>
      <td>Under the direction of a ruthless instructor, ...</td>
      <td>64.299990</td>
      <td>...</td>
      <td>10/10/14</td>
      <td>105.0</td>
      <td>[{'iso_639_1': 'en', 'name': 'English'}]</td>
      <td>Released</td>
      <td>The road to greatness can take you to the edge.</td>
      <td>Whiplash</td>
      <td>[{'id': 1416, 'name': 'jazz'}, {'id': 1523, 'n...</td>
      <td>[{'cast_id': 5, 'character': 'Andrew Neimann',...</td>
      <td>[{'credit_id': '54d5356ec3a3683ba0000039', 'de...</td>
      <td>13092000</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 23 columns</p>
</div>




```python
n_records = original_datasets['train'].shape[0]
n_columns = original_datasets['train'].shape[1]

print("Total number of records: {}".format(n_records))
print("Total number of features: {}".format(n_columns))
```

    Total number of records: 3000
    Total number of features: 23



```python
original_datasets['train'].describe(include='all')
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
      <th>belongs_to_collection</th>
      <th>budget</th>
      <th>genres</th>
      <th>homepage</th>
      <th>imdb_id</th>
      <th>original_language</th>
      <th>original_title</th>
      <th>overview</th>
      <th>popularity</th>
      <th>...</th>
      <th>release_date</th>
      <th>runtime</th>
      <th>spoken_languages</th>
      <th>status</th>
      <th>tagline</th>
      <th>title</th>
      <th>Keywords</th>
      <th>cast</th>
      <th>crew</th>
      <th>revenue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3000.000000</td>
      <td>604</td>
      <td>3.000000e+03</td>
      <td>2993</td>
      <td>946</td>
      <td>3000</td>
      <td>3000</td>
      <td>3000</td>
      <td>2992</td>
      <td>3000.000000</td>
      <td>...</td>
      <td>3000</td>
      <td>2998.000000</td>
      <td>2980</td>
      <td>3000</td>
      <td>2403</td>
      <td>3000</td>
      <td>2724</td>
      <td>2987</td>
      <td>2984</td>
      <td>3.000000e+03</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>NaN</td>
      <td>422</td>
      <td>NaN</td>
      <td>872</td>
      <td>941</td>
      <td>3000</td>
      <td>36</td>
      <td>2975</td>
      <td>2992</td>
      <td>NaN</td>
      <td>...</td>
      <td>2398</td>
      <td>NaN</td>
      <td>401</td>
      <td>2</td>
      <td>2400</td>
      <td>2969</td>
      <td>2648</td>
      <td>2975</td>
      <td>2984</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>top</th>
      <td>NaN</td>
      <td>[{'id': 645, 'name': 'James Bond Collection', ...</td>
      <td>NaN</td>
      <td>[{'id': 18, 'name': 'Drama'}]</td>
      <td>http://www.transformersmovie.com/</td>
      <td>tt0186045</td>
      <td>en</td>
      <td>The Three Musketeers</td>
      <td>Evicted from his squat and suddenly alone on t...</td>
      <td>NaN</td>
      <td>...</td>
      <td>9/10/15</td>
      <td>NaN</td>
      <td>[{'iso_639_1': 'en', 'name': 'English'}]</td>
      <td>Released</td>
      <td>Based on a true story.</td>
      <td>Ghost</td>
      <td>[{'id': 10183, 'name': 'independent film'}]</td>
      <td>[]</td>
      <td>[{'credit_id': '52fe4c639251416c75118f21', 'de...</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>NaN</td>
      <td>16</td>
      <td>NaN</td>
      <td>266</td>
      <td>4</td>
      <td>1</td>
      <td>2575</td>
      <td>2</td>
      <td>1</td>
      <td>NaN</td>
      <td>...</td>
      <td>5</td>
      <td>NaN</td>
      <td>1817</td>
      <td>2996</td>
      <td>3</td>
      <td>2</td>
      <td>27</td>
      <td>13</td>
      <td>1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1500.500000</td>
      <td>NaN</td>
      <td>2.266135e+07</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.463274</td>
      <td>...</td>
      <td>NaN</td>
      <td>107.856571</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6.668176e+07</td>
    </tr>
    <tr>
      <th>std</th>
      <td>866.169729</td>
      <td>NaN</td>
      <td>3.702662e+07</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>12.104000</td>
      <td>...</td>
      <td>NaN</td>
      <td>22.086434</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.375149e+08</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>NaN</td>
      <td>0.000000e+00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000001</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>750.750000</td>
      <td>NaN</td>
      <td>0.000000e+00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.018053</td>
      <td>...</td>
      <td>NaN</td>
      <td>94.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.401550e+06</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1500.500000</td>
      <td>NaN</td>
      <td>8.000000e+06</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.374861</td>
      <td>...</td>
      <td>NaN</td>
      <td>104.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.680707e+07</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2250.250000</td>
      <td>NaN</td>
      <td>3.000000e+07</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10.890983</td>
      <td>...</td>
      <td>NaN</td>
      <td>118.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6.877599e+07</td>
    </tr>
    <tr>
      <th>max</th>
      <td>3000.000000</td>
      <td>NaN</td>
      <td>3.800000e+08</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>294.337037</td>
      <td>...</td>
      <td>NaN</td>
      <td>338.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.519558e+09</td>
    </tr>
  </tbody>
</table>
<p>11 rows × 23 columns</p>
</div>



Some fields contain list with dictionaries. We will need to extract and transform these features. 

 

### Missing values


```python
original_datasets['train'].isna().sum()
```




    id                          0
    belongs_to_collection    2396
    budget                      0
    genres                      7
    homepage                 2054
    imdb_id                     0
    original_language           0
    original_title              0
    overview                    8
    popularity                  0
    poster_path                 1
    production_companies      156
    production_countries       55
    release_date                0
    runtime                     2
    spoken_languages           20
    status                      0
    tagline                   597
    title                       0
    Keywords                  276
    cast                       13
    crew                       16
    revenue                     0
    dtype: int64



### Distribution of target variable Revenue
Let's take a look at the target variable revenue and how its distributed. We are also plotting the Log1p, since we are going to use this in out model


```python
quant_data = original_datasets['train'].set_index('id')[['budget','popularity','runtime', 'revenue']]

sb.set_style("white")

fig, axes = plt.subplots(1, 2, figsize = (14,6))

sb.distplot(quant_data['revenue'], bins=20, kde=True, ax=axes[0])
sb.distplot(np.log1p(quant_data['revenue']), bins=40, kde=True, ax=axes[1])

for ax in axes: ax.grid()
    
```


![png](output_12_0.png)



```python
# scatter matrix for each pair of features in the data
pd.plotting.scatter_matrix(quant_data, alpha = 0.3, figsize = (16,10));
```


![png](output_13_0.png)


## 2. Data cleansing and feature engineering


```python
# copy the original dataset and replace NaN with empty string
all_data = original_datasets['train'].set_index('id')
```


```python
def data_prep_clean_na(data):
    # two movies have NaN runtime, we fill those with the mean
    data['runtime'].fillna(data['runtime'].mean(), inplace=True)

    # replace NaN in strings
    data.fillna('', inplace=True)

data_prep_clean_na(all_data)
```


```python
def data_prep_dates(data):
    #let's break down month/day/year from release_date
    data[['rel_month','rel_day','rel_year']] = data['release_date'].str.split('/', expand=True).astype(int)

    # fix 2-digit year for 1920-1999
    data['rel_year'] += 1900
    # 2000-2019
    data.loc[data['rel_year'] <= 1919, "rel_year"] += 100

    # extract day of week and quarter
    rel_date = pd.to_datetime(data['release_date']) 
    data['rel_dow'] = rel_date.dt.dayofweek
    data['rel_quarter'] = rel_date.dt.quarter

    
data_prep_dates(all_data)
all_data[['rel_month','rel_day','rel_year','rel_dow','rel_quarter']].head(3)
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
      <th>rel_month</th>
      <th>rel_day</th>
      <th>rel_year</th>
      <th>rel_dow</th>
      <th>rel_quarter</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>20</td>
      <td>2015</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>6</td>
      <td>2004</td>
      <td>4</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10</td>
      <td>10</td>
      <td>2014</td>
      <td>4</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
yr_counts = all_data['rel_year'].value_counts().sort_index()
yr_xticks = range(yr_counts.index.values[0],yr_counts.index.values[-1],5)

plt.figure(figsize = (14,6))
plt.plot(yr_counts.index.values, yr_counts.values)
plt.xticks(yr_xticks)
plt.title("Movies released by year",fontsize=20)
plt.grid()
```


![png](output_18_0.png)



```python
yr_avgs = all_data.groupby('rel_year').mean()
yr_totals = all_data.groupby('rel_year').sum()

fig, axes = plt.subplots(3, 1, figsize=(14,14))

axes[0].set_title("Mean revenue & budget by year")
axes[0].plot(yr_avgs.index.values, yr_avgs['revenue'], label='Mean revenue')
axes[0].plot(yr_avgs.index.values, yr_avgs['budget'], label='Mean budget')
axes[0].set_xticks(yr_xticks)

axes[1].set_title("Total revenue & budget by year")
axes[1].plot(yr_totals.index.values, yr_totals['revenue'], label='Total revenue')
axes[1].plot(yr_totals.index.values, yr_totals['budget'], label='Total budget')
axes[1].set_xticks(yr_xticks)

axes[2].set_title("Mean runtime by year")
axes[2].plot(yr_avgs.index.values, yr_avgs['runtime'], label='Mean runtime')
axes[2].set_xticks(yr_xticks)

for ax in axes: 
    ax.grid()
    ax.legend()
```


![png](output_19_0.png)



```python
def data_prep_collection(data, mincount=2):

    # belongs_to_collection
    data['from_collection'] = data['belongs_to_collection'].apply(lambda x: 1 if len(x) > 0 else 0)

    data['collection_name'] = data['belongs_to_collection'].map(lambda x: 'col_{}'.format(dp.get_dictionary(x)[0]['name']) if len(x) > 0 else '')

    collection_names = data['collection_name'].str.translate({ord(i): None for i in '[]<'}).str.get_dummies()
    
    cols = list(collection_names.sum()[collection_names.sum()>=mincount].index)
    
    data[cols] = collection_names[cols]
    
    return cols
    
encoded_cols = data_prep_collection(all_data)
print("Movies that belong to a collection: {} / {}".format(sum(all_data['from_collection']), n_records))
print('{} collection columns added'.format(len(encoded_cols)))
all_data[encoded_cols].rename(lambda x:x[4:], axis='columns').sum().sort_values(ascending=False)
```

    Movies that belong to a collection: 604 / 3000
    116 collection columns added





    James Bond Collection                        16
    Friday the 13th Collection                    7
    The Pink Panther (Original) Collection        6
    Pokémon Collection                            5
    Police Academy Collection                     5
    Paranormal Activity Collection                4
    The Fast and the Furious Collection           4
    Rocky Collection                              4
    Resident Evil Collection                      4
    Rambo Collection                              4
    Child's Play Collection                       4
    Ice Age Collection                            4
    Alien Collection                              4
    Transformers Collection                       4
    The Vengeance Collection                      3
    Indiana Jones Collection                      3
    Halloween Collection                          3
    Star Trek: The Original Series Collection     3
    Missing in Action Collection                  3
    The Dark Knight Collection                    3
    Scary Movie Collection                        3
    Mexico Trilogy                                3
    Rush Hour Collection                          3
    Alex Cross Collection                         3
    Planet of the Apes Original Collection        3
    Diary of a Wimpy Kid Collection               3
    Pirates of the Caribbean Collection           3
    The Jaws Collection                           3
    REC Collection                                3
    Three Heroes Collection                       3
                                                 ..
    V/H/S Collection                              2
    Wall Street Collection                        2
    What the Bleep! Collection                    2
    World of Watches Collection                   2
    Would I Lie to You? Collection                2
    The Godfather Collection                      2
    The Exorcist Collection                       2
    Percy Jackson Collection                      2
    Step Up Collection                            2
    Pet Sematary Collection                       2
    Peter Pan Collection                          2
    Predator Collection                           2
    X-Men Collection                              2
    Recep İvedik Serisi                           2
    Saw Collection                                2
    Smokey and the Bandit Collection              2
    Species Collection                            2
    Star Wars Collection                          2
    Superman Collection                           2
    The Conjuring Collection                      2
    Swan Princess Series                          2
    TRON Collection                               2
    Taxi Collection                               2
    Ted Collection                                2
    Texas Chainsaw Massacre Collection            2
    The Avengers Collection                       2
    The Blue Lagoon collection                    2
    The Bourne Collection                         2
    The Chronicles of Narnia Collection           2
    ... Has Fallen Collection                     2
    Length: 116, dtype: int64




```python
def data_prep_homepage(data):
    #homepage
    data['has_homepage'] = data['homepage'].apply(lambda x: 1 if len(x) > 0 else 0)
    
data_prep_homepage(all_data)  
print("Movies with homepage: {} / {}".format(sum(all_data['has_homepage']), n_records))
```

    Movies with homepage: 946 / 3000



```python
def data_prep_genres(data):

    #extract genre information from genre column
    data['genres_new'] = data['genres'].map(lambda x: sorted(['genre_{}'.format(d['name']) for d in dp.get_dictionary(x)])).map(lambda x: ','.join(map(str, x)))

    #one-hot-encoding 
    genres = data.genres_new.str.get_dummies(sep=',')

    #concatenate one-hot-encoding for genres to data
    data[genres.columns] = genres
    
    return genres.columns


encoded_cols= data_prep_genres(all_data)
print('{} genre columns added'.format(len(encoded_cols)))
all_data[encoded_cols].rename(lambda x:x[6:], axis='columns').sum()
```

    20 genre columns added





    Action              741
    Adventure           439
    Animation           141
    Comedy             1028
    Crime               469
    Documentary          87
    Drama              1531
    Family              260
    Fantasy             232
    Foreign              31
    History             132
    Horror              301
    Music               100
    Mystery             225
    Romance             571
    Science Fiction     290
    TV Movie              1
    Thriller            789
    War                 100
    Western              43
    dtype: int64




```python
def data_prep_cast_crew(data):
    
    for feat in ['cast','crew']:
        totcol = '{}_cnt'.format(feat)
        data[totcol] = 0
        for i in range(0,3):
            col = '{}_g{}'.format(feat,i)
            data[col] = data['cast'].apply(lambda x: sum([1 for d in dp.get_dictionary(x) if d['gender'] == i]))
            data[totcol] += data[col]
            
        
data_prep_cast_crew(all_data)
all_data[['cast_g0', 'cast_g1', 'cast_g2', 'cast_cnt']].head()
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
      <th>cast_g0</th>
      <th>cast_g1</th>
      <th>cast_g2</th>
      <th>cast_cnt</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>6</td>
      <td>8</td>
      <td>10</td>
      <td>24</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>10</td>
      <td>10</td>
      <td>20</td>
    </tr>
    <tr>
      <th>3</th>
      <td>31</td>
      <td>7</td>
      <td>13</td>
      <td>51</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>1</td>
      <td>2</td>
      <td>7</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
all_data[['crew_g0', 'crew_g1', 'crew_g2', 'crew_cnt']].head()
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
      <th>crew_g0</th>
      <th>crew_g1</th>
      <th>crew_g2</th>
      <th>crew_cnt</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>6</td>
      <td>8</td>
      <td>10</td>
      <td>24</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>10</td>
      <td>10</td>
      <td>20</td>
    </tr>
    <tr>
      <th>3</th>
      <td>31</td>
      <td>7</td>
      <td>13</td>
      <td>51</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>1</td>
      <td>2</td>
      <td>7</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
def data_prep_lang(data):
    
    original_langs = pd.get_dummies(data['original_language'], prefix='lang_')

    data[original_langs.columns] = original_langs
    
    return original_langs.columns

encoded_cols = data_prep_lang(all_data)
print('{} language columns added'.format(len(encoded_cols)))
all_data[encoded_cols].sum().sort_values(ascending=False).head(20)
```

    36 language columns added





    lang__en    2575
    lang__fr      78
    lang__ru      47
    lang__es      43
    lang__hi      42
    lang__ja      37
    lang__it      24
    lang__ko      20
    lang__cn      20
    lang__zh      19
    lang__de      18
    lang__ta      16
    lang__sv       8
    lang__pt       6
    lang__nl       6
    lang__fa       5
    lang__da       5
    lang__ro       4
    lang__hu       3
    lang__tr       3
    dtype: int64




```python
def data_prep_prod_companies(data, mincount=15):

    #extract production company information from column
    data['production_companies_new'] = data['production_companies'].map(lambda x: sorted(['pcomp_{}'.format(d['name'].strip()) for d in dp.get_dictionary(x)])).map(lambda x: '|'.join(map(str, x)))

    #one-hot-encoding 
    pcomp = data.production_companies_new.str.get_dummies(sep='|')

    #only columns with more than [mincount] entries
    cols = list(pcomp.sum()[pcomp.sum()>mincount].index)
        
    #concatenate one-hot-encoding for companies to data
    data[cols] = pcomp[cols]
    
    return cols

#prodcs.rename(lambda x:x[6:], axis='columns').sum()
encoded_cols = data_prep_prod_companies(all_data)
print('{} production companies columns added'.format(len(encoded_cols)))
all_data[encoded_cols].rename(lambda x:x[6:], axis='columns').sum().sort_values(ascending=False)
```

    45 production companies columns added





    Warner Bros.                              202
    Universal Pictures                        188
    Paramount Pictures                        161
    Twentieth Century Fox Film Corporation    138
    Columbia Pictures                          91
    Metro-Goldwyn-Mayer (MGM)                  84
    New Line Cinema                            75
    Touchstone Pictures                        63
    Walt Disney Pictures                       62
    Columbia Pictures Corporation              61
    TriStar Pictures                           53
    Relativity Media                           48
    Canal+                                     46
    United Artists                             44
    Miramax Films                              40
    Village Roadshow Pictures                  36
    Regency Enterprises                        31
    Dune Entertainment                         30
    Working Title Films                        30
    BBC Films                                  30
    Fox Searchlight Pictures                   29
    StudioCanal                                28
    Lionsgate                                  28
    DreamWorks SKG                             27
    Fox 2000 Pictures                          25
    Orion Pictures                             24
    Hollywood Pictures                         24
    Summit Entertainment                       24
    Dimension Films                            23
    Amblin Entertainment                       23
    Focus Features                             21
    Epsilon Motion Pictures                    21
    Original Film                              21
    Castle Rock Entertainment                  21
    Morgan Creek Productions                   21
    Legendary Pictures                         19
    Participant Media                          19
    Film4                                      18
    New Regency Pictures                       18
    Blumhouse Productions                      18
    Spyglass Entertainment                     17
    Imagine Entertainment                      16
    Millennium Films                           16
    Screen Gems                                16
    TSG Entertainment                          16
    dtype: int64




```python
def data_prep_prod_country(data, mincount=5):

    #extract production company information from genre column
    data['production_countries_new'] = data['production_countries'].map(lambda x: sorted(['pctry_{}'.format(d['name'].strip()) for d in dp.get_dictionary(x)])).map(lambda x: '|'.join(map(str, x)))

    #one-hot-encoding 
    pcntry = data.production_countries_new.str.get_dummies(sep='|')
    
    #only columns with more than [mincount] entries
    cols = list(pcntry.sum()[pcntry.sum()>mincount].index)
    
    all_data[cols] = pcntry[cols]
    
    return cols

encoded_cols = data_prep_prod_country(all_data)
print('{} countries columns added'.format(len(encoded_cols)))
all_data[encoded_cols].rename(lambda x:x[6:], axis='columns').sum().sort_values(ascending=False)
```

    35 countries columns added





    United States of America    2282
    United Kingdom               380
    France                       222
    Germany                      167
    Canada                       120
    India                         81
    Italy                         64
    Japan                         61
    Australia                     61
    Russia                        58
    Spain                         54
    China                         42
    Hong Kong                     42
    Belgium                       23
    Ireland                       23
    South Korea                   22
    Mexico                        19
    Sweden                        18
    New Zealand                   17
    Netherlands                   15
    Czech Republic                14
    Denmark                       13
    Brazil                        12
    Luxembourg                    10
    South Africa                  10
    Hungary                        9
    United Arab Emirates           9
    Romania                        8
    Switzerland                    8
    Austria                        8
    Greece                         7
    Norway                         7
    Finland                        6
    Chile                          6
    Argentina                      6
    dtype: int64




```python
def data_prep_final(data):

    #convert quantities to log 
    data[['log_budget','log_popularity','log_runtime']] = np.log1p(data[['budget','popularity','runtime']])

    # remove original columns
    drop_cols = ['release_date', 
                 'belongs_to_collection', 
                 'collection_name',
                 'homepage', 
                 'genres',
                 'genres_new',
                 'crew',
                 'cast',
                 'original_language',

                 'original_title',
                 'overview',
                 'production_companies',
                 'production_companies_new',
                 'production_countries',
                 'production_countries_new',
                 'spoken_languages',
                 'tagline',
                 'title',
                 'Keywords',
                 'status',

                 'imdb_id',
                 'poster_path',

                 'budget',
                 'popularity',
                 'runtime'
                ]

    data.drop(drop_cols, axis=1, inplace=True);
    
    
data_prep_final(all_data)
all_data.shape
```




    (3000, 271)




```python
def prepare_data(in_data):
    
    #copy
    data = in_data.set_index('id')
    
    data_prep_clean_na(data)
    data_prep_dates(data)
    data_prep_collection(data)
    data_prep_homepage(data)
    data_prep_genres(data)
    data_prep_cast_crew(data)
    data_prep_lang(data)
    data_prep_prod_companies(data)
    data_prep_prod_country(data)
    data_prep_final(data)
    
    return data
    
    """
    # two movies have NaN runtime, we fill those with the mean
    data['runtime'].fillna(data['runtime'].mean(), inplace=True)

    # replace NaN in strings
    data.fillna('', inplace=True)

    #let's break down month/day/year from release_date
    data[['rel_month','rel_day','rel_year']] = data['release_date'].str.split('/', expand=True).astype(int)

    # fix 2-digit year for 1920-1999
    data['rel_year'] += 1900
    # 2000-2019
    data.loc[data['rel_year'] <= 1919, "rel_year"] += 100

    # extract day of week and quarter
    rel_date = pd.to_datetime(data['release_date']) 
    data['rel_dow'] = rel_date.dt.dayofweek
    data['rel_quarter'] = rel_date.dt.quarter

    data[['rel_month','rel_day','rel_year','rel_dow','rel_quarter']].head(3)

    # belongs_to_collection
    data['from_collection'] = data['belongs_to_collection'].apply(lambda x: 1 if len(x) > 0 else 0)
    data['collection_name'] = data['belongs_to_collection'].map(lambda x: 'col_{}'.format(dp.get_dictionary(x)[0]['name']) if len(x) > 0 else '')
    collection_names = data['collection_name'].str.translate({ord(i): None for i in '[]<'}).str.get_dummies()

    #data = pd.concat([data, collection_names], axis=1, sort=False)
    data[collection_names.columns] = collection_names
    
    #homepage
    data['has_homepage'] = data['homepage'].apply(lambda x: 1 if len(x) > 0 else 0)

    #extract genre information from genre column
    data['genres_new'] = data['genres'].map(lambda x: sorted(['genre_{}'.format(d['name']) for d in dp.get_dictionary(x)])).map(lambda x: ','.join(map(str, x)))

    #one-hot-encoding 
    genres = data.genres_new.str.get_dummies(sep=',')

    #concatenate one-hot-encoding for genres to data
    #data = pd.concat([data, genres], axis=1, sort=False)
    data[genres.columns] = genres
    
    genres.rename(lambda x:x[6:], axis='columns').sum()

    data['crew_g0'] = data['crew'].apply(lambda x: sum([1 for i in dp.get_dictionary(x) if i['gender'] == 0]))
    data['crew_g1'] = data['crew'].apply(lambda x: sum([1 for i in dp.get_dictionary(x) if i['gender'] == 1]))
    data['crew_g2'] = data['crew'].apply(lambda x: sum([1 for i in dp.get_dictionary(x) if i['gender'] == 2]))
    data['crew_cnt'] = data['crew_g0'] + data['crew_g1'] + data['crew_g2']  
    #TODO: more with crew

    data[['crew_g0', 'crew_g1', 'crew_g2', 'crew_cnt']].head()

   
    data['cast_g0'] = data['cast'].apply(lambda x: sum([1 for i in dp.get_dictionary(x) if i['gender'] == 0]))
    data['cast_g1'] = data['cast'].apply(lambda x: sum([1 for i in dp.get_dictionary(x) if i['gender'] == 1]))
    data['cast_g2'] = data['cast'].apply(lambda x: sum([1 for i in dp.get_dictionary(x) if i['gender'] == 2]))

    data['cast_cnt'] = data['cast_g0'] + data['cast_g1']+data['cast_g2']

    #TODO: more with cast

    data[['cast_g0', 'cast_g1', 'cast_g2', 'cast_cnt']].head()

    original_langs = pd.get_dummies(data['original_language'], prefix='lang_')
    #data = pd.concat([data, original_langs], axis=1, sort=False)
    data[original_langs.columns] = original_langs
    
    
    #extract production company information from column
    data['production_companies_new'] = data['production_companies'].map(lambda x: sorted(['pcomp_{}'.format(d['name'].strip()) for d in dp.get_dictionary(x)])).map(lambda x: '|'.join(map(str, x)))

    #one-hot-encoding 
    pcomp = data.production_companies_new.str.get_dummies(sep='|')

    # one-hot-encoding for genres to data
    data[pcomp.columns] = pcomp

    
    data['production_countries_new'] = data['production_countries'].map(lambda x: sorted(['pctry_{}'.format(d['name'].strip()) for d in dp.get_dictionary(x)])).map(lambda x: '|'.join(map(str, x)))

    #one-hot-encoding 
    pcntry = data.production_countries_new.str.get_dummies(sep='|')

    #concatenate one-hot-encoding for genres to all_data
    #all_data = pd.concat([all_data, pcntry], axis=1, sort=False)

    data[pcntry.columns] = pcntry

    #convert quantities to log 
    data[['log_budget','log_popularity','log_runtime']] = np.log1p(data[['budget','popularity','runtime']])

    # remove original columns
    drop_cols = ['release_date', 
                 'belongs_to_collection', 
                 'collection_name',
                 'homepage', 
                 'genres',
                 'genres_new',
                 'crew',
                 'cast',
                 'original_language',

                 'original_title',
                 'overview',
                 'production_companies',
                 'production_companies_new',
                 'production_countries',
                 'production_countries_new',
                 'spoken_languages',
                 'tagline',
                 'title',
                 'Keywords',
                 'status',

                 'imdb_id',
                 'poster_path',

                 'budget',
                 'popularity',
                 'runtime'
                ]

    data.drop(drop_cols, axis=1, inplace=True);
    
    return data
    
    """


```

## 3. Train the KNN benchmark model based on budget, popularity and runtime


```python
# remove table meta data, column names etc. 
X = all_data[['log_budget','log_popularity','log_runtime']].values
y = all_data[['revenue']].values

# create Validation Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.01, random_state=16)

# scale
X_scaler = StandardScaler()
X_train_scaled  = X_scaler.fit_transform(X_train)
X_val_scaled    = X_scaler.transform(X_val)

y_train_scaled = np.log1p(y_train)

#create regressor, fit the data
reg = KNeighborsRegressor().fit(X_train_scaled, y_train_scaled)

#define score function
def score_function(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

# apply the regression model on the prepared train, validation and test set and invert the logarithmic scaling
y_train_pred  = np.expm1(reg.predict(X_train_scaled))
y_val_pred    = np.expm1(reg.predict(X_val_scaled))
#y_test_pred   = inverseY(reg.predict(X_test_scaled))

# print the RMLS error on training & validation 
print("RMLS Error on Training Dataset:\t", score_function(y_train , y_train_pred), score_function(y_train, y_train_pred))
print("RMLS Error on Val Dataset:\t", score_function(y_val , y_val_pred), score_function(y_val , y_val_pred))
```


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    <ipython-input-23-a37910dabd46> in <module>
          1 # remove table meta data, column names etc.
    ----> 2 X = all_data[['budget','popularity','runtime']].values
          3 y = all_data[['revenue']].values
          4 
          5 # create Validation Split


    /opt/anaconda3/envs/ml-1/lib/python3.6/site-packages/pandas/core/frame.py in __getitem__(self, key)
       2932                 key = list(key)
       2933             indexer = self.loc._convert_to_indexer(key, axis=1,
    -> 2934                                                    raise_missing=True)
       2935 
       2936         # take() does not accept boolean indexers


    /opt/anaconda3/envs/ml-1/lib/python3.6/site-packages/pandas/core/indexing.py in _convert_to_indexer(self, obj, axis, is_setter, raise_missing)
       1352                 kwargs = {'raise_missing': True if is_setter else
       1353                           raise_missing}
    -> 1354                 return self._get_listlike_indexer(obj, axis, **kwargs)[1]
       1355         else:
       1356             try:


    /opt/anaconda3/envs/ml-1/lib/python3.6/site-packages/pandas/core/indexing.py in _get_listlike_indexer(self, key, axis, raise_missing)
       1159         self._validate_read_indexer(keyarr, indexer,
       1160                                     o._get_axis_number(axis),
    -> 1161                                     raise_missing=raise_missing)
       1162         return keyarr, indexer
       1163 


    /opt/anaconda3/envs/ml-1/lib/python3.6/site-packages/pandas/core/indexing.py in _validate_read_indexer(self, key, indexer, axis, raise_missing)
       1244                 raise KeyError(
       1245                     u"None of [{key}] are in the [{axis}]".format(
    -> 1246                         key=key, axis=self.obj._get_axis_name(axis)))
       1247 
       1248             # We (temporarily) allow for some missing keys with .loc, except in


    KeyError: "None of [Index(['budget', 'popularity', 'runtime'], dtype='object')] are in the [columns]"


### Conclusion

## 4. Boosting models


```python
ds_train = prepare_data(original_datasets['train'])
ds_test_kaggle = prepare_data(original_datasets['test'])

ds_train.shape, ds_train.columns

```




    ((3000, 236),
     Index(['revenue', 'rel_month', 'rel_day', 'rel_year', 'rel_dow', 'rel_quarter',
            'from_collection', 'col_... Has Fallen Collection',
            'col_48 Hrs. Collection', 'col_Ace Ventura Collection',
            ...
            'pcomp_Twentieth Century Fox Film Corporation', 'pcomp_United Artists',
            'pcomp_Universal Pictures', 'pcomp_Village Roadshow Pictures',
            'pcomp_Walt Disney Pictures', 'pcomp_Warner Bros.',
            'pcomp_Working Title Films', 'log_budget', 'log_popularity',
            'log_runtime'],
           dtype='object', length=236))




```python
class MovieRevenuePredictor():
    
    BASE_MODELS = ['knn','xgb','lgb','cat']
    GB_MODELS = ['xgb','lgb','cat']
    
    def __init__(self, data, random_seed=1, splits=5, test_size=0.1): 
        
        self.random_seed = random_seed
    
        np.random.seed(random_seed)
        
        self.prepare_data(data, splits, test_size)
        self.init_models()
        
        
    
    def prepare_data(self, dataset, splits, test_size):
        
        train, test,  = train_test_split(dataset, test_size=0.1, random_state=self.random_seed)
        
        self.data = {
            'raw': dataset, 
            'train': train,
            'test' : test
        }
        
        kfold = KFold(splits, shuffle = True, random_state = self.random_seed)
        
        self.fold = list(kfold.split(self.data['train'].values))
        
        
    def init_models(self):
        
        self.models = {
            
            'knn': KNeighborsRegressor(),
            
            'xgb': xgb.XGBRegressor(max_depth=6, 
                                    learning_rate=0.01, 
                                    n_estimators=10000, 
                                    eta=0.01,
                                    objective='reg:linear', 
                                    gamma=1.45, 
                                    seed=self.random_seed, 
                                    silent=True,
                                    subsample=0.6, 
                                    colsample_bytree=0.7, 
                                    colsample_bylevel=0.50,
                                    eval_metric='rmse'),
            
            'lgb': lgb.LGBMRegressor(n_estimators=10000, 
                                     objective="regression", 
                                     metric="rmse", 
                                     num_leaves=20, 
                                     min_child_samples=100,
                                     learning_rate=0.01, 
                                     bagging_fraction=0.8, 
                                     feature_fraction=0.8, 
                                     bagging_frequency=1, 
                                     bagging_seed=self.random_seed, 
                                     subsample=.9, 
                                     colsample_bytree=.9,
                                     use_best_model=True),
            
            'cat': cat.CatBoostRegressor(iterations=10000, 
                                         learning_rate=0.01, 
                                         depth=5, 
                                         eval_metric='RMSE',
                                         colsample_bylevel=0.7,
                                         bagging_temperature = 0.2,
                                         metric_period = None,
                                         early_stopping_rounds=200,
                                         random_seed=self.random_seed),
            
           
        }
                    
        
    def train(self, models=BASE_MODELS, stacking=False, test=True, **kwargs):
        
        fit_params = kwargs.get('fit_params', {"early_stopping_rounds": 500, "verbose": 100})
        
        result_dict = { m: {} for m in models }

        X = self.data['train'].drop(['revenue'], axis=1).values
        y = self.data['train']['revenue'].values
        
        if stacking: 
            # prepare new features for stacked model training
            self.data['train_meta'] = copy.deepcopy(self.data['train']).reset_index()

        final_err = 0
        verbose = False
    
        for m in models:
            
            print("\n[{} model] start".format(m))

            start = datetime.now()
            
            result_dict[m]['valid'] = []
            
            for i, (trn, val) in enumerate(self.fold):
                                
                start_f = datetime.now()
                
                trn_x = X[trn]
                trn_y = np.log1p(y[trn])

                val_x = X[val]
                val_y = np.log1p(y[val])

                fold_val_pred = []
                fold_err = []
                
                fit_args= {} if m=='knn' or m=='knn1' else {**fit_params, 'eval_set':[(val_x, val_y)] }
                
                self.models[m].fit(trn_x, trn_y, **fit_args)
                
                val_pred = self.models[m].predict(val_x)
                val_score = np.sqrt(mean_squared_error(val_y, val_pred))
                
                trn_time_f = (datetime.now()-start_f).seconds/60
                
                result_dict[m]['valid'].append({ 'score': val_score, 'time': trn_time_f })
                
                print("\n[{} model][Fold {:>2}/{}] val score: {:.5f} ({:.2f} mins)".format(m, i+1, len(self.fold), val_score, trn_time_f))
                
                if stacking: 
                    self.data['train_meta'].loc[val,m] = np.expm1(val_pred)
                    
                
            trn_time = (datetime.now()-start).seconds/60
            trn_pred = self.models[m].predict(X)
            trn_score = np.sqrt(mean_squared_error(np.log1p(y), trn_pred))
            val_score = np.mean(pd.DataFrame(result_dict[m]['valid'])['score'])
            
            print("\n[{} model] end: val avg score: {:.5f} ({:.2f} mins)".format(m, val_score ,trn_time))
            
            result_dict[m]['train'] = { 'score': trn_score, 'time': trn_time }
            
            
        if test: 
            test_result = self.test(models=models) 
            
            for m in test_result:
                result_dict[m]['test'] = test_result[m] 
            
        if stacking: 
            self.data['train_meta'].set_index('id', inplace=True)
            self.train_meta(stack=models)
            
        return result_dict
            
        
    def test(self, models=BASE_MODELS, **kwargs):
     
        test_src = kwargs.get('test_data', self.data['test'])
        test_set = pd.DataFrame(data=test_src, columns=self.data['train'].columns).fillna(0) 
        
        result_dict = {}
        
        X = test_set.drop(['revenue'], axis=1).values
        y = test_set['revenue'].values
        
        for m in models:

            test_pred = np.expm1(self.models[m].predict(X))
            test_score = np.sqrt(mean_squared_log_error(y, test_pred))
            
            result_dict[m] = test_score
        
        return result_dict
    
    
    def predict(self, X, models=BASE_MODELS, prep_data=False):
        
        X_in = prepare_data(X) if prep_data else X        
        X_in = pd.DataFrame(data=X_in, columns=self.data['train'].columns).fillna(0) 
        X_in = X_in.drop(['revenue'], axis=1).values
        
        preds = { m: np.expm1(self.models[m].predict(X_in)) for m in models}
        
        return pd.DataFrame(data=preds, index=X.index)
    
    
    def train_meta(self, stack=BASE_MODELS, **kwargs):
        
        add_feats = kwargs.get('add_feats', [])
        
        train_set = copy.deepcopy(self.data['train_meta'])
        train_set[['revenue', *stack]] = np.log1p(train_set[['revenue', *stack]])
        
        x_train_meta = train_set[[*stack, *add_feats]]
        y_train_meta = train_set['revenue']

        self.meta_model = LinearRegression()
        self.meta_model.fit(x_train_meta, y_train_meta)
    
    def predict_meta(self, X, stack=BASE_MODELS, **kwargs):
        
        add_feats = kwargs.get('add_feats', [])
        
        # making sure we have the right features
        self.train_meta(stack=stack, **kwargs)
        
        data_set = pd.DataFrame(data=X, columns=self.data['train_meta'].columns).fillna(0)

        data_set[stack] = np.log1p(self.predict(models=stack, X=X))

        x_meta = data_set[[*stack, *add_feats]]

        pred = self.meta_model.predict(x_meta)

        return pd.DataFrame(np.expm1(pred), index=X.index, columns=['revenue'])
                 
        
    def test_meta(self, stack=BASE_MODELS, **kwargs):
        
        test_data = kwargs.get('test_data', self.data['test'])
                
        test_pred = self.predict_meta(test_data, stack=stack, **kwargs)
        
        return { 'preds': test_pred, 
                 'score': np.sqrt(mean_squared_log_error(test_pred, test_data['revenue'])) }
    
    
    
    
```


```python
movie_reg = MovieRevenuePredictor(ds_train, random_seed=2019, splits=10, test_size=0)

train_result = movie_reg.train(stacking=True, fit_params={"early_stopping_rounds": 500, "verbose": False })

```

    
    [knn model] start
    
    [knn model][Fold  1/10] val score: 2.56984 (0.00 mins)
    
    [knn model][Fold  2/10] val score: 2.40967 (0.00 mins)
    
    [knn model][Fold  3/10] val score: 2.42665 (0.00 mins)
    
    [knn model][Fold  4/10] val score: 2.55669 (0.00 mins)
    
    [knn model][Fold  5/10] val score: 2.57421 (0.00 mins)
    
    [knn model][Fold  6/10] val score: 2.52762 (0.00 mins)
    
    [knn model][Fold  7/10] val score: 2.50550 (0.00 mins)
    
    [knn model][Fold  8/10] val score: 2.24833 (0.00 mins)
    
    [knn model][Fold  9/10] val score: 2.54270 (0.00 mins)
    
    [knn model][Fold 10/10] val score: 2.69836 (0.00 mins)
    
    [knn model] end: val avg score: 2.50596 (0.00 mins)
    
    [xgb model] start
    
    [xgb model][Fold  1/10] val score: 1.78770 (1.07 mins)
    
    [xgb model][Fold  2/10] val score: 1.88346 (0.83 mins)
    
    [xgb model][Fold  3/10] val score: 1.92612 (0.32 mins)
    
    [xgb model][Fold  4/10] val score: 1.83281 (0.67 mins)
    
    [xgb model][Fold  5/10] val score: 1.88235 (0.73 mins)
    
    [xgb model][Fold  6/10] val score: 1.86312 (0.35 mins)
    
    [xgb model][Fold  7/10] val score: 1.81597 (0.50 mins)
    
    [xgb model][Fold  8/10] val score: 1.82785 (0.40 mins)
    
    [xgb model][Fold  9/10] val score: 1.96265 (0.25 mins)
    
    [xgb model][Fold 10/10] val score: 2.18098 (1.15 mins)
    
    [xgb model] end: val avg score: 1.89630 (6.35 mins)
    
    [lgb model] start
    
    [lgb model][Fold  1/10] val score: 1.81438 (0.10 mins)
    
    [lgb model][Fold  2/10] val score: 1.98397 (0.02 mins)
    
    [lgb model][Fold  3/10] val score: 1.89373 (0.05 mins)
    
    [lgb model][Fold  4/10] val score: 1.92439 (0.08 mins)
    
    [lgb model][Fold  5/10] val score: 1.83995 (0.08 mins)
    
    [lgb model][Fold  6/10] val score: 1.95367 (0.07 mins)
    
    [lgb model][Fold  7/10] val score: 1.84046 (0.05 mins)
    
    [lgb model][Fold  8/10] val score: 1.89156 (0.02 mins)
    
    [lgb model][Fold  9/10] val score: 2.03719 (0.03 mins)
    
    [lgb model][Fold 10/10] val score: 2.20421 (0.08 mins)
    
    [lgb model] end: val avg score: 1.93835 (0.67 mins)
    
    [cat model] start
    
    [cat model][Fold  1/10] val score: 1.82929 (2.23 mins)
    
    [cat model][Fold  2/10] val score: 1.96558 (0.65 mins)
    
    [cat model][Fold  3/10] val score: 1.96783 (0.80 mins)
    
    [cat model][Fold  4/10] val score: 1.91728 (2.28 mins)
    
    [cat model][Fold  5/10] val score: 1.91161 (2.17 mins)
    
    [cat model][Fold  6/10] val score: 1.87064 (1.80 mins)
    
    [cat model][Fold  7/10] val score: 1.90150 (1.57 mins)
    
    [cat model][Fold  8/10] val score: 1.93691 (0.27 mins)
    
    [cat model][Fold  9/10] val score: 2.01091 (0.77 mins)
    
    [cat model][Fold 10/10] val score: 2.25596 (2.15 mins)
    
    [cat model] end: val avg score: 1.95675 (14.75 mins)



```python
movie_reg2 = MovieRevenuePredictor(ds_train, random_seed=681)

train_result2 = movie_reg2.train(stacking=True)

```


```python
movie_reg = MovieRevenuePredictor(ds_train, random_seed=2019, splits=10)

movie_reg.data['test'] = movie_reg.data['train'].iloc[1]

train_result = movie_reg.train(stacking=True, test=False, fit_params={"early_stopping_rounds": 500, "verbose": False })

```


```python
trn_mods=list(train_result.keys())

var=movie_reg.data['train_meta'][['revenue',*trn_mods]]

var.head()
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
      <th>revenue</th>
      <th>knn</th>
      <th>xgb</th>
      <th>lgb</th>
      <th>cat</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>941</th>
      <td>18948425</td>
      <td>4.379821e+07</td>
      <td>3544104.0</td>
      <td>3.736819e+06</td>
      <td>7.019522e+06</td>
    </tr>
    <tr>
      <th>1569</th>
      <td>11263966</td>
      <td>3.870598e+07</td>
      <td>14150026.0</td>
      <td>1.176697e+07</td>
      <td>9.735208e+06</td>
    </tr>
    <tr>
      <th>119</th>
      <td>1796389</td>
      <td>1.161414e+06</td>
      <td>3182211.5</td>
      <td>8.784035e+05</td>
      <td>1.993180e+06</td>
    </tr>
    <tr>
      <th>894</th>
      <td>13000000</td>
      <td>8.653245e+06</td>
      <td>39659232.0</td>
      <td>2.276641e+07</td>
      <td>4.700110e+07</td>
    </tr>
    <tr>
      <th>1850</th>
      <td>18000000</td>
      <td>2.537058e+06</td>
      <td>5295027.0</td>
      <td>2.238780e+07</td>
      <td>1.537014e+06</td>
    </tr>
  </tbody>
</table>
</div>




```python
for m in train_result:
    print("[{}] train score: {:.5f} ({:.2f} mins)".format(m, train_result[m]['train']['score'], train_result[m]['train']['time']))
    #for i, fold in enumerate(train_result[m]['valid']):
    #    print("[{}][{}] valid score: {:.5f} ({:.2f} mins)".format(m, i+1, fold['score'], fold['time']))
    print("[{}] avg valid score: {:.5f}".format(m, pd.DataFrame(train_result[m]['valid'])['score'].mean()))
    print("[{}] test score: {:.5f}\n".format(m, train_result[m]['test']))
    
```

    [knn] train score: 2.08433 (0.00 mins)
    [knn] avg valid score: 2.50596
    [knn] test score: 2.64758
    
    [xgb] train score: 0.82146 (6.35 mins)
    [xgb] avg valid score: 1.89630
    [xgb] test score: 2.08754
    
    [lgb] train score: 1.26540 (0.67 mins)
    [lgb] avg valid score: 1.93835
    [lgb] test score: 2.09008
    
    [cat] train score: 1.46971 (14.75 mins)
    [cat] avg valid score: 1.95675
    [cat] test score: 2.06959
    



```python
add_feats=['log_budget', 'log_popularity']

movie_reg.test_meta()['score'], movie_reg.test_meta(add_feats=add_feats)['score']
```




    (2.0486911953388858, 2.0438489715948593)




```python
stack=['xgb']
add_feats=['log_budget', 'log_popularity']

movie_reg.test_meta(stack=['xgb'])['score'], movie_reg.test_meta(stack=['xgb'], add_feats=add_feats)['score']
```




    (2.088351607428507, 2.0863558069807238)




```python
stack=['xgb','lgb']
add_feats=['log_budget', 'log_popularity']

movie_reg.test_meta(stack=stack)['score'], movie_reg.test_meta(stack=stack, add_feats=add_feats)['score']
```




    (2.0603749072852593, 2.0584113102786303)




```python
stack=['xgb', 'cat']
add_feats=['log_budget', 'log_popularity']

movie_reg.test_meta(stack=stack)['score'], movie_reg.test_meta(stack=stack, add_feats=add_feats)['score']
```




    (2.0626728597167925, 2.0600813131090585)




```python
stack=['xgb','lgb', 'cat']
add_feats=['log_budget', 'log_popularity']

movie_reg.test_meta(stack=stack)['score'], movie_reg.test_meta(stack=stack, add_feats=add_feats)['score']
```




    (2.048765932378094, 2.04412595089169)




```python
movie_reg.test()
```




    {'knn': 2.6475774808608494,
     'xgb': 2.0875418378962687,
     'lgb': 2.0900807692562084,
     'cat': 2.0695926334817414}




```python
meta_preds = movie_reg.predict_meta(ds_test_kaggle)
meta_preds.head()
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
      <th>revenue</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3001</th>
      <td>5.999989e+07</td>
    </tr>
    <tr>
      <th>3002</th>
      <td>1.256172e+06</td>
    </tr>
    <tr>
      <th>3003</th>
      <td>3.293025e+06</td>
    </tr>
    <tr>
      <th>3004</th>
      <td>1.915197e+06</td>
    </tr>
    <tr>
      <th>3005</th>
      <td>1.309963e+06</td>
    </tr>
  </tbody>
</table>
</div>




```python
meta_preds.to_csv("submission_stacked.csv")
pd.read_csv("submission_stacked.csv".format(m)).head(5)
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
      <th>revenue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3001</td>
      <td>5.999989e+07</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3002</td>
      <td>1.256172e+06</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3003</td>
      <td>3.293025e+06</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3004</td>
      <td>1.915197e+06</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3005</td>
      <td>1.309963e+06</td>
    </tr>
  </tbody>
</table>
</div>




```python

preds = movie_reg.predict(ds_test_kaggle)

for m in preds: 
    p = preds[[m]].rename(columns={m:'revenue'})
    p.to_csv("submission_{}.csv".format(m))
    print("submission_{}.csv\n{}\n".format(m, p.head()))
    

meta_p = movie_reg.predict_meta(ds_test_kaggle, add_feats=add_feats)

meta_p.to_csv("submission_stacked.csv")

pd.read_csv("submission_stacked.csv").head()

```

    submission_knn.csv
               revenue
    id                
    3001  5.517532e+06
    3002  1.346190e+07
    3003  6.125836e+05
    3004  1.145506e+07
    3005  2.252262e+05
    
    submission_xgb.csv
               revenue
    id                
    3001  9.684370e+07
    3002  1.224628e+06
    3003  3.768918e+06
    3004  1.037579e+06
    3005  2.232792e+06
    
    submission_lgb.csv
               revenue
    id                
    3001  1.594823e+07
    3002  7.385629e+05
    3003  1.801145e+06
    3004  8.645131e+06
    3005  7.051149e+05
    
    submission_cat.csv
               revenue
    id                
    3001  8.618598e+07
    3002  5.683990e+06
    3003  7.541188e+06
    3004  1.336553e+06
    3005  6.802918e+05
    





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
      <th>revenue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3001</td>
      <td>6.707117e+07</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3002</td>
      <td>1.269512e+06</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3003</td>
      <td>3.323810e+06</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3004</td>
      <td>1.795603e+06</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3005</td>
      <td>1.256354e+06</td>
    </tr>
  </tbody>
</table>
</div>




```python

#!kaggle competitions submit -c tmdb-box-office-prediction -f submission_knn.csv -m "knn_f3"
#!kaggle competitions submit -c tmdb-box-office-prediction -f submission_xgb.csv -m "xgb_f3"
#!kaggle competitions submit -c tmdb-box-office-prediction -f submission_lgb.csv -m "lgb_f3"
#!kaggle competitions submit -c tmdb-box-office-prediction -f submission_cat.csv -m "cat_f3"

#!kaggle competitions submit -c tmdb-box-office-prediction -f submission_stacked.csv -m "stacked_f3"

```

    100%|████████████████████████████████████████| 100k/100k [00:05<00:00, 17.6kB/s]
    Successfully submitted to TMDB Box Office Prediction


```python

def test_meta_model(data, stack=movie_reg.BASE_MODELS, **kwargs):

    train_set = copy.deepcopy(movie_reg.data['train_meta'])
    train_set[['revenue', *stack]] = np.log1p(train_set[['revenue', *stack]])
    
    x_train_meta = train_set[stack]
    y_train_meta = train_set['revenue']

    model = LinearRegression()
    model.fit(x_train_meta, y_train_meta)
    
    test_set = pd.DataFrame(data=data, columns=movie_reg.data['train_meta'].columns).fillna(0)
    #addfeats = movie_reg.predict(models=stack, X=data)           

    test_set[stack] = movie_reg.predict(models=stack, X=data) 
    test_set[['revenue', *stack]] = np.log1p(test_set[['revenue', *stack]])

    x_test_meta = test_set[stack]
    y_test_meta = test_set['revenue']

    test_pred = model.predict(x_test_meta)

    return { 'preds': pd.DataFrame(np.expm1(test_pred), index=data.index, columns=['revenue']), 'score': np.sqrt(mean_squared_error(test_pred, y_test_meta)) }
    
    
test_meta_model(movie_reg.data['test'])['score']

#test_set[addfeats.columns].assign(revenue=test_set['revenue'].values), movie_reg.data['train_meta'][[*addfeats.columns, 'revenue']]
```


```python
from sklearn.linear_model import SGDRegressor

#model = SGDRegressor(loss='squared_loss', learning_rate='adaptive', random_state=2019)

#TODO: log1p to X too
X = movie_reg.data['train_meta'][movie_reg.GB_MODELS]
y = np.log1p(movie_reg.data['train_meta']['revenue'])

#model.fit(X, y)

from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.metrics import mean_squared_error

reg = LinearRegression(random_state=movie_reg.random_seed, max_iter=1000, tol=None)

parameters = { 'penalty': ['none', 'l2', 'l1', 'elasticnet'],  'max_iter':[1000], 'tol':[None]}

scorer = make_scorer(mean_squared_error)

cv_sets = ShuffleSplit(n_splits = 10, test_size = 0.20, random_state = movie_reg.random_seed)

grid_obj = GridSearchCV(reg, parameters, scoring=scorer, cv=cv_sets)

grid_fit = grid_obj.fit(X, y)

best_reg = grid_fit.best_estimator_
best_reg

```
