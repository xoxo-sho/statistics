#!/usr/bin/env python
# coding: utf-8

# In[1]:


from itertools import combinations
import random
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from factor_analyzer import FactorAnalyzer


# ## import data

# In[2]:


df = pd.read_csv('sample_data.csv')
display(df.head())
display(df.describe())


# 最初の25列が質問文への回答（6段階評定）、残り3列は回答者の属性

# ## Preprocessing

# In[3]:


df = df.iloc[:,0:25] # excluding sex,academic background and age
df.info()


# In[4]:


df.dropna(inplace=True) # deleting missing value
df = df.astype('int') 
df.info()


# ## Factor Analysis

# In[5]:


n_factors = 5

fa_promax = FactorAnalyzer(rotation='promax', n_factors=n_factors)
fa_promax.fit(df)


# In[6]:


pd.DataFrame(fa_promax.loadings_, index=df.columns, columns=['factor_{}'.format(i) for i in range(n_factors)])


# In[7]:


result_df = pd.DataFrame(fa_promax.loadings_, index=df.columns, columns=['factor_{}'.format(i) for i in range(n_factors)])


# In[8]:


# adding new column from index as 'Question_Type'
result_df['Question_Type'] = [x[0] for x in result_df.index]


elements = result_df.drop('Question_Type', axis=1).columns
combinations_list = list(combinations(elements, 2))

for combo in combinations_list: 
    sns.scatterplot(x=combo[0], y=combo[1], hue='Question_Type', data=result_df)
    
    legend = plt.legend(loc='upper right', ncol=2)
    legend.set_title("Question_Type") # Title
    for label in legend.get_texts():
        label.set_fontsize(10)  # text size

    plt.show()


# In[9]:


# rundom data
random_list = [[random.randint(1, 6) for _ in range(25)] for i in range(5)]
df_sample = pd.DataFrame(random_list, columns=df.columns)

factor_scores = fa_promax.transform(df_sample)  # compute factor scores
pd.DataFrame(factor_scores, columns=['factor_{}'.format(i) for i in range(n_factors)], index=df_sample.index)

