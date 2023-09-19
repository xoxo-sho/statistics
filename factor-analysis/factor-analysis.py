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

# 最初の25列が質問文への回答（6段階評定）、残り3列は回答者の属性

# ### 質問内容
# A1
# Am indifferent to the feelings of others
# 
# A2
# Inquire about others' well-being.
# 
# A3
# Know how to comfort others.
# 
# A4
# Love children.
# 
# A5
# Make people feel at ease.
# 
# C1
# Am exacting in my work
# 
# C2
# Continue until everything is perfect
# 
# C3
# Do things according to a plan
# 
# C4
# Do things in a half-way manner
# 
# C5
# Waste my time.
# 
# E1
# Don't talk a lot
# 
# E2
# Find it difficult to approach others
# 
# E3
# Know how to captivate people.
# 
# E4
# Make friends easily.
# 
# E5
# Take charge.
# 
# N1
# Get angry easily
# 
# N2
# Get irritated easily
# 
# N3
# Have frequent mood swings.
# 
# N4
# Often feel blue.
# 
# N5
# Panic easily.
# 
# O1
# Am full of ideas
# 
# O2
# Avoid difficult reading materia
# 
# O3
# Carry the conversation to a higher level
# 
# O4
# Spend time reflecting on things.
# 
# O5
# Will not probe deeply into a subject.

# ## Preprocessing

# In[3]:


df = df.iloc[:,0:25] # excluding sex,academic background and age


# In[4]:


df.dropna(inplace=True) # deleting missing value
df = df.astype('int') 


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


# ## Discussion
# 
# ### factor_0
# - factor_0は質問Nに大きな因子負荷量が見られる。
# - 精神的にネガティブの状態になりやすいか質問内容で問われていて、factor_0はNの全ての質問に対してプラスの負荷。
# 
# \>>> **Neuroticism（神経症傾向・情緒不安定性）を表す**
# 
# ### factor_1
# - factor1は質問Eに大きな因子負荷量が見られる。
# - コミュニケーションに関する質問内容で、コミュニケーションにおいてネガティブな要素でfactor_１がマイナスの負荷、ポジディブな要素でfactor_１がプラスの負荷。
# 
# \>>> **Extraversion（外向性）**
# 
# ### factor_2
# - factor_2は質問Cに大きな因子負荷量が見られる。
# - 物事や時間に対する姿勢に関する質問内容で、ネガティブな要素でfactor_2がマイナスの負荷、ポジディブな要素でfactor_2がプラスの負荷。
# 
# \>>> **Conscientiousness（勤勉性・誠実性）**
# 
# ### factor_3
# - factor_3は質問Aに大きな因子負荷量が見られる。
# - 他人への思いやりや関心に関する質問内容で、ネガティブな要素でfactor_3がマイナスの負荷、ポジディブな要素でfactor_3がプラスの負荷。
# 
# \>>> **Agreeableness（協調性・調和性）**
# 
# ### factor_4
# - factor_4は質問Oに大きな因子負荷量が見られる。
# - 好奇心やチャレンジングな姿勢に関する質問内容で、ネガティブな要素でfactor_4がマイナスの負荷、ポジディブな要素でfactor_4がプラスの負荷。
# 
# \>>> **Openness（開放性・経験への開放性）**

# ## Experiment
# 
# ランダムな回答のデータを生成して、どのような人格の特徴があるか考察する。
# <br>
# ※ あくまでサンプルデータの人間性

# In[9]:


# rundom data
random_list = [[random.randint(1, 6) for _ in range(25)] for i in range(5)]
df_sample = pd.DataFrame(random_list, columns=df.columns)

factor_scores = fa_promax.transform(df_sample)  # compute factor scores
pd.DataFrame(factor_scores, columns=['factor_{}'.format(i) for i in range(n_factors)], index=df_sample.index)


# ### sample0
# - factor_1、factor_2の因子がマイナスに大きい負荷がある。
# - factor_0、factor_3、factor_4がプラスの因子であるがfactor_3、factor_4の影響は小さい
# 
# \>>>　外向性はなく勤勉性・誠実性にも欠けるが、精神的にネガティブな事を考えることは少ない傾向
# 
# ### sample1
# - factor_2の因子がマイナスに大きい負荷があり、factor_１、factor_3も同様にマイナスの因子である
# - factor_0、factor_4がプラスの因子であるが絶対値が小さい
# 
# \>>>　勤勉性・誠実性に欠けている
# 
# ### sample2
# - 全ての因子がマイナスだがfactor_3の因子の絶対値が他と比べて大きい
# 
# \>>>協調性・調和性に欠けている
# 
# ### sample3
# - factor_0以外はマイナスの因子負荷量でfactor_3の影響が大きい
# - factor_0は絶対値が小さいので性格を現すまでに至らない可能性がある
# 
# \>>>協調性・調和性に欠けているが、sample2と比べて他の因子の性格は穏やかである
# 
# ### sample4
# - factor_3以外はプラスの因子負荷量で突出した因子はない
# - factor_4の絶対値が小さいので無視できる要素である
# 
# \>>>協調性・調和性には欠けるが、真面目で明るい性格だと捉えれられる
