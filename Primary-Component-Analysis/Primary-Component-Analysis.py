#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA


# In[2]:


df = pd.read_csv('./sample_data.csv')
display(df.head())
display(df.describe())


# # Preprocessing

# In[3]:


df = df.iloc[:,0:25] # excluding sex,academic background and age
df.info()


# In[4]:


df.dropna(inplace=True) # deleting missing value
df = df.astype('int')
df.info()


# In[5]:


plt.figure(figsize=(15,15))
sns.heatmap(df.corr(), square=True, annot=True)
plt.show()


# # Primary Component Analysis

# In[6]:


PCA = PCA() # インスタンスの生成


# $$
# \large
# \begin{align}
# \boldsymbol{\omega}^T\boldsymbol{\Sigma}\boldsymbol{\omega} = \boldsymbol{\Lambda}
# \end{align}
# $$
# 
# <br>
# 
# $$
# \begin{align}
# &\boldsymbol{\omega} = \left(
# \omega_1 , \cdots , \omega_p \right) \quad, \sqrt{\boldsymbol{\omega}^T\boldsymbol{\omega}} = 1 \quad : \quad Primary \quad Component \quad coefficient \\
# \\
# &\boldsymbol{\Sigma} = \begin{pmatrix} 
#   \sigma_{11} & \dots & \sigma_{1m} \\
#   \vdots &  \ddots & \vdots\\
#   \sigma_{n1} & \dots & \sigma_{nm}
# \end{pmatrix} \quad : \quad variance-covariance \quad matrix \\
# \\
# &\boldsymbol{\Lambda} = \begin{pmatrix}
#   \lambda_{1}                                 \\
#     & \ddots                                  \\
#     &        & \lambda_{p} 
# \end{pmatrix} \quad : \quad sample \quad variance \\
# \end{align}
# $$
# 
# 

# - ωはXに対する分散共分散行列Σの固有ベクトル
# - ωの第一列（ω_1）が第一主成分の係数ベクトルを表す
# - ΛはΣの固有値で、変数を主成分に変換したときの分散に値する
# - Λの第一列（λ_1）が第一主成分の分散を表す

# In[7]:


# データを当てはめる
# 分散共分散行列の固有値と固有ベクトルの決定
PCA.fit(df)


# n個のp次元データ$\left(x_1 , \cdots, x_n \right)^T$の各々と$\boldsymbol{\omega}_1$との内積
# 
# $$
# \begin{align}
# \boldsymbol{\omega}_1^Tx_i, \left(i = 1 , \cdots, n \right)
# \end{align}
# $$
# 
# が第一主成分の主成分得点（principal component score）
# 

# In[8]:


# 主成分得点の値が入っている
df_pca = PCA.transform(df)
# 第一主成分得点　
df_pca[:,1]


# <br>
# $\omega_i$の寄与率　= i番目に大きい固有値　÷ 固有値の合計 = $\omega_i$の分散 ÷ 全分散の合計
# <br>

# In[9]:


# 主成分の寄与りつを並べる
PCA.explained_variance_ratio_


# In[10]:


# 累積寄与率
np.cumsum(PCA.explained_variance_ratio_)


# In[11]:


for i, ratio in enumerate(np.concatenate((np.array([0]), np.cumsum(PCA.explained_variance_ratio_)))):
    print(f'{i} component : {ratio * 100:.2f} % are explained')

fig = plt.figure()
plt.plot(list(range(26)), np.concatenate((np.array([0]), np.cumsum(PCA.explained_variance_ratio_))), 'o-')
plt.xlabel('# of components')
plt.ylabel('explained variance ratio')

#　枠線を非表示
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)

# 補助線の削除
plt.tick_params(bottom=False, left=False, right=False, top=False) 

plt.show()


# ## Discussion
# 
# - 5個の主成分で50%程度の情報をまとめられた
# - 6番目の主成分以降は累積寄与率がなだらかに増加していく
# - パーソナリティの5因子モデルの質問だったた5個の主成分の寄与率が大きいことは想像できるが、半分程度しかデータの説明ができない
# 
# \>>>　因子間での質問内容の相関が高くないこと、似たような質問によって因子間で相関にばらつきがあることが原因と思われる
