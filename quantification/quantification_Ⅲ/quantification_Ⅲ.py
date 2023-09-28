#!/usr/bin/env python
# coding: utf-8

# In[1]:


from adjustText import adjust_text

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv('qualitization_wanna_buy.csv')
display(df.head())


# Meanings
# 
# - 購買意欲: how the respondant want to buy the item
#     - ◯: want to buy
#     - △: medium
#     - ×: don't want to buy
# - 容積: volume
#     - 1l: 1l
#     - 500ml: 500ml
#     - 300ml: 300ml
# - 形: shape
#     - 円柱: cylinder
#     - 4角柱: quadangular-prism
# - 色: color
#     - 赤: Red
#     - 緑: Green
#     - 青: Blue
# - 回答者: respondant
# 

# ## apply quantification_III

# In[3]:


df['item_type'] = df['容量']+ '_' + df['形'] + '_' + df['色'] # 商品の特徴の組み合わせを列に追加
df['response'] = (df['購買意欲']=='◯').astype('int') # 購買意欲が'○'を'1'に変換して（その他は0）列に追加
df_response = df.set_index(['回答者', 'item_type'])['response'].unstack() # 回答者→index, item_type→column
df_response


# In[4]:


# 値が0のみの行・列がある場合（逆行列が存在しないので）計算ができないので除外
# >>>そもそも完全に1もしくは0の場合趣味嗜好の偏りがなく特徴がないので分析に使用できない
df_response = df_response.loc[df_response.sum(axis=1)>0, :]
df_response = df_response.loc[:, df_response.sum(axis=0)>0]
df_response


# In[5]:


nda_respondent = np.diag(df_response.sum(axis=1))
nda_response = df_response.values
nda_item = np.diag(df_response.sum(axis=0))

nda_respondent_half_inv = np.diag(df_response.sum(axis=1)**(-1/2))
nda_item_half_inv = np.diag(df_response.sum(axis=0)**(-1/2))

nda_standardized_response = nda_respondent_half_inv.dot(nda_response).dot(nda_item_half_inv)

print('========', 'nda_respondent', '========')
display(nda_respondent)
print('========', 'nda_response', '========')
display(nda_response)
print('========', 'nda_item', '========')
display(nda_item)
# print('========', 'nda_respondent_half_inv', '========')
# display(nda_respondent_half_inv)
# print('========', 'nda_item_half_inv', '========')
# display(nda_item_half_inv)
# print('========', 'nda_standardized_response', '========')
# display(nda_standardized_response)


# In[6]:


nda_respondent = np.diag(df_response.sum(axis=1)) #　回答者のresponseの合計を対角行列に
nda_response = df_response.values
nda_item = np.diag(df_response.sum(axis=0)) #　item_typeのresponseの合計を対角行列に

nda_respondent_half_inv = np.diag(df_response.sum(axis=1)**(-1/2)) #　回答者のresponseの-1/2乗合計を対角行列に
nda_item_half_inv = np.diag(df_response.sum(axis=0)**(-1/2)) #　item_typeのresponseの-1/2乗合計を対角行列に

nda_standardized_response = nda_respondent_half_inv@(nda_response)@(nda_item_half_inv)

np.set_printoptions(linewidth=10000)
print('\n========', 'nda_respondent', '========')
display(nda_respondent)
print('\n========', 'nda_response', '========')
display(nda_response)
print('\n========', 'nda_item', '========')
display(nda_item)
print('\n========', 'nda_respondent_half_inv', '========')
display(nda_respondent_half_inv)
print('\n========', 'nda_item_half_inv', '========')
display(nda_item_half_inv)
print('\n========', 'nda_standardized_response', '========')
display(nda_standardized_response)


# In[7]:


# 特異行列を取得
u, s, vh = np.linalg.svd(nda_standardized_response)

# qualitization vector
qualitization_vector_respondent = nda_respondent_half_inv@(u)
qualitization_vector_item = nda_item_half_inv@(vh.T)

# eigen values
eigen_values = s**2

df_qual_vec_respondent = pd.DataFrame(qualitization_vector_respondent, index=df_response.index)
df_qual_vec_item = pd.DataFrame(qualitization_vector_item, index=[k for k in df_response.columns])

print('========', 'df_qual_vec_respondent', '========')
display(df_qual_vec_respondent)
print('========', 'df_qual_vec_item', '========')
display(df_qual_vec_item)


# ## visualize

# In[8]:


# 相関係数を最大にする配置の散布図

df_with_1st_qual = df.merge(df_qual_vec_respondent[[1]], how='left', left_on='回答者', right_index=True)
df_with_1st_qual = df_with_1st_qual.merge(df_qual_vec_item[[1]], how='left', left_on='item_type', right_index=True)
df_with_1st_qual = df_with_1st_qual.rename(columns={'1_x': 'respondent_1st_qual', '1_y': 'item_1st_qual'})

plt.scatter(df_with_1st_qual['respondent_1st_qual'], df_with_1st_qual['item_1st_qual'], alpha=0.2)

plt.xlabel('Respondent')
plt.ylabel('Item')

#　枠線を非表示
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)

# 補助線の削除
plt.tick_params(bottom=False, left=False, right=False, top=False) 

plt.axhline(y=0, color='gray', alpha=0.5, linewidth=1, linestyle='--')
plt.axvline(x=0, color='gray', alpha=0.5, linewidth=1, linestyle='--')


# ## scatter plot of respondents and items

# In[9]:


# 回答者の第1、第2固有値をプロット

plt.figure(figsize=(20, 15))
plt.scatter(df_qual_vec_respondent[1], df_qual_vec_respondent[2], s=100)
texts = [plt.text(df_qual_vec_respondent[1][i], df_qual_vec_respondent[2][i], df_qual_vec_respondent.index[i], fontsize=18, ha='center', va='center') for i in range(len(df_qual_vec_respondent))]
adjust_text(texts)

plt.axhline(y=0, color='gray', alpha=0.5, linewidth=1, linestyle='--')
plt.axvline(x=0, color='gray', alpha=0.5, linewidth=1, linestyle='--')

#　枠線を非表示
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)

# 補助線の削除
plt.tick_params(bottom=False, left=False, right=False, top=False) 

print("visualization of respondents")


# In[10]:


map_shape = {"円柱":"cyl", "4角柱":"quad"}
map_color = {'赤': 'R', '緑': 'G', '青': 'B'}

def label_ja2en(s):
    l = s.split('_')
    return '_'.join([l[0]] + [map_shape[l[1]]] + [map_color[l[2]]])


# In[11]:


# アイテムの第1、第2固有値をプロット
plt.figure(figsize=(20, 15))
plt.scatter(df_qual_vec_item[1], df_qual_vec_item[2], s=100)
texts = [plt.text(df_qual_vec_item[1][i], df_qual_vec_item[2][i], label_ja2en(df_qual_vec_item.index[i]), fontsize=18, ha='center', va='center') for i in range(len(df_qual_vec_item))]
adjust_text(texts)

plt.axhline(y=0, color='gray', alpha=0.5, linewidth=1, linestyle='--')
plt.axvline(x=0, color='gray', alpha=0.5, linewidth=1, linestyle='--')

# 枠線を非表示
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)

# 補助線の削除
plt.tick_params(bottom=False, left=False, right=False, top=False)

print("visualization of items")


# In[12]:


# 回答者とアイテムの第1、第2固有値をプロット
plt.figure(figsize=(20,15))
plt.scatter(df_qual_vec_respondent[1], df_qual_vec_respondent[2], s=100, c='r')
texts = [plt.text(df_qual_vec_respondent[1][i], df_qual_vec_respondent[2][i], df_qual_vec_respondent.index[i], fontsize=18, ha='center', va='center') for i in range(len(df_qual_vec_respondent))]
adjust_text(texts)


plt.scatter(df_qual_vec_item[1], df_qual_vec_item[2], s=100, c='g')
texts = [plt.text(df_qual_vec_item[1][i], df_qual_vec_item[2][i], label_ja2en(df_qual_vec_item.index[i]), fontsize=18, ha='center', va='center') for i in range(len(df_qual_vec_item))]
adjust_text(texts)

plt.axhline(y=0, color='gray', alpha=0.5, linewidth=1, linestyle='--')
plt.axvline(x=0, color='gray', alpha=0.5, linewidth=1, linestyle='--')

# 枠線を非表示
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)

# 補助線の削除
plt.tick_params(bottom=False, left=False, right=False, top=False)

print("visualiztion of respondents")


# In[ ]:




