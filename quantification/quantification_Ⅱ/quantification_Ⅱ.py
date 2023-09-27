#!/usr/bin/env python
# coding: utf-8

# In[1]:


import zipfile
import shutil

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from adjustText import adjust_text


# In[2]:


# download data form kaggle
get_ipython().system('kaggle datasets download prathamtripathi/drug-classification')


# In[3]:


# unzip
shutil.unpack_archive('数量化理論/数量化II類/drug-classification.zip', 'drug-classification')


# In[4]:


df = pd.read_csv('./数量化理論/数量化II類/drug-classification/drug200.csv')
df.head()


# In[5]:


df.describe()


# ## Preproccessing

# In[6]:


X = pd.get_dummies(df, dtype='uint8')


# In[7]:


standardize_columns = ['Age', 'Na_to_K']
X[standardize_columns] = X[standardize_columns].apply(lambda x : (x - x.mean())/ x.std(), axis=0)
X.head()


# In[8]:


dependent_vars = ['Age', 'Na_to_K',
                  'Sex_M',
                  'BP_LOW','BP_NORMAL',
                  'Cholesterol_NORMAL',]
print(X.columns)
print(dependent_vars)


# ## compute variation matrices

# In[9]:


# total variation

S_total = X[dependent_vars].cov(ddof=0)
S_total


# In[10]:


# "within" and "between" variation
N_Drug_DrugY = X['Drug_DrugY'].sum()
N_Drug_drugA = X['Drug_drugA'].sum()
N_Drug_drugB = X['Drug_drugB'].sum()
N_Drug_drugC = X['Drug_drugC'].sum()
N_Drug_drugX = X['Drug_drugX'].sum()

S_Drug_DrugY = X[X['Drug_DrugY'] == 1][dependent_vars].cov(ddof=0)
S_Drug_drugA = X[X['Drug_drugA'] == 1][dependent_vars].cov(ddof=0)
S_Drug_drugB = X[X['Drug_drugB'] == 1][dependent_vars].cov(ddof=0)
S_Drug_drugC = X[X['Drug_drugC'] == 1][dependent_vars].cov(ddof=0)
S_Drug_drugX = X[X['Drug_drugX'] == 1][dependent_vars].cov(ddof=0)

S_within = (N_Drug_DrugY * S_Drug_DrugY +  N_Drug_drugA * S_Drug_drugA + N_Drug_drugB * S_Drug_drugB + N_Drug_drugC * S_Drug_drugC + N_Drug_drugX * S_Drug_drugX)
S_between = S_total - S_within

S_between


# ## Solve maximizing equation

# In[11]:


np.linalg.eig(np.linalg.inv(S_total)@S_between)


# In[12]:


eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_total)@S_between) # 固有値、固有ベクトルを取得

idx_eta_square = np.argmax(eig_vals) # 最大固有値の列を取得
eta_square = eig_vals[idx_eta_square] # 最大固有値
qualitization_vector = eig_vecs[:, idx_eta_square] #　最大固有ベクトル

print('idx_eta_square', idx_eta_square)
print('eta_square', '{:.3f}'.format(eta_square))
np.set_printoptions(precision=3)
print('qualitization_vector', qualitization_vector)


# ## Compute and Visualize

# In[13]:


df_data_with_y = pd.concat([df[['Drug']], X[dependent_vars]] , axis=1)
df_data_with_y['y'] = X[dependent_vars]@(qualitization_vector)

print('mean of y in each group')
print(df_data_with_y.groupby('Drug')['y'].mean())

mapper_label = {'DrugY':'Y', 'drugA':'A', 'drugB':'B', 'drugC':'C', 'drugX':'X'}
df_data_with_y.groupby('Drug')[['Drug', 'y']].apply(lambda x : plt.hist(x['y'], alpha=0.3, label=mapper_label[x['Drug'].values[0]]))

plt.legend()
plt.show()


# thus, I have the result of quantification-Ⅱ as:
# 
# $$
# \begin{align}
# a_{Age} &=  0.062 \\
# a_{Na\_to\_K} &= 0.966 \\
# a_{Sex\_F} &= 0 \\
# a_{Sex\_M} &= 0.096 \\
# a_{BP\_H} &= 0 \\
# a_{BP\_L} &= -0.223 \\
# a_{BP\_N} &= -0.056 \\
# a_{Cho\_H} &= 0 \\
# a_{Cho\_N} &= 0.016 \\
# \end{align}
# $$
# 
# We can understand the following facts form the above result:
# 
# - Na to Potassium Ration the most influence on DrugY.
# - BP_Low is a likely facter in recomending drugX.
# - Although drugA, drugB and drugC have close figure of 'y', that is in the order of drugB, drugA, drugC.
# - It is reasonable to say that patients who are given DrugY are differ from others.

# ## scatter visualization of data

# In[14]:


eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_total).dot(S_between))

argsort_ev = np.argsort(eig_vals)

idx_eta_square_2 = argsort_ev[-2] # 2番目に大きい固有値の列を取得
eta_square_2 = eig_vals[idx_eta_square_2] # 2番目に大きい固有値
qualitization_vector_2 = eig_vecs[:, idx_eta_square_2] # 2番目に大きい固有ベクトル

print('idx_eta_square_2:', idx_eta_square_2)
print('eta_square_2:', '{:.3f}'.format(eta_square_2))
np.set_printoptions(precision=3)
print('qualitization_vector_2:', qualitization_vector_2)


# In[15]:


df_data_with_y['y2'] = X[dependent_vars]@(qualitization_vector_2)
 
df_data_with_y


# In[16]:


from matplotlib.legend_handler import HandlerTuple
markers = {
    'DrugY':'o',
    'drugA':'>',
    'drugB':'<',
    'drugC':'^',
    'drugX':'x'
}

# 1番目のyの値と2番目のyの値をプロット
for accurate_drug in ['DrugY', 'drugA', 'drugB', 'drugC', 'drugX']:
    data = df_data_with_y[df_data_with_y['Drug'] == accurate_drug]
    plt.scatter(data['y'], data['y2'], s=500, marker=markers[accurate_drug], alpha=0.2, label=accurate_drug)
    

# Set the legend handler to change the marker size only
plt.legend(fontsize='x-small',markerscale=0.3)
plt.show()


# ## scatter visualization of category

# In[17]:


def centerize_qualitization(qual):
    '''
    ダミー変数かした列の固有ベクトルから元の列の固有ベクトル（固有ベクトルの平均)の差を算出

    各列の平均が1なので列のy1、y2が求められる
    '''
    y_age = qual[:1]
    y_NK = qual[1:2]
    y_sex = np.concatenate([[0], qual[2:3]])
    y_bp = np.concatenate([[0], qual[3:5]])
    y_cho = np.concatenate([[0],qual[5:]])
    
    y_sex = y_sex - y_sex.mean()
    y_bp = y_bp - y_bp.mean()
    y_cho = y_cho - y_cho.mean()
    
    return np.concatenate([y_age, y_NK, y_sex, y_bp, y_cho])


# In[18]:


y1 = centerize_qualitization(qualitization_vector)
y2 = centerize_qualitization(qualitization_vector_2)

df_qualitization = pd.DataFrame(
    {
        'category': ['Age', 'Na_to_K', 'Sex_F', 'Sex_M', 'BP_HIGH', 'BP_LOW', 'BP_NORMAL', 'Cholesterol_HIGH', 'Cholesterol_NORMAL'],
        'y1': y1,
        'y2': y2,
    })

df_qualitization


# In[19]:


# y1とy2の軸にカテゴリーをプロット
plt.scatter(df_qualitization['y1'], df_qualitization['y2'], s=100)
texts = [plt.text(df_qualitization['y1'][i], df_qualitization['y2'][i], df_qualitization['category'][i], fontsize=12, ha='center', va='center') for i in range(len(df_qualitization))]
adjust_text(texts)

print("visualiztion of categories")


# I have the result form above chart:
# 
# - DrugY pretty depens on Na_to_k.
# - BP makes a big difference in drugA or drugB and drugC or drugX.

# In[ ]:




