#!/usr/bin/env python
# coding: utf-8

# # 数量化Ⅰ類
# 
# カテゴリ変数を説明変数として、他の変数への影響を調べる。<br>
# \>>> one-hot本数を用いた重回帰分析
# 
# カテゴリを数値で表すことができる<br>
# \>>> 数量化

# In[1]:


# import library
import zipfile
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn import linear_model
from sklearn.model_selection import train_test_split


# ## Import data

# In[2]:


get_ipython().system('pip3 install kaggle')


# In[3]:


# download data form kaggle
get_ipython().system('kaggle datasets download hellbuoy/car-price-prediction')


# In[4]:


# unzip
shutil.unpack_archive('car-price-prediction.zip', 'car-price-prediction')


# In[5]:


df = pd.read_csv("./car-price-prediction/CarPrice_Assignment.csv")
df.head()


# In[6]:


df.nunique()


# In[7]:


df.info()


# In[8]:


df.describe()


# ## Preprocessing

# In[9]:


df.drop('car_ID' , axis=1, inplace=True)
X = df.iloc[:, :-1]
Y = df.iloc[:, -1]


# In[10]:


X = pd.get_dummies(X, dummy_na=True, dtype='uint8') # カテゴリーデータをダミーデータに変換 
X = X.fillna(X.median()) # 数量データの欠損値を中央値で置換
X.head()


# <br>
# 
# **データの差がない変数は予測に使用できないため、分散が0の列を取り除く**

# In[11]:


# 分散が0の列を取り除く
X = X.iloc[:, [i for i, x in enumerate(X.var()) if x != 0]]
X.head()


# <br>
# 
# **変数のサンプルが1のものはそれだけでyの特徴を表しているため、予測には使用できない**

# In[12]:


# サンプル数が1のカテゴリーデータを取り除く
X = X.iloc[:, [i for i, x in enumerate(X.sum()) if x != 1]]
X.head()


# In[13]:


# エラーを起こす行を除外

safe = [] # モデルが正常に動作する行
drop_list = [] # エラーを起こす行

for i in range(len(X)): # len(X)= 行の長さ
    try:
        clf = linear_model.LinearRegression() # 重回帰分析
        clf.fit(X.iloc[safe + [i], :], Y.iloc[safe + [i]])
        safe.append(i)
    except:
        drop_list.append(i)


# In[14]:


X = X.iloc[safe, :]
Y = Y.iloc[safe]


# ## The first type of quantification method v.1

# In[15]:


result_scores = []

for _ in range(100):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
    clf = linear_model.LinearRegression()
    clf.fit(X_train, Y_train)
    result_scores.append(clf.score(X_test, Y_test))
print('{0:.4f} ± {1:.4f}'.format(np.mean(result_scores), np.std(result_scores)))


# In[16]:


# 回帰係数を調べて目的変数への影響をみる
clf = linear_model.LinearRegression()
clf.fit(X, Y)
coefs = pd.DataFrame(clf.coef_, index=X.columns).sort_values(0, ascending=False)
coefs # 回帰係数を大きい順に並べたもの


# <br>
# 
# **説明変数はダミー変数に変更したため、カラム数が元の説明変数のユニーク分になり解釈が複雑になる**<br>
# \>>> **解釈を用意にするため、元のカテゴリデータの列にまとめる**

# In[17]:


# 同じカテゴリカルデータ由来の列をまとめる
col_group = {}
for col in X.columns:
    a = col.split("_") # 元の説明変数_ユニーク名を"_"で分割
    if len(a) == 2: # ["元の説明変数", "ユニーク名"]
        if a[0] not in col_group.keys():
            col_group[a[0]] = []
        col_group[a[0]].append(a[1])
    else:
        col_group[col] = [col]

#　まとめた列の出力
for k, v in col_group.items():
    print(k,":", v)


# In[18]:


# 同じ列の回帰係数をまとめる
names = []
values = []
for k,v in col_group.items():
    if len(v) == 1:
        coef = coefs.loc[k][0]
        names.append(k)
        values.append([coef])
    else:
        names.append(k)
        ary = []
        for val in v:
            coef = coefs.loc["{}_{}".format(k,val)][0]
            ary.append(coef)
        values.append(ary)


# In[19]:


# 回帰係数を元の同じ列をx軸に置いて比較
for i, y in enumerate(values):
    plt.scatter([i for _ in y], y)
plt.xticks([j for j in range(len(names))], names, rotation=90)
plt.grid()
plt.show()


# **カテゴリ変数は0or1のため回帰係数の絶対値が大きい傾向がある**

# 以下の2点に当てはまるカラムはモデルの説明するのに情報が少ない（もしくは精度を悪くする）ので除外する
# - 相関係数が 0 に近い説明変数は、価格にはあまり影響しない要因である。
# - 相関係数の正負と、回帰係数の正負が一致しているものが、価格への影響が説明できる要因である。正負が逆転している場合は、価格への影響が（少なくとも線形関係としては）説明できない。

# In[20]:


#目的変数に対して影響力のあるカラムをリストアップ

meaningful_cols = [] # 回帰係数と目的変数の相関係数の絶対値が0.5以上のものを目的変数に対して影響力のあるカラムとして定義
for k,v in col_group.items():
    if len(v) != 1: # ダミー変数（カテゴリカル変数）
        x_axis = []
        y_axis = []
        xy_names = []
        for val in v:
            try:
                x_axis.append(coefs.loc["{}_{}".format(k,val)][0])
                y_axis.append(Y[X["{}_{}".format(k,val)] == 1].median()) # ダミー変数が1の時の目的変数Y（price）の中央値
                xy_names.append(val)
            except:
                continue

        corrcoef = np.corrcoef(x_axis, y_axis)[0][1] # 回帰係数と目的変数の相関係数
        plt.title("{}, corrcoef={}".format(k, corrcoef))
        for x, y, name in zip(x_axis, y_axis, xy_names):
            plt.scatter(x, y)
            plt.text(x, y, name)
        plt.xlabel("regression coefficient")
        plt.ylabel("median of Y")
        plt.grid()
        plt.show()

        if abs(np.corrcoef(x_axis, y_axis)[0][1]) >= 0.5:
            meaningful_cols.append(k)
    else:
        corrcoef = np.corrcoef(X[k].values, Y.values)[0][1] # 回帰係数と目的変数の相関係数
        regrcoef = coefs.loc[k][0]                          # 回帰係数
        plt.title("{0}, corrcoef={1:.4f}, regrcoef={2:.4f}".format(k, corrcoef, regrcoef))
        plt.scatter(X[k].values, Y.values)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid()
        plt.show()
        if corrcoef * regrcoef > 0: # 回帰係数と相関係数の正負が一致しているかの判定
            if abs(np.corrcoef(X[k].values, Y.values)[0][1]) >= 0.5:
                meaningful_cols.append(k)


# In[21]:


meaningful_cols


# ## The first type of quantification method of v.2
# **v.1の結果から得た回帰係数と目的変数の相関関係が大きいカラムで再びmodelを作成**

# In[22]:


X = df[meaningful_cols]
X = pd.get_dummies(X, dummy_na=True, dtype='uint8')
X = X.fillna(X.median())
X = X.iloc[:, [i for i, x in enumerate(X.var()) if x != 0]]
X = X.iloc[:, [i for i, x in enumerate(X.sum()) if x != 1]]
Y = df.iloc[:, -1]


# In[23]:


from sklearn import linear_model
safe = []
droplist = []
for i in range(len(X)):
    try:
        clf = linear_model.LinearRegression()
        clf.fit(X.iloc[safe + [i], :], Y.iloc[safe + [i]])
        safe.append(i)
    except:
        droplist.append(i)


# In[24]:


from sklearn.model_selection import train_test_split
result_scores = []
for _ in range(100):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
    clf = linear_model.LinearRegression()
    clf.fit(X_train, Y_train)
    result_scores.append(clf.score(X_test, Y_test))
print('{0:.4f} ± {1:.4f}'.format(np.mean(result_scores), np.std(result_scores)))


# In[25]:


X_std = X.apply(lambda x: (x-x.mean())/x.std(), axis=0) # 説明変数の標準化
X_std.head()


# In[26]:


Y_std = [(y - Y.mean())/Y.std() for y in Y] # 目的変数の標準化


# In[27]:


clf = linear_model.LinearRegression()
clf.fit(X_std, Y_std)
coefs = pd.DataFrame(clf.coef_, index=X_std.columns).sort_values(0, ascending=False)
coefs # 回帰係数を大きい順に並べたもの


# In[28]:


col_group = {}
for col in X_std.columns:
    a = col.split("_")
    if len(a) == 2:
        if a[0] not in col_group.keys():
            col_group[a[0]] = []
        col_group[a[0]].append(a[1])
    else:
        col_group[col] = [col]


# In[29]:


names = []
values = []
for k,v in col_group.items():
    if len(v) == 1:
        coef = coefs.loc[k][0]
        names.append(k)
        values.append([coef])
    else:
        names.append(k)
        ary = []
        for val in v:
            coef = coefs.loc["{}_{}".format(k,val)][0]
            ary.append(coef)
        values.append(ary)


# In[30]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

for i, y in enumerate(values):
    plt.scatter([i for _ in y], y)
plt.xticks([i for i in range(len(names))], names, rotation=90)
plt.grid()
plt.show()
