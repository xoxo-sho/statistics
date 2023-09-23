# 数量化I類

- ### algorithm<br><br>

$$
\large
\begin{align}
Y &= \boldsymbol{a}^T X = a_1 X_1 + \cdots + a_N X_N
\end{align}
$$

<br>

$$
\begin{align}
&\boldsymbol{a} = \left(
a_1 \cdots + a_N
\right)
\quad : \quad regression \quad coefficient \\
\\
&X = \left(
X_1 , \cdots, X_N 
\right) \quad : \quad  explanatory \quad variables \\
\\
&MSE = \frac{1}{N} \sum_{i=1}^{n} \left( y_i -\left(a_{i1} + \cdots + a_{iN} \right) \right)^2
\end{align}
$$

<br>

- カテゴリ変数を説明変数として、他の変数への影響を調べる。<br>
\>>> one-hot本数を用いた重回帰分析

- 目的変数 $Y$ の特徴を回帰係数 $\boldsymbol{a_i}$ が良く表すようにする<br>
\>>> MSEを最小にする。（MSE / V[Y]が小さい時に近似が上手くいっている）<br>
\>>>最小2乗法


## dataset (car-price-prediction/CarPrice_Assignment.csv)
price of cars （205rows）

## Description of dataset
Based on various market surveys, the consulting firm has gathered a large data set of different types of cars across the America market.
