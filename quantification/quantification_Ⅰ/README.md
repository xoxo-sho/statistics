# 数量化I類

- ### algorithm<br><br>

$$
\large
\begin{align}
Y &= \boldsymbol{a}^T X + \boldsymbol{b} = a_1 X_1 + \cdots + a_N X_N + b
\end{align}
$$

<br>

$$
\begin{align}
&\boldsymbol{a} = \left(
a_1 \cdots a_N
\right)
\quad : \quad \text{regression coefficient} \\
\\
&X = \left(
X_1 , \cdots, X_N 
\right) \quad : \quad \text{explanatory variables} \\
\\
&\boldsymbol{b} = \left(
b \cdots b
\right) \quad : \quad \text{constant term} \\
&MSE = \frac{1}{N} \sum_{i=1}^{n} \left( y_i -\left(a_{i1} + \cdots + a_{iN} \right) \right)^2
\end{align}
$$

<br>

- カテゴリ変数を説明変数として、他の変数への影響を調べる。<br>
\>>> one-hot変数を用いた重回帰分析

- 定数項bは全目的変数共通<br>
\>>> 平均的な値を補う数字（今回はprice）

- 目的変数 $Y$ の特徴を回帰係数 $\boldsymbol{a_i}$ が良く表すようにする<br>
\>>> MSEを最小にする。（MSE / V[Y]が小さい時に近似が上手くいっている）<br>
\>>>最小2乗法

- one-hotした説明変数は多重共線性を生む（元の説明変数が同じ場合線形の関係）<br>
\>>> 回帰分析とは異なり回帰係数に一意の値を定めるのではなく、解釈しやすいものを選ぶ

<br>

## dataset (car-price-prediction/CarPrice_Assignment.csv)
price of cars （205rows）

## Description of dataset
Based on various market surveys, the consulting firm has gathered a large data set of different types of cars across the America market.
