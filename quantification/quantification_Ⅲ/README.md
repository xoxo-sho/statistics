# 数量化Ⅲ類

- ### algorithm<br><br>

$$
\large
\begin{align}
\rho &= \text{Corr} \left( \tilde{a} , \tilde{b}\right) 
= \frac{\text{Cov} \left( \tilde{a} , \tilde{b}\right)}{\sqrt{\vphantom{V[\tilde{a}]V[\tilde{b}]}V[\tilde{a}]} \sqrt{V[\tilde{b}]}}
= \frac{{}^t b Z a}{\sqrt{\vphantom{V[\tilde{a}]V[\tilde{b}]}{}^taXa}\sqrt{\vphantom{V[\tilde{a}]V[\tilde{b}]}{}^tbYb}}
\end{align}
$$

$$
\begin{align}
&\tilde{a} = \left(a_1 \cdots a_n \right) \in R^n 
\quad : \quad\text{level a} \\ \\
&\tilde{b} = \left(b_1 \cdots b_m \right) \in R^m 
\quad : \quad \text{level b} \\ \\
&Z = \begin{pmatrix} 
  a_1  b_1 & \dots & a_1  b_m \\
  \vdots &  \ddots & \vdots\\
  a_n  b_1 & \dots & a_n  b_m
\end{pmatrix} \quad : \quad \text{values} \\ \\
&X = {}^t\mathbb{I}_n Z \quad : \quad \text{diagonal matrix of a}\\ \\
&Y =  Z \mathbb{I}_m \quad : \quad \text{diagonal matrix of b}
\end{align}
$$

<br>

- 対角線にデータを並べて相関の最大化 <br>
\>>> $\rho$ が最大になるa,bを求める <br>

- $\tilde{a}$、$\tilde{b}$の平均が0の条件を設定 <br>
\>>> $V[\tilde{a}]$ = $E[{\tilde{a^2}}]$ = $\frac{1}{N}{}^taZa$（bも同様）

- ${}^taXa$ = ${}^tbYb$ = 1の条件下で ${}^t b Z a$ を最大化する <br>
\>>> Lagrangeの未定乗数法

## dataset (quantification_wanna_buy.csv)
purchasing intention of bottles.(217 rows)

## Description of dataset
This dataset consists the information of bottles (capacity, shape, color) and some intention whether  respondents purchase bottles.

