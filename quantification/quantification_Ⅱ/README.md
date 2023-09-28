# 数量化II類

- ### algorithm<br><br>

$$
\large
\begin{align}
Y &= \boldsymbol{a}^T X = a_1 X_1 + \cdots + a_N X_N
\end{align}
$$

$$
\begin{align}
\sum_{k=1}^{N}\sum_{i=1}^{n_k}\left(y_i^{\left( k \right)} - \bar{y} \right)^2 
= \sum_{k=1}^{N}n_k\left(\bar{y}^{\left( k \right)} - \bar{y} \right)^2
+ \sum_{k=1}^{N}\sum_{i=1}^{n_k}\left(y_i^{\left( k \right)} - \bar{y}^{\left( k \right)} \right)^2 
\end{align}
$$

$$
\begin{align}
S_{total} &= \sum_{k=1}^{N}\sum_{i=1}^{n_k}\left(y_i^{\left( k \right)} - \bar{y} \right)^2 \quad : \quad \text{total deviation sum of squares} \\
S_{between} &= \sum_{k=1}^{N}n_k\left(\bar{y}^{\left( k \right)} - \bar{y} \right)^2 \quad : \quad \text{between-group deviation sum of squares} \\
S_{with} &= \sum_{k=1}^{N}\sum_{i=1}^{n_k}\left(y_i^{\left( k \right)} - \bar{y}^{\left( k \right)} \right)^2 \quad : \quad \text{within-group deviation sum of squares} \\ \\
\eta^2 &= \frac{S_{between}}{S_{total}} \quad : \quad \text{correlation ratio}
\end{align}
$$


- 説明変数も目的変数もカテゴリ変数<br>
\>>> ロジスティック分析との違いは可視化でき、3以上の分類が可能

- 群間の分散を大きくして、郡内の分散を小さくするよなパラメータを見つける<br>
\>>> $\eta^2$ を最大化にする $\boldsymbol{a}$ を求める

## dataset (drug-classification/drug200.csv)
drugs that might be accurate for the patient.(200 raws)

## Description of dataset
The data set contains various information that effect the predictions like Age, Sex, BP, Cholesterol levels, Na to Potassium Ratio and finally the drug type.