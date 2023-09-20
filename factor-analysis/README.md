# Factor Analysis
- ### algorithm<br><br>

$$
X = \Lambda f + \varepsilon
$$

<br>

$$
X = \left(
\begin{matrix}
X_{1}\\
\vdots\\
X_{n}
\end{matrix}
\right) \quad : \quad Data
\quad , \quad \Lambda =
\begin{pmatrix} 
  \Lambda_{11} & \dots & \Lambda_{1m} \\
  \vdots &  \ddots & \vdots\\
  \Lambda_{n1} & \dots & \Lambda_{nm}
\end{pmatrix} \quad : \quad Factor \quad Loading \quad Matrix
$$

<br>

$$
f = \left(
\begin{matrix}
f_{1}\\
\vdots\\
f_{m}
\end{matrix}
\right) \quad : \quad Common \quad Factor
\quad , \quad \varepsilon = \left(
\begin{matrix}
\varepsilon_{1}\\
\vdots\\
\varepsilon_{n}
\end{matrix}
\right) \quad : \quad Unique \quad Factor
$$
- ### Promax rotatio
    

## dataset (sample_data.csv)
国際パーソナリティ項目プール（International Personality Item Pool）で提供されている、パーソナリティの5因子モデル（IPIP-NEO）についての質問紙への回答（2800人分）

## Description of dataset
25 personality self report items taken from the International Personality Item Pool (ipip.ori.org) were included as part of the Synthetic Aperture Personality Assessment (SAPA) web based personality assessment project. The data from 2800 subjects are included here as a demonstration set for scale construction, factor analysis, and Item Response Theory analysis. Three additional demographic variables (sex, education, and age) are also included. This data set is deprecated and users are encouraged to use bfi.
