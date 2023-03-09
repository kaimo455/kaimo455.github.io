---
title: Formula Derivation
author: Kai Mo
date: 2023-03-09 00:19:00 +0800
categories: [math, machine learning, deep learning]
tags: [math]
math: true
mermaid: true
---

- [1 Linear Regression](#1-linear-regression)
  - [1.1 Model formula and objective function](#11-model-formula-and-objective-function)
  - [1.2 Calculate $R^2$](#12-calculate-r2)

## 1 Linear Regression

### 1.1 Model formula and objective function

Assume the linear regression is:

$$
y_{i}=\beta_0+\beta_1 x_{i}+\epsilon_{i}
$$

OLS loss is which is need to minimized:

$$
loss = \sum_{i=1}^{N}[\ y_i - (\beta_0 + \beta_1 x_i + \epsilon_i) ]
$$

So the closed-form of $\beta_0$ and $\beta_1$ are:

$$
\begin{aligned}
\hat{\beta_0} &= \bar{y}-\hat{\beta_1} \bar{x}\\
\hat{\beta_1} &= \frac{\sum_{i=1}^{N}(x_i-\bar{x})(y_i-\bar{y})}{\sum_{i=1}^{N}(x_i-\bar{x})^2}
\end{aligned}
$$

### 1.2 Calculate $R^2$

$$
R^2 = 1 - \frac{RSS}{TSS} = \frac{ESS}{TSS}
$$

Proving $1 - \frac{RSS}{TSS} = \frac{ESS}{TSS}$:

- step 1:

  $$
  \left(y_{i}-\bar{y}\right)=\left(y_{i}-\hat{y}_{i}\right)+\left(\hat{y}_{i}-\bar{y}\right)\\
  \sum_{i=1}^{n}\left(y_{i}-\bar{y}\right)^{2}=\sum_{i=1}^{n}\left(y_{i}-\hat{y}_{i}\right)^{2}+\sum_{i=1}^{n}\left(\hat{y}_{i}-\bar{y}\right)^{2}+\sum_{i=1}^{n} 2\left(\hat{y}_{i}-\bar{y}\right)\left(y_{i}-\hat{y}_{i}\right)
  $$

  What we need is to prove $\sum_{i=1}^{n} 2\left(\hat{y}_{i}-\bar{y}\right)\left(y_{i}-\hat{y}_{i}\right) = 0$

- step 2:

  For $\hat{y}_{i}-\bar{y}$:

  $$
  \begin{aligned}
  &\hat{y}_{i}=\hat{\beta_0}+\hat{\beta_1} x_{i}\\
  &\bar{y}=\hat{\beta_0}+\hat{\beta_1} \bar{x}\\
  &\hat{y}_{i}-\bar{y}=\hat{\beta_1}\left(x_{i}-\bar{x}\right)
  \end{aligned}
  $$

  For $y_{i}-\hat{y}_{i}$:

  $$
  \begin{aligned}
  y_{i}-\hat{y}_{i}&=\left(y_{i}-\bar{y}\right)-\left(\hat{y}_{i}-\bar{y}\right)\\
  &=\left(y_{i}-\bar{y}\right)-\hat{\beta_1}\left(x_{i}-\bar{x}\right)
  \end{aligned}
  $$

  Finally:

  $$
  \begin{aligned}
    \sum_{i=1}^{n} 2\left(\hat{y}_{i}-\bar{y}\right)\left(y_{i}-\hat{y}_{i}\right) &= \sum_{i=1}^{n} 2 \hat{\beta_1}\left(x_{i}-\bar{x}\right)\left(y_{i}-\hat{y}_{i}\right)\\
    &= \sum_{i=1}^{n} 2 \hat{\beta_1}\left(x_{i}-\bar{x}\right)\left(\left(y_{i}-\bar{y}\right)-\hat{\beta_1}\left(x_{i}-\bar{x}\right)\right)\\
    &= 2 \hat{\beta_1} \left(\sum_{i=1}^{n}\left(x_{i}-\bar{x}\right)\left(y_{i}-\bar{y}\right) - \sum_{i=1}^{n}\hat{\beta_1}\left(x_{i}-\bar{x}\right)^2 \right)\\
    &= 2 \hat{\beta_1} \left(\sum_{i=1}^{n}\left(x_{i}-\bar{x}\right)\left(y_{i}-\bar{y}\right) - \sum_{i=1}^{n}\left(x_{i}-\bar{x}\right)^2 \frac{\sum_{i=1}^{n}(x_i-\bar{x})(y_i-\bar{y})}{\sum_{i=1}^{n}(x_i-\bar{x})^2}\right)\\
    & = 2 \hat{\beta_1}\left(0\right) = 0
  \end{aligned}
  $$


