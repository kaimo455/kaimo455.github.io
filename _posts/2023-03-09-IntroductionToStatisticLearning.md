---
title: Introduction to Statistic Learning Notes
author: kai
date: 2023-03-09 12:21:00 +0800
categories: [reading, machine learning, deep learning]
tags: [reading]
math: true
mermaid: true
---

- [Introduction to Statistical Learning](#introduction-to-statistical-learning)
  - [1. Introduction](#1-introduction)
  - [2. Statistical Learning](#2-statistical-learning)
  - [3. Linear Regression](#3-linear-regression)
  - [4. Classification](#4-classification)
    - [4.2 Why not linear regression](#42-why-not-linear-regression)
    - [4.3 Logistic regression](#43-logistic-regression)
    - [4.x Naive Bayes classifier](#4x-naive-bayes-classifier)
    - [4.4 linear discriminant analysis](#44-linear-discriminant-analysis)
  - [5. Resampling methods](#5-resampling-methods)
    - [5.1 cross-validation](#51-cross-validation)
  - [6. Linear model selection and regularization](#6-linear-model-selection-and-regularization)
    - [6.1 subset selection](#61-subset-selection)
    - [6.2 shrinkage methods](#62-shrinkage-methods)
    - [6.3 dimention reduction methods](#63-dimention-reduction-methods)
  - [7. Moving Beyond Linearity](#7-moving-beyond-linearity)
  - [8. Tree-based methods](#8-tree-based-methods)
    - [8.1 The basics of decision trees](#81-the-basics-of-decision-trees)
    - [8.2 bagging(bootstrap aggregating), random forests, boosting](#82-baggingbootstrap-aggregating-random-forests-boosting)
  - [9. Support Vector Machines](#9-support-vector-machines)
    - [9.1 maximal margin classifier](#91-maximal-margin-classifier)
    - [9.2 support vector classifiers](#92-support-vector-classifiers)
    - [9.3 support vector machines](#93-support-vector-machines)
    - [9.4 SVMs with more than two classes](#94-svms-with-more-than-two-classes)
    - [9.5 relationship to logistic regression](#95-relationship-to-logistic-regression)
  - [10. Unsupervised Learning](#10-unsupervised-learning)
    - [10.1 the challenge of unsupervised learning](#101-the-challenge-of-unsupervised-learning)
    - [10.2 principal components analysis](#102-principal-components-analysis)
    - [10.3 clustering methods](#103-clustering-methods)

# Introduction to Statistical Learning

## 1. Introduction

## 2. Statistical Learning

- 2.1 What is statistical learning
  - why estimate $f$
  - how do we estimate $f$
    - parametric methods
    - non-parametric methods
  - the trade-off between prediction accuracy and model interpretability
    - interpretability versus flexibility/complexity
  - supervised versus unsupervised learning
  - regression versus classification problems
    - quantitative versus qualitative/categorical

- 2.2 assessing model accuracy
  - measuring the quality of fit
    - mean squared errer (MSE)

      $$
      M S E=\frac{1}{n} \sum_{i=1}^{n}\left(y_{i}-\hat{f}\left(x_{i}\right)\right)^{2}
      $$

  - the bias-variance trade-off

    $$
    E\left(y_{0}-\hat{f}\left(x_{0}\right)\right)^{2}=\operatorname{Var}\left(\hat{f}\left(x_{0}\right)\right)+\left[\operatorname{Bias}\left(\hat{f}\left(x_{0}\right)\right)\right]^{2}+\operatorname{Var}(\epsilon)
    $$

  - the classification setting

    error rate:

    $$
    \frac{1}{n} \sum_{i=1}^{n} I\left(y_{i} \neq \hat{y}_{i}\right)
    $$

    - the Bayes classifier

      the Bayes classifier produces the lowest possible test error rate, called the Bayes error rate. However for real data, we do not know the conditional distribution of Y given X, and so computing the Bayes classifier is impossible. Therefore, the Bayes classifier serves as an unattainable gold standard.

    - K-Nearest neighbors

      many approached attempt to estimate the conditional distribution of Y given X, and then classify a given observation to the class with highest estimated probability. One such method is the K-nearest neighbors (KNN) classifier.

## 3. Linear Regression

- 3.1 simple linear regression
  - estimating the coefficients

    $$
    \mathrm{RSS}=e_{1}^{2}+e_{2}^{2}+\cdots+e_{n}^{2}
    $$

    closed-form for coefficients:

    $$
    \begin{aligned} \hat{\beta}_{1} &=\frac{\sum_{i=1}^{n}\left(x_{i}-\overline{x}\right)\left(y_{i}-\overline{y}\right)}{\sum_{i=1}^{n}\left(x_{i}-\overline{x}\right)^{2}} \\ \hat{\beta}_{0} &=\overline{y}-\hat{\beta}_{1} \overline{x} \end{aligned}
    $$

  - assesing the accuracy of the coefficient estimates

    How far off will that single estimate of $\hat{\mu}$ be? In general, computing the standard error of $\hat{\mu}$. There $\mu$ represents the $E(x) = \overline{x}$ and $E(y) = \overline{y}$

    $$
    \operatorname{Var}(\hat{\mu})=\operatorname{SE}(\hat{\mu})^{2}=\frac{\sigma^{2}}{n}
    $$

    we can see how close $\hat{\beta_0}$ and $\hat{\beta_1}$ are to the true values.

    $$
    \operatorname{SE}\left(\hat{\beta}_{0}\right)^{2}=\sigma^{2}\left[\frac{1}{n}+\frac{\overline{x}^{2}}{\sum_{i=1}^{n}\left(x_{i}-\overline{x}\right)^{2}}\right], \quad \operatorname{SE}\left(\hat{\beta}_{1}\right)^{2}=\frac{\sigma^{2}}{\sum_{i=1}^{n}\left(x_{i}-\overline{x}\right)^{2}}
    $$

    In general, $\sigma^2$ is not known, but can be estimated from the data. The estimated $\sigma$ is known as the Residual Standard Error(RSE), and is given by the formula.

    $$
    \mathrm{RSE}=\sqrt{\mathrm{RSS} /(n-2)}
    $$

    then for each coefficient we have the 95% confidence interval.

    $$
    \hat{\beta}_{1} \pm 2 \cdot \operatorname{SE}\left(\hat{\beta}_{1}\right)
    $$

    To test the null hypothesis, we need to determine whether $\hat{\beta_1}$ is sufficiently far from zero that we can be confident that $\beta_1$ is non-zero. Use t-statistic.

    $$
    t=\frac{\hat{\beta}_{1}-0}{\operatorname{SE}\left(\hat{\beta}_{1}\right)}
    $$

  - assessing the accuracy of the model
    - residual standard error (RSE)

      $$
      \mathrm{RSE}=\sqrt{\frac{1}{n-2} \mathrm{RSS}}=\sqrt{\frac{1}{n-2} \sum_{i=1}^{n}\left(y_{i}-\hat{y}_{i}\right)^{2}}
      $$

    - $R^2$ statistic, form of a proportion - the proportion of variance explained.

      $$
      R^{2}=\frac{\mathrm{TSS}-\mathrm{RSS}}{\mathrm{TSS}}=1-\frac{\mathrm{RSS}}{\mathrm{TSS}}
      $$

      TSS measures the total variance in the response Y, can be thought of as the amount of variability inherent in the response before the regression is performed. In contrast, RSS measures the amount of variability that is **left** unexplained after performing the regression. Hence, TSS - RSS measures the amount of variability in the response that is explained by performing the regression.

      $$
      \mathrm{TSS}=\sum\left(y_{i}-\overline{y}\right)^{2}
      $$

- 3.2 multiple linear regression

  $$
  Y=\beta_{0}+\beta_{1} X_{1}+\beta_{2} X_{2}+\cdots+\beta_{p} X_{p}+\epsilon
  $$

  - estimating the regression coefficients

    $$
    \begin{aligned} \mathrm{RSS} &=\sum_{i=1}^{n}\left(y_{i}-\hat{y}_{i}\right)^{2} \\ &=\sum_{i=1}^{n}\left(y_{i}-\hat{\beta}_{0}-\hat{\beta}_{1} x_{i 1}-\hat{\beta}_{2} x_{i 2}-\cdots-\hat{\beta}_{p} x_{i p}\right)^{2} \end{aligned}
    $$

  - some important questions
    - one: is there a relationship between the response and predictors

      us F-statistic to exam null hypothesis

      $$
      F=\frac{(\mathrm{TSS}-\mathrm{RSS}) / p}{\mathrm{RSS} /(n-p-1)}
      $$

      but the F-stats does not have a specific metric for evaluating the degree of association with response, so we need the adjusted R-squre to evalute.

      ![](https://cdn.mathpix.com/snip/images/m8aX7x_3Y7mdb6xpxMiRxYh__7Zg8g_8qOL3Sl742zs.original.fullsize.png)

      if the linear model assumptions are correct, can show that:

      $$
      E\{\mathrm{RSS} /(n-p-1)\}=\sigma^{2}
      $$

      provided $H_0$ is true,

      $$
      E\{(\mathrm{TSS}-\mathrm{RSS}) / p\}=\sigma^{2}
      $$

      Hence when there is no relationship between the response and predictors, one would expect the F-statistic to take on a value close to 1. On the other hand, if $H_a$ is true, then $E\{(\mathrm{TSS}-\mathrm{RSS}) / p\}>\sigma^{2}$, so we expect F to be greater than 1.

    - two: deciding on important variables
      - forward selection
      - backward selection
      - mixed selection

    - three: model fit

      $$
      \mathrm{RSE}=\sqrt{\frac{1}{n-p-1} \mathrm{RSS}}
      $$

    - four: predictions

- 3.3 other considerations in the regression model

  - qualitative predictions
    - predictions with only two levels (0/1 to represent is_xxx)
    - qualitative predictions with more than two levels

  - extensions of the linear model
    - removeing the addictive assumption

      addictive assumption means that the effect of changes in a predictor on the response $Y$ is independent of the values of the other predictors.

      $$
      Y=\beta_{0}+\beta_{1} X_{1}+\beta_{2} X_{2}+\epsilon
      $$

      after removing addictive assumptions

      $$
      Y=\beta_{0}+\beta_{1} X_{1}+\beta_{2} X_{2}+\beta_{3} X_{1} X_{2}+\epsilon
      $$

    - non-linear relationships

  - potential problems

    when fitting a linear regression model to a particular data set, many problems may occur:

    - 1 non-linearity of the response-predictor relationships

      then a simple approach is to use non-linear transformations of the predictors, such as $logX, \sqrt{X}, X^2$

    - 2 correlation of error terms

      the estimated regression coefficients or the fitted values are based on the assumption of uncorrelated error terms. That is, $\epsilon_{1}, \epsilon_{2}, \ldots, \epsilon_{n},$ are uncorrelated.

      As an extreme example, suppose we accidentally doubled our data, lead- ing to observations and error terms identical in pairs. If we ignored this, our standard error calculations would be as if we had a sample of size 2n, when in fact we have only n samples. Our estimated parameters would be the same for the 2n samples as for the n samples, but the confidence intervals would be narrower by a factor of √2!

      In time series data, the adjecent error terms may have close value.

    - 3 non-constant variance of error terms

      e.g. variances of the error terms may increase with the value of the response - heteroscedasticity

    - 4 outliers
    - 5 high-leverage points

      In face, high leverage observations tend to have a sizable impact on the estimated regression line. It is cause for concern if the least squares line is heavily affected by just a couple of observations.

      leverage statistic: a large value of this statistic indicates an observation with high leverage.

      $$
      h_{i}=\frac{1}{n}+\frac{\left(x_{i}-\overline{x}\right)^{2}}{\sum_{i^{\prime}=1}^{n}\left(x_{i^{\prime}}-\overline{x}\right)^{2}}
      $$

    - 6 collinearity

      refers to the situation in which two or more predictor variables are closely related to one another. It can be difficult to determine how each one separately is associated with the response.

      better way to access multi-collinearity is to compute the variance inflation factor(VIF). The smallest possible value for VIF is 1, which indicates the complete absence of collinearity. As a rule of thumb, a VIF value that exceeds 5 or 10 indicates a problematic amount of collinearity.

      $$
      \operatorname{VIF}\left(\hat{\beta}_{j}\right)=\frac{1}{1-R_{X_{j} | X_{-j}}^{2}}
      $$

      $$
      \begin{array}{l}{\text { where } R_{X_{j} | X_{-j}}^{2} \text { is the } R^{2} \text { from a regression of } X_{j} \text { onto all of the other }} \\ {\text { predictors. If } R_{X_{j} | X_{-j}}^{2} \text { is close to one, then collinearity is present, and so }} \\ {\text { the VIF will be large. }}\end{array}
      $$

- 3.5 comparison of linear regression with K-Nearest neighbors

## 4. Classification

### 4.2 Why not linear regression

If using label encoding it will imply an ordering on the outcomes.

However, the dummy variable approach cannot be easily extended to accommodate qualitative response with more than two levels. For these reasons, it is preferable to use a classification method that is truly suited for qualitative response values, such as the ones presented next.

### 4.3 Logistic regression

- the logistic model

  use logisitic function to model $p(X)$ that gives outputs between 0 and 1 for all values.

  $$
  p(X)=\frac{e^{\beta_{0}+\beta_{1} X}}{1+e^{\beta_{0}+\beta_{1} X}}
  $$

  to fit the model, we use maximum likelihood.

  odds:

  $$
  \frac{p(X)}{1-p(X)}=e^{\beta_{0}+\beta_{1} X}
  $$

  the quantity $p(X) / [1 - p(X)]$ is called the odds, and can take on any value between 0 and $\infty$.

  by taking the logarithm of both sides of odds formula:

  $$
  \log \left(\frac{p(X)}{1-p(X)}\right)=\beta_{0}+\beta_{1} X
  $$

  the left side is called the $log-odds$ or $logit$. So in a logistic regression model, increasing $X$ bu one unit changes the log odds by $\beta_1$, or equivalently it multiplies the odds by $e^{\beta_1}$.

- estimating the regression coefficients

  cost function:

  $$
  \operatorname{cost}\left(h_{\theta}(x), y\right)=\left\{\begin{array}{ll}{-\log \left(h_{\theta}(x)\right)} & {\text { if } y=1} \\ {-\log \left(1-h_{\theta}(x)\right)} & {\text { if } y=0}\end{array}\right.
  $$

  $$
  \operatorname{cost}\left(h_{\theta}(x), y\right)=-y \log \left(h_{\theta}(x)\right)-(1-y) \log \left(1-h_{\theta}(x)\right)
  $$

  $$
  J(\theta)=\frac{1}{m} \sum_{i=1}^{m} \operatorname{cost}\left(h_{\theta}\left(x^{(i)}\right), y^{(i)}\right)
  $$

  The function can be formalized using a mathematical equation called a $\text{likelihood function}$ (we take exponential function):

  $$
  \ell\left(\beta_{0}, \beta_{1}\right)=\prod_{i : y_{i}=1} p\left(x_{i}\right) \prod_{i^{\prime} : y_{i}=0}\left(1-p\left(x_{i^{\prime}}\right)\right)
  $$

  z-statistic

  $$
  Z_{\beta_1} = \hat{\beta}_{1} / S E\left(\hat{\beta}_{1}\right)
  $$

- making predictions

  $$
  \hat{p}(X)=\frac{e^{\hat{\beta}_{0}+\hat{\beta}_{1} X}}{1+e^{\hat{\beta}_{0}+\hat{\beta}_{1} X}}=\frac{e^{-10.6513+0.0055 \times 1,000}}{1+e^{-10.6513+0.0055 \times 1,000}}=0.00576
  $$

- multiple logisitic regression

  $$
  \log \left(\frac{p(X)}{1-p(X)}\right)=\beta_{0}+\beta_{1} X_{1}+\cdots+\beta_{p} X_{p}
  $$

  $$
  p(X)=\frac{e^{\beta_{0}+\beta_{1} X_{1}+\cdots+\beta_{p} X_{p}}}{1+e^{\beta_{0}+\beta_{1} X_{1}+\cdots+\beta_{p} X_{p}}}
  $$

  There are dangers and subtleties associated with performing regression involving only a single predictor when other predictors may also be relevant. In general, the phenomenon is known as $confounding$

- logisitic regression for > 2 response classes

  we sometimes wish to classify a response variable that has more than two classes. The two-calss logisitic regression models discussed in the previous sections have multiple-class extensions, but in practive they tend not to be used all that often, One of reasons is that the method we discuss in the next section, $discriminant analysis$ is popular for multiple-class classifciation.

- choosing a threshold

  - Youden's J-statistic (find the threshold that corresponds to the maximum J-statistic)

  $$
  J=\left(\frac{\text{TruePositives}}{\text {True Positives }+\text {False Negatives}}\right)+\left(\frac{\text {True} \text {Negatives}}{\text {True Negatives }+\text {False} \text {Positives}}\right)-1
  $$

  equals to:

  $$
  J = recall/sensitivity + specificity - 1
  $$

### 4.x Naive Bayes classifier

Abstractly, naive Bayes is a conditional probability model: given a problem instance to be classified, represented by a vector $\mathbf{x}=\left(x_{1}, \dots, x_{n}\right)$ representing some n features (independent variables), it assigns to this instance probabilities $p\left(C_{k} | x_{1}, \ldots, x_{n}\right)$ for each of $K$ possible outcomes or classes $C_k$

- Gaussian naice Bayes

  when dealing with continuous data, a typical assumption is taht the continuous values associated with each class are distributed according to a normal distribution.

  $$
  p\left(x=v | C_{k}\right)=\frac{1}{\sqrt{2 \pi \sigma_{k}^{2}}} e^{-\frac{\left(v-\mu_{k}\right)^{2}}{2 \sigma_{k}^{2}}}
  $$

- Multinomial naice Bayes

  with a multinomial event model, samples (features vectors) represent the frequencies with which certain events have been generated by a multinomial $\left(p_{1}, \ldots, p_{n}\right)$ where $p_i$ is the probability that event $i$ occurs. A feature $\mathbf{x}=\left(x_{1}, \dots, x_{n}\right)$ is then a histogram, with $x_i$ counting the number of times event $i$ was observed in a particular instance. this is the event model typically used for document classification, with events representing the occurrence of a word in a single document.
  
  $$
  p\left(\mathbf{x} | C_{k}\right)=\frac{\left(\sum_{i} x_{i}\right) !}{\prod_{i} x_{i} !} \prod_{i} p_{k i} x_{i}
  $$

  the multinomial naive Bayes classifier becomes a linear classifier when expressed in log-space:

  $$
  \begin{aligned} \log p\left(C_{k} | \mathbf{x}\right) & \propto \log \left(p\left(C_{k}\right) \prod_{i=1}^{n} p_{k i}^{x_{i}}\right) \\ &=\log p\left(C_{k}\right)+\sum_{i=1}^{n} x_{i} \cdot \log p_{k i} \\ &=b+\mathbf{w}_{k}^{\top} \mathbf{x} \end{aligned}
  $$

  where $b=\log p\left(C_{k}\right)$ and $w_{k i}=\log p_{k i}$

### 4.4 linear discriminant analysis

- using Bayes's theotem for classification

  $$
  \operatorname{Pr}(Y=k | X=x)=\frac{\pi_{k} f_{k}(x)}{\sum_{l=1}^{K} \pi_{l} f_{l}(x)}
  $$

  let $f_{k}(X) \equiv \operatorname{Pr}(X=x | Y=k)$ denote the $\text{density function}$ of X for an observation that comes from the kth class. let $\pi_k$ represent the overall or prior probability that a randomly chosen observation comes from the kth class.

  however, estimating $f_{k}(X)$ tends to be more challenging, unless aw assume some simple forms for these densities. We refer to $p_k(x)$ as the posterior probability that an observation $X = x$ belongs to the kth class.

- linear discriminant analysis for p=1

  that is, we only have only one predictor. we will classify an observation to the calss for which $P_k(x)$ is greatest, in order to estimate $f_k(x)$, we will first make some assumptions about its form. (we assume that $f_k(x)$ is normal or Gaussian distribution)

  $$
  f_{k}(x)=\frac{1}{\sqrt{2 \pi} \sigma_{k}} \exp \left(-\frac{1}{2 \sigma_{k}^{2}}\left(x-\mu_{k}\right)^{2}\right)
  $$

  Then we have:

  $$
  p_{k}(x)=\frac{\pi_{k} \frac{1}{\sqrt{2 \pi} \sigma} \exp \left(-\frac{1}{2 \sigma^{2}}\left(x-\mu_{k}\right)^{2}\right)}{\sum_{l=1}^{K} \pi_{l} \frac{1}{\sqrt{2 \pi} \sigma} \exp \left(-\frac{1}{2 \sigma^{2}}\left(x-\mu_{l}\right)^{2}\right)}
  $$

  for now, let us further assume that $\sigma_{1}^{2}=\ldots=\sigma_{K}^{2}$, that is. there is a shared variance term across all K classes. And both sides take log(), then we have:

  $$
  \delta_{k}(x)=x \cdot \frac{\mu_{k}}{\sigma^{2}}-\frac{\mu_{k}^{2}}{2 \sigma^{2}}+\log \left(\pi_{k}\right)
  $$

  and we have:

  $$
  x=\frac{\mu_{1}^{2}-\mu_{2}^{2}}{2\left(\mu_{1}-\mu_{2}\right)}=\frac{\mu_{1}+\mu_{2}}{2}
  $$

  in particular, the following estimates are used:

  $$
  \begin{aligned} \hat{\mu}_{k} &=\frac{1}{n_{k}} \sum_{i : y_{i}=k} x_{i} \\ \hat{\sigma}^{2} &=\frac{1}{n-K} \sum_{k=1}^{K} \sum_{i : y_{i}=k}\left(x_{i}-\hat{\mu}_{k}\right)^{2} \\ \hat{\pi}_{k}&=n_{k} / n \end{aligned}
  $$

- linear discriminant analysis for p>1

  multivariate Gaussian (or multivariate normal) distribution

  ![](https://cdn.mathpix.com/snip/images/vNT6lw5HwmPYzREUD5mPOz7MY_z22hcb5VLhiauQUR4.original.fullsize.png)

  to indicate that a p-dimensional random variable X has a multivariate Gaussian distribution, we write $X \sim N(\mu, \Sigma)$. Here $E(X)=\mu$ is the mean of $X$ (a vector with $p$ components), and $\operatorname{Cov}(X)=\Sigma$ is the $p \times p$ covariance matrix of X.

  $$
  f(x)=\frac{1}{(2 \pi)^{p / 2}|\mathbf{\Sigma}|^{1 / 2}} \exp \left(-\frac{1}{2}(x-\mu)^{T} \mathbf{\Sigma}^{-1}(x-\mu)\right)
  $$

  In the case of $p \lt 1$ predictors, the LDA classifier assumes that the observations in the kth calss are drawn from a multivariate gaussian distribution $N\left(\mu_{k}, \mathbf{\Sigma}\right)$, where $\mu_k$ is a class-specific mean vector, and $\Sigma$ is a covariance matrix that is common to all K classes. Then we have (for LDA):

  $$
  \delta_{k}(x)=x^{T} \boldsymbol{\Sigma}^{-1} \mu_{k}-\frac{1}{2} \mu_{k}^{T} \boldsymbol{\Sigma}^{-1} \mu_{k}+\log \pi_{k}
  $$

  by assigning different threshold we can get different sensitivity and specificity. By default it is 0.5 as threshold. Then we can draw ROC and AUC curve

- quadratic discriminant analysis

  compared with LDA, which assumes that the observations within each class are drawn from a multivariate Gaussian distribution with a class-specific mean vector and a covariance matrix that is common to all K classes. QDA provides an alternative approach, similar to LDA, QDA assumes that the observations from each class are drawn from a Gaussian distribution, but each class has its own covariance matrix. That is, $X \sim N\left(\mu_{k}, \mathbf{\Sigma}_{k}\right)$ for kth class.

  Then we have:

  $$
  \begin{aligned} \delta_{k}(x) &=-\frac{1}{2}\left(x-\mu_{k}\right)^{T} \mathbf{\Sigma}_{k}^{-1}\left(x-\mu_{k}\right)-\frac{1}{2} \log \left|\mathbf{\Sigma}_{k}\right|+\log \pi_{k} \\ &=-\frac{1}{2} x^{T} \mathbf{\Sigma}_{k}^{-1} x+x^{T} \mathbf{\Sigma}_{k}^{-1} \mu_{k}-\frac{1}{2} \mu_{k}^{T} \mathbf{\Sigma}_{k}^{-1} \mu_{k}-\frac{1}{2} \log \left|\mathbf{\Sigma}_{k}\right|+\log \pi_{k} \end{aligned}
  $$

  unlike LDA, QDA's quantity $x$ appears as a quadratic function.

  why would one prefer LDA to QDA or vice-versa, it is the bias-variance trade-off.

  For LDA, the estimated covariance matrix requires estimating $p\times(p+1)/2$ parameters

  For QDA, for a total of $K \times p \times (p+1)/2$ parameters

## 5. Resampling methods

### 5.1 cross-validation

- the validation set approach
- leave-one-out cross-validation
- k-fold cross-validation
- bias-variance trade-off for k-fold cross-validation

  from the perspective of bias reduction, it is clear that LOOCV is to be preferred to k-fold CV, cause LOOCV contains more dataset while training.

  It turns out that LOOCV has higher variance than does k-fold CV with $k<n$. Cause when we perform LOOCV, we are in effect averaging the outputs of $n$ fitted models, each of which is trained on an almost identical set of observations, therefore, these outputs are highly correlated with each other.

- the bootstrap

## 6. Linear model selection and regularization

### 6.1 subset selection

- best subset selection

  fit a separate least squares regression for each possible combination of the $p$ predictors.

- stepwise selection

  - forward stepwise selection
  - backward stepwise selection
  - hybrid approaches

    hybrid versions of forward and backward stepwise selection are available, in which variables are added to the model sequentially, in analogy to forward selection. However, after adding each new variable, the method may also any variables that no longer provide an improvement in the model fit.

- choosing the optimal model
  - $C_{p}, \mathrm{AIC}, \mathrm{BIC},$ and Adjusted $R^{2}$

    In particular, the training error will decrease as more variables are included in the model, but the test error may not. Therefore, training set RSS and training set $R^2$ cannot be used to select from among a set of models with different numbers of variables.

    - $C_p$ - minimize

      $$
      C_{p}=\frac{1}{n}\left(\mathrm{RSS}+2 d \hat{\sigma}^{2}\right)
      $$

      where $\hat{\sigma}^2$ is an estimate of the variance of the error $\epsilon$ associated with each response measurement.

    - Akaike information criterion (AIC) - minimize

      $$
      \mathrm{AIC}=\frac{1}{n \hat{\sigma}^{2}}\left(\mathrm{RSS}+2 d \hat{\sigma}^{2}\right)
      $$

    - Bayesian information criterion (BIC) - minimize

      $$
      \mathrm{BIC}=\frac{1}{n}\left(\mathrm{RSS}+\log (n) d \hat{\sigma}^{2}\right)
      $$

    - adjusted $R^2$ - maximize

      $$
      \text { Adjusted } R^{2}=1-\frac{\mathrm{RSS} /(n-d-1)}{\mathrm{TSS} /(n-1)}
      $$

      equals to minimize $\frac{\mathrm{RSS}}{n-d-1}$

    - validation and cross-validation

### 6.2 shrinkage methods

- Ridge regression

  $$
  \mathrm{RSS}+\lambda \sum_{j=1}^{p} \beta_{j}^{2}
  $$

- the Lasso

  $$
  \mathrm{RSS}+\lambda \sum_{j=1}^{p}\left|\beta_{j}\right|
  $$

- selecting the tuning parameter

  use grid search with cross-validation to tune hyperparameters, then re-fit using all of the available observations and the selected value of the runing hyperparameters.

### 6.3 dimention reduction methods

Now we explore a class of approaches that transform the predictores and then fit as least squares model using the trasformd variables. We assume $Z_1, Z_2, ..., Z_M$ represent $M<p$ linear combinations of our original $p$ predictors. That is,

$$
Z_{m}=\sum_{j=1}^{p} \phi_{j m} X_{j}
$$

Then fit the linear regression model on there tranformed predictors:

$$
y_{i}=\theta_{0}+\sum_{m=1}^{M} \theta_{m} z_{i m}+\epsilon_{i}, \quad i=1, \ldots, n
$$

The dimension of the problem has been reduced from $p+1$ to $M+1$

$$
\sum_{m=1}^{M} \theta_{m} z_{i m}=\sum_{m=1}^{M} \theta_{m} \sum_{j=1}^{p} \phi_{j m} x_{i j}=\sum_{j=1}^{p} \sum_{m=1}^{M} \theta_{m} \phi_{j m} x_{i j}=\sum_{j=1}^{p} \beta_{j} x_{i j}
$$

where $\beta_{j}=\sum_{m=1}^{M} \theta_{m} \phi_{j m}$

Hence can be thought of as a special case of the original linear regression model - dimension reduction serves to constrain the estimated $\beta_j$ coefficients, since now they must take the form aboved. Selecting a value of $M \ll p$ can significantly reduce the variance of the fitted coefficients. if $M=p$ and all the $Z_m$ are lienarly independent, then the formula aboved poses no constraints(no dimentsion reduction occurs).

How to choose the linear linear combination perameters, i.e. $\phi_{j m}$'s. There are two methods.

- principal components regression (PCA)

  - overview of principal components analysis

    PCA is a technique for reducing the dimension of a $n \times p$ data matrix $X$. The first principal component direction of the data is that along which the observations varu the most.

    some parameters of the linear combination to construct new predictors - component loadings

    e.g. $Z_{1}=0.839 \times(\mathrm{pop}-\overline{\mathrm{pop}})+0.544 \times(\mathrm{ad}-\overline{\mathrm{ad}})$

    this formula with the highest variance, i.e. this is the linear combination for which $\operatorname{Var}\left(\phi_{11} \times(\operatorname{pop}-\overline{\mathrm{pop}})+\phi_{21} \times(\mathrm{ad}-\overline{\mathrm{ad}})\right)$ is maximinzed.

  - the principal components regression(PRC) approach

- partial least squares(PLS)

  PCA captures the directions in an unsupervised way, since the resoinse $Y$ is not sued to help determine the principal component. That is the response does not supervise the identification of the principal components. Consuquently PCA suffers from a drawback: there is no a geurantee that the directions taht best explain the predictors will also be the best directions to use for predicting the response.

  PLS, a supervised alternative to PCA. Similar with PCA, PLS is also a linear combinatino of original predictors.

  We now describe how the first PLS direction is computed. After stan-
  dardizing the $p$ predictors, PLS computes the first direction $Z_{1}$ by setting
  each $\phi_{j 1}$ in $(6.16)$ equal to the coefficient from the simple linear regression
  of $Y$ onto $X_{j} .$ One can show that this coefficient is proportional to the cor-
  relation between $Y$ and $X_{j} .$ Hence, in computing $Z_{1}=\sum_{j=1}^{p} \phi_{j 1} X_{j},$ PLS
  places the highest weight on the variables that are most strongly related
  to the response.

## 7. Moving Beyond Linearity

- Polynomial regression

  For example, a cubic regression uses three variables, $X, X^2, and X^3$.

- Step functions

  Cutting the range of a variable into $K$ distinct regions in order to produce a qualitative variables.

- Regression splines

  Involving dividing the range of $X$ into $K$ distinct regions, within each region, a polynomial function is fit to the data.

- Smoothing splines

  Similar to regression splines, but it smooths splines result from minimizing a residual sum of squares criterion subject to a smoothness penalty.

- Local regression

  Similar to splines, but the regions are allowed to overlap, and indeed they do so in a very smooth way.

- Generalized additive models

  Allowing us to extend the methods above to deal with multiple predictions.

## 8. Tree-based methods

### 8.1 The basics of decision trees

- regression tree

  - prediction via stratification of the feature space
    - we divide the predictor space - that is, the set of possible values for $X_1, X_2, ..., X_p$ into $J$ distinct and non-overlapping regions, $R_1, R_2, ..., R_j$
    - For every observation that falls into the region $R_j$, we make the same prediction, whihch is simply the mean of the response values for the training observations in $R_j$
    
    The goal is to find boxes $R_1, R_2, ..., R_j$ that minimize the RSS:

    $$
    \sum_{j=1}^{J} \sum_{i \in R_{j}}\left(y_{i}-\hat{y}_{R_{j}}\right)^{2}
    $$

    For computationally infeasible to consider every possible partition of the feature space into J boxes, we take a top-down, greedy appraoch that is known as recursive binary splitting.

    In order to perform recursive binary splitting, we first select the predictor $X_j$ and the cutpoint $s$ such that splitting the predictor space into the regions $\{X|X_j < s>\}$ and $\{X|X_j \ge s\}$ leads to the greatest possible reduction in RSS. That is, we consider all predictors $X_1, X_2, ...X_p$ and all possible values of the cutpoint s for each of the predictores, and then choose the predictor and cutpoint such that the resulting tree has the lowest RSS.

    $$
    R_{1}(j, s)=\left\{X | X_{j}<s\right\} \text { and } R_{2}(j, s)=\left\{X | X_{j} \geq s\right\}
    $$

    and we seek the value of $j$ and $s$ that minimize the equation

    $$
    \sum_{i : x_{i} \in R_{1}(j, s)}\left(y_{i}-\hat{y}_{R_{1}}\right)^{2}+\sum_{i : x_{i} \in R_{2}(j, s)}\left(y_{i}-\hat{y}_{R_{2}}\right)^{2}
    $$

    where $\hat{y}_{R_1}$ us the mean response for that training observations in $R_1(j,s)$.

  - tree pruning

    - cost complexity pruning: also known as weakest link pruning: rather than considering every possible subtree, we consider a sequence of trees indexed by a nonnegative tuning parameter $\alpha$

      ![](https://cdn.mathpix.com/snip/images/YeiN2vBj4nooVjqQP0zIwJJmbLmPTs7ZR5DIuIrT-BY.original.fullsize.png)

      for each value of $\alpha$ there corresponds a subtree $T \isin T_0$ such that:
      
      $$
      \sum_{m=1}^{|T|} \sum_{i : x_{i} \in R_{m}}\left(y_{i}-\hat{y}_{R_{m}}\right)^{2}+\alpha|T|
      $$

      is as small as possible.

      Here $|T|$ indicates the number of terminal nodes of the tree T. $\alpha$ is the runing parameter that controls trade-off between the subtree's complexity and its fit to the training data. When $\alpha = 0$ then the subtree $T$ will simply equal $T_0$

- classification trees

  - Gini index

    $$
    G=\sum_{k=1}^{K} \hat{p}_{m k}\left(1-\hat{p}_{m k}\right)
    $$

  - croos-entropy

    $$
    D=-\sum_{k=1}^{K} \hat{p}_{m k} \log \hat{p}_{m k}
    $$

- trees versus linear models
- advantages and disadvantages of trees

  pros:

  - trees are very easy to explain to people. In face, they are even easier to explain than linear regression
  - some people believe that decision trees more closely mirror human decision-making than do the regression and classification approaches
  - trees can be displayed graphically, and are easily interpreted even by a non-expert
  - trees can easily handle qualitative predictions without the need to create dummy variables

  cons:

  - trees generally do not have same level of predictive accuracy as some of the other regression and classification approaches
  - additionally, trees can be very non-robot, in other word, a small change in the data can cause a large change in the final estimated tree

### 8.2 bagging(bootstrap aggregating), random forests, boosting

- bagging

  taking repeated samples from the training dataset, and finally average the predicted values.

  $$
  \hat{f}_{\mathrm{bag}}(x)=\frac{1}{B} \sum_{b=1}^{B} \hat{f}^{* b}(x)
  $$

  we create $B$ bootstrapped training set, and average resulting predictions. There trees are grown deep, and are not pruned. Hence each individual tree has high variance, but low bias. Averaging there $B$ trees reduces the variance.

  - out-of-bag error estimation

    each bagged tree makes use of around 2/3 of the observations, the remaining 1/3 of the observations not used to fit a given bagged tree are regerred to as out-of-bag(OOB) observations. We can predict the response for the $i$th observation using each of the trees in which that observation was OOB. This will yield around B/3 predictions for the $i$th observation. Then we can average these predicted values as the final predicted value.

    It can be shown that with B sufficiently large, OOB error is virtually equivalent to leave-one-out croos-validation error.

  - variable importance measures

    In the case of bagging regression trees, we can record the total amount that the RSS is decreased due to splits over a given predictor, averaged over all B trees.

- random forests

  random forests provide an improvement over bagged trees by way of a small tweak that decorrelates the trees. Based on the bagged trees, when building there decision trees, each time a split in a tree is considered, a random sample of $m$ predictors is chosen as split condidates from the entire set of $p$ predictors. Typically we choose $m \approx \sqrt{p}$.

  random forests overcome this problem(bacause the majority of important predictors, averaging bagged trees(if there are highly correlated) does not lead to as large of a reduction in variance as averaging many uncorrelated quantities.) by forcing each split to consider only a subset of the predictors. Therefore, on average $(p-m)/p$ of the splits will not even consider the strong predictor and so other predictors.

- boosting

  boosting each tree is grown using information from previously grown trees. Boosting does not involve boortstrap samling, instead each tree is fit on a modified version of the original dataset.

  - hyperparameters:
    - number of trees (number of iterations)
    - shrinkage parameter $\lambda$, this controls the rate at which boosting learns
    - the number $d$ of splits in each tree, which controls the complexity of the boosted ensemble. Oftern $d=1$ works well, in which case each tree is a stump, consisting of a single split. In this case, the boosted ensemble is fitting an addictive model, since each term involves only a single variable.

  ![](https://cdn.mathpix.com/snip/images/CXaY5IzGyb-zE1tJ42fszPToHknW2PQSgBX8k8lUkMg.original.fullsize.png)

## 9. Support Vector Machines

### 9.1 maximal margin classifier

- what is a hyperplane?

  the mathematical definition of a hyperplane is quite simple. In two dimensions a hyperplane is defined by the equation:

  $$
  \beta_{0}+\beta_{1} X_{1}+\beta_{2} X_{2}=0
  $$

  be extended to the $p$-dimensional setting:

  $$
  \beta_{0}+\beta_{1} X_{1}+\beta_{2} X_{2}+\ldots+\beta_{p} X_{p}=0
  $$

- classification using a separating hyperplane

  now suppoer that we have $n \times p$ data matrix $X$ that consists of $n$ training observations in $p$-dimensional space, and these observations fall into two classes.

  $$
  x_{1}=\left(\begin{array}{c}{x_{11}} \\ {\vdots} \\ {x_{1 p}}\end{array}\right), \ldots, x_{n}=\left(\begin{array}{c}{x_{n 1}} \\ {\vdots} \\ {x_{n p}}\end{array}\right)
  $$

  assume that it is possible to construct a hyperplane that separetes the training observations perfectly according to thier class labels. Then a separating hyperplane has the property that:

  $$
  \beta_{0}+\beta_{1} x_{i 1}+\beta_{2} x_{i 2}+\ldots+\beta_{p} x_{i p}>0 \text { if } y_{i}=1
  $$

  $$
  \beta_{0}+\beta_{1} x_{i 1}+\beta_{2} x_{i 2}+\ldots+\beta_{p} x_{i p}<0 \text { if } y_{i}=-1
  $$

  or in a more elegant format:

  $$
  y_{i}\left(\beta_{0}+\beta_{1} x_{i 1}+\beta_{2} x_{i 2}+\ldots+\beta_{p} x_{i p}\right)>0
  $$

- the maximal margin classifier

  In general if our data can be perfectly separated using a hyperplane, then there will in fact exist an infinite number of such hyperplanes. However maximal margin hyperplane is the separating hyperplane that is farthest from the training observations. That is, we can comput the distance from each training observation to a given separating hyperplane. The largest such distance is the maximal distance from the observations to the hyperplane, known as the $maximal margin$.

  ![](https://cdn.mathpix.com/snip/images/hq8ojPw_GzTfY2pNX74xiqvTgYZoPPKvJ4rMJUWSBAQ.original.fullsize.png)

  These three observations are known as support vectors, if these points were moved slightly then the maximal margin hyperplane would move as well.

- construction of the maximal margin classifier

  briefly, the maximal margin hyperplane is the solution to the optimization problem

  $$
  \underset{\beta_{0}, \beta_{1}, \ldots, \beta_{p}}{\operatorname{maximize}} M
  $$

  constraint-1:

  $$
  \text { subject to } \sum_{j=1}^{p} \beta_{j}^{2}=1
  $$

  constraint-2:

  $$
  y_{i}\left(\beta_{0}+\beta_{1} x_{i 1}+\beta_{2} x_{i 2}+\ldots+\beta_{p} x_{i p}\right) \geq M \forall i=1, \ldots, n
  $$

  constraint-2 guarantees that each observation will be on the correct side of the huperplane, provided that $M$ is positive.

  since if $\beta_{0}+\beta_{1} x_{i 1}+\beta_{2} x_{i 2}+\ldots+\beta_{p} x_{i p}=0$ defines a hyperplane, then so does $k\left(\beta_{0}+\beta_{1} x_{i 1}+\beta_{2} x_{i 2}+\ldots+\beta_{p} x_{i p}\right)=0$ for any $k \neq 0$, however constraint-1 adds meaning to constraint-2, with this constraint the perpendicular distance from the $i$th observation to the hyperplane is given by:

  $$
  y_{i}\left(\beta_{0}+\beta_{1} x_{i 1}+\beta_{2} x_{i 2}+\ldots+\beta_{p} x_{i p}\right)
  $$

  _NOTE: because we have distance formula:_

  $$
  \operatorname{distance}\left(a x+b y+c=0,\left(x_{0}, y_{0}\right)\right)=\frac{\left|a x_{0}+b y_{0}+c\right|}{\sqrt{a^{2}+b^{2}}}
  $$

  That is, we can divide out the $\sqrt{a^{2}+b^{2}}$, then get the distance from the observation to the hyperplane formula.

  Therefore, the constraint-1 and constraint-2 ensure that each observation is on the correct side of the hyperplane and at least a distance $M$ from the hyperplane. Hence, $M$ represents the margin of our hyperplane, and the optimization problem choose $\beta_{0}, \beta_{1}, \ldots, \beta_{p}$ to maximize $M$.

- the non-separable case

  we can extend the concept of a separating hyperplane in order to develop a hyperplane that almost separetes the classes, using a so-called $\text{soft margin}$.

### 9.2 support vector classifiers

- overview of the support vector classifier

  the support vector classifier, sometimes called a soft margin classifier. The margin is soft because it can be violated by some of the training observations.

- details of the support vector classifier

  it may misclassify a few observations, it is the solution to the optimization problem:

  $$
  \underset{\beta_{0}, \beta_{1}, \ldots, \beta_{p}, \epsilon_{1}, \ldots, \epsilon_{n}}{\operatorname{maximize}} M
  $$

  $$
  \text { subject to } \sum_{j=1}^{p} \beta_{j}^{2}=1
  $$

  $$
  y_{i}\left(\beta_{0}+\beta_{1} x_{i 1}+\beta_{2} x_{i 2}+\ldots+\beta_{p} x_{i p}\right) \geq M\left(1-\epsilon_{i}\right)
  $$

  $$
  \epsilon_{i} \geq 0, \quad \sum_{i=1}^{n} \epsilon_{i} \leq C
  $$

  where $C$ is a nonnegative tuning parameter, we seek to make $M$ as latge as possible. $\epsilon_{1}, \dots, \epsilon_{n}$ are $\text{slack variables}$ that allows individual observations to be on the wrong side of the margin or the hyperplane.

   First of all, the slack variable $\epsilon_i$ tells us where the $i$th observation is located, relative to the hyperplane and relative to the margin. If $\epsilon_i$ = 0 then the $i$th observation is on the correct side of the margin. if $\epsilon_i \gt 0$ then the $i$th observation is on the wrong side of the margin, and if $\epsilon_i \gt 1$ then it is on the wrong side of the hyperplane.

   $C$ bounds the sum of the $\epsilon_i$'s and so it determines the number and severity of the violations to the margin and to the hyperplane. For $C = 0$ there is no violations to the margin. For $C \gt 0$ no mmre than $C$ observations can be on the wrong side of the hyperplane, because for each $\epsilon_i$ if we want $i$th observation not to violate the hyperplane, we need to make the $\epsilon_i \lt 1$.

   $C$ controls the bias-variance trade-off of the statistical learning technique. When $C$ is small we seek narrow margins that are rareky violated which may have low bias but high variance.

### 9.3 support vector machines

- classification with non-linear decision boundaries

  we could address the problem of possibly non-linear boundaries between classes in a similar way, by enlarging the feature space using higher-order polunomial functions of the predictors. For instance, rather than fitting a support vector classifier using $p$ features: $X_{1}, X_{2}, \ldots, X_{p}$, we could instead fit a support vector classifier using $2p$ features: $X_{1}, X_{1}^{2}, X_{2}, X_{2}^{2}, \ldots, X_{p}, X_{p}^{2}$. Then would become:

  $$
  \operatorname{maximize}_{\beta_{0}, \beta_{11}, \beta_{12} \ldots, \beta_{p 1}, \beta_{p 2}, \epsilon_{1}, \ldots, \epsilon_{n}} M
  $$

  $$
  \text { subject to } y_{i}\left(\beta_{0}+\sum_{j=1}^{p} \beta_{j 1} x_{i j}+\sum_{j=1}^{p} \beta_{j 2} x_{i j}^{2}\right) \geq M\left(1-\epsilon_{i}\right)
  $$

  $$
  \sum_{i=1}^{n} \epsilon_{i} \leq C, \quad \epsilon_{i} \geq 0, \quad \sum_{j=1}^{p} \sum_{k=1}^{2} \beta_{j k}^{2}=1
  $$

  of course we can increase the order instead of suing $p=2$

- the support vector machine

  the support vector machine is an extension of the support vector classifier that results from enlarging the feature space in a specific way, i.e. using kernels.

  it turns out that the solution to the support vector classifier problem involves onl the $\text{inner products}$ of the observations. The inner products of two $r$-vectors $a$ and $b$ is defined as $\langle a, b\rangle=\sum_{i=1}^{r} a_{i} b_{i}$. Thus the inner product of two observations $x_i$, $x_i^{'}$ is given by:

  $$
  \left\langle x_{i}, x_{i^{\prime}}\right\rangle=\sum_{j=1}^{p} x_{i j} x_{i^{\prime} j}
  $$

  it can be shown that:

    - the linear support vector classifier can be represented as:

      $$
      f(x)=\beta_{0}+\sum_{i=1}^{n} \alpha_{i}\left\langle x, x_{i}\right\rangle
      $$

      where there are $n$ parameters $\alpha_i$, $i=1,...n$, one per training observation

    - $$
      \begin{array}{l}{\text { To estimate the parameters } \alpha_{1}, \ldots, \alpha_{n} \text { and } \beta_{0}, \text { all we need are the }} \\ {\left(\begin{array}{l}{n} \\ {2}\end{array}\right) \text { inner products }\left\langle x_{i}, x_{i}\right\rangle \text { between all pairs of training observations. }} \\ {\text { (The notation }\left(\begin{array}{l}{n} \\ {2}\end{array}\right) \text { means } n(n-1) / 2, \text { and gives the number of pairs. }} \\ {\text { among a set of } n \text { items.) }}\end{array}
      $$

  in order to evaluate the function $f(x)$ we need to compute the inner product between the new point $x$ and each of the training points $x_i$, however it turns out that $\alpha_i$ is nonzero only for the support vectors in the solution. So if $\mathcal{S}$ is the collection of indices of these support points, we can rewrite and solution function of the this form:

  $$
  f(x)=\beta_{0}+\sum_{i \in \mathcal{S}} \alpha_{i}\left\langle x, x_{i}\right\rangle
  $$

  now suppose that every time the inner product appears in the representation or in a calculation of the solution for the support vector classifier, we replace it with a $generalization$ of the inner product of the form:

  $$
  K\left(x_{i}, x_{i^{\prime}}\right)
  $$

  - linear kernel: where $K$ is some function that we will refer to as a $kernel$, if we take:

    $$
    K\left(x_{i}, x_{i^{\prime}}\right)=\sum_{j=1}^{p} x_{i j} x_{i^{\prime} j}
    $$

    then we back to the support vector classifier. It is known as a linear kernel.

  - polynomial kernel of degree $d$: also we can take:

    $$
    K\left(x_{i}, x_{i^{\prime}}\right)=\left(1+\sum_{j=1}^{p} x_{i j} x_{i^{\prime} j}\right)^{d}
    $$

  - radial kernel: we can take:

    $$
    K\left(x_{i}, x_{i^{\prime}}\right)=\exp \left(-\gamma \sum_{j=1}^{p}\left(x_{i j}-x_{i^{\prime} j}\right)^{2}\right)
    $$

    $\gamma$ is a positive constant, like a inverse of variance(recall Gaussian distribution formula)

    if a given test observation $x^{*}=\left(x_{1}^{*} \ldots x_{p}^{*}\right)^{T}$ is far from a training observation $x_i$ in terms of Euclidean distance, then $\sum_{j=1}^{p}\left(x_{j}^{*}-x_{i j}\right)^{2}$ will be large. so $K\left(x_{i}, x_{i^{\prime}}\right)=\exp \left(-\gamma \sum_{j=1}^{p}\left(x_{i j}-x_{i^{\prime} j}\right)^{2}\right)$ will be very tiny, this means that $x_i$ will play virtually no role in $f(x^{*})$. Recall that the predicted class label for the test observation $x_{*}$ is based on the sign of $f(x^{*})$. This means that the radial kernel has very local behavior.

### 9.4 SVMs with more than two classes

- one-versus-one classification

  we classify a test observation using each of the $\left(\begin{array}{l}{K} \\ {2}\end{array}\right)$ classifiers, and assign the major predicted class to test observation.

- one-versus-all classification

  we train $K$ classifiers and assign the observation to the class for which $\beta_{0 k}+\beta_{1 k} x_{1}^{*}+\beta_{2 k} x_{2}^{*}+\ldots+\beta_{pk}x_p^{*}$ is largest.

### 9.5 relationship to logistic regression

It turns out that one can rewrite the criterion for fitting the support vector classifier $f(X)=\beta_{0}+\beta_{1} X_{1}+\ldots+\beta_{p} X_{p}$ as:

$$
\underset{\beta_{0}, \beta_{1}, \ldots, \beta_{p}}{\operatorname{minimize}}\left\{\sum_{i=1}^{n} \max \left[0,1-y_{i} f\left(x_{i}\right)\right]+\lambda \sum_{j=1}^{p} \beta_{j}^{2}\right\}
$$

![](https://cdn.mathpix.com/snip/images/XhpNKLkFvWKPS5GTDqRzzqz_E21bH9qBeKmlCEs7ucY.original.fullsize.png)

![](https://cdn.mathpix.com/snip/images/uOrAxSHMYpi6-CZ_kEpyuF-XjAzKGiWglx-6y0W2Fls.original.fullsize.png)

$$
L(\mathbf{X}, \mathbf{y}, \beta)=\sum_{i=1}^{n} \max \left[0,1-y_{i}\left(\beta_{0}+\beta_{1} x_{i 1}+\ldots+\beta_{p} x_{i p}\right)\right]
$$

This is known as $\text{hinge loss}$


## 10. Unsupervised Learning

### 10.1 the challenge of unsupervised learning

### 10.2 principal components analysis

$$
\underset{\phi_{11}, \ldots, \phi_{p 1}}{\operatorname{maximize}}\left\{\frac{1}{n} \sum_{i=1}^{n}\left(\sum_{j=1}^{p} \phi_{j 1} x_{i j}\right)^{2}\right\} \text { subject to } \sum_{j=1}^{p} \phi_{j 1}^{2}=1
$$

$$
z_{i 1}=\phi_{11} x_{i 1}+\phi_{21} x_{i 2}+\ldots+\phi_{p 1} x_{i p}
$$

since $\frac{1}{n} \sum_{i=1}^{n} x_{i j}=0$, and $\frac{1}{n} \sum_{i=1}^{n} z_{i 1}^{2}$, then we have the average of the $z_{11}, \ldots, z_{n 1}$ will be zero as well.

![](https://cdn.mathpix.com/snip/images/pFXXcQ0yzWPy1Z5av-cw2iCXXrYjfYsoHZji_tZaJAo.original.fullsize.png)

- another interpretation of principal components

  principal components provide low-dimensional linear surfaces that are closest to the observations.

  ![](https://cdn.mathpix.com/snip/images/l0JQcIJ7JGQYu6kW3lce-t8-PfIIlFcUyns0Fj03O5U.original.fullsize.png)

- more on PCA
  - scaling the variables

    the variables should be contered to have mean zero. Furthermore, the results obtained when we perform PCA will also depend on whether the variables have been individually scaled.

  - uniqueness of the principal components

    each principal component loading vector is unique, up to a sign flip.

  - the proportion of variance explained (proportion of variance explained PVE)

    the total variance present in a dataset:

    $$
    \sum_{j=1}^{p} \operatorname{Var}\left(X_{j}\right)=\sum_{j=1}^{p} \frac{1}{n} \sum_{i=1}^{n} x_{i j}^{2}
    $$

    the variance explainded by the mth principal components is:

    $$
    \frac{1}{n} \sum_{i=1}^{n} z_{i m}^{2}=\frac{1}{n} \sum_{i=1}^{n}\left(\sum_{j=1}^{p} \phi_{j m} x_{i j}\right)^{2}
    $$

    PVE of mth principal componenet is given by:

    $$
    \frac{\sum_{i=1}^{n}\left(\sum_{j=1}^{p} \phi_{j m} x_{i j}\right)^{2}}{\sum_{j=1}^{p} \sum_{i=1}^{n} x_{i j}^{2}}
    $$

  - deciding how many principal componenets to use

    we typically decide on the number of principal componenets required to visualize the data by examing a $\text{scree plot}$

### 10.3 clustering methods

- K-Means clustering

  for cluster $C_k$ is a measure $W(C_k)$ of the amount by which the observations within a cluster differ from each other.

  $$
  \underset{C_{1}, \ldots, C_{K}}{\operatorname{minimize}}\left\{\sum_{k=1}^{K} W\left(C_{k}\right)\right\}
  $$

  so we need t odefine the the within-cluster variance, by far the most common choice involves $\text{squared Euclidian distance}$

  $$
  W\left(C_{k}\right)=\frac{1}{\left|C_{k}\right|} \sum_{i, i^{\prime} \in C_{k}} \sum_{j=1}^{p}\left(x_{i j}-x_{i^{\prime} j}\right)^{2}
  $$

  the optimization problem that defined K-Means clustering:

  $$
  \underset{C_{1}, \ldots, C_{K}}{\operatorname{minimize}}\left\{\sum_{k=1}^{K} \frac{1}{\left|C_{k}\right|} \sum_{i, i^{\prime} \in C_{k}} \sum_{j=1}^{p}\left(x_{i j}-x_{i^{\prime} j}\right)^{2}\right\}
  $$

  ![](https://cdn.mathpix.com/snip/images/sNOdcKyr3t2JjWu6SBge1YSsooRUWXLe78tFsHRVMJQ.original.fullsize.png)

  There will be a local optimum, for this reason, it is important to run the algorithm multiple times from different random initial configuration. Then one selects the best solution, i.e. that for which the object is smallest.

- hierarchical clustering

  - interpreting a dendrogram

    ![](https://cdn.mathpix.com/snip/images/wG-sNUcfQvIM2sjPWPUMCDachSOZzOa1uFBLFdN9zTg.original.fullsize.png)

  - the hierarchical clustering algorithm

    the hierarchical clustering dendrogram is obtained via an extremely simple algorithm, we begin by defining some sort of $dissimilarity$ measure between each pair of observations.

    the concept of dissimilarity between a pair of observations needs to be extended to a pair of groups of observations. This extension is achieved by developing the notion of linkage, which defines the dissimilarity betweem two groups of observations.

    - complete
    - average
    - single
    - centroid

    ![](https://cdn.mathpix.com/snip/images/zjKYvDGFh5-2T8HqdY7i4ZljbPANA7l5hnA-SvL6GG4.original.fullsize.png)

  - choice of dissimilarity measure

    sometimes other dissimilarity measures might be preferred, e.g. $\text{}$correlation-based distance$ considers two observations to be similar if their features are highly correlated.

- practical issures in clustering

  - small decisions with bug consequences

    in order to perform clustering, some decisions must be made.

    - Should the observations or features first be standardized in some way? For instance, maybe the variables should be centered to have mean zero and scaled to have standard deviation one.
    - In the case of hierarchical clustering.
      - What dissimilarity measure should be used?
      - What type of linkage should be used?
      - Where should we cut the dendrogram in order to obtain clusters?
    - In the case of K-means clsutering, how many clusters should we look for in the data?

  - validating the clusters obtained
  - other considerations in clustering

    Then since K- means and hierarchical clustering force every observation into a cluster, the clusters found may be heavily distorted due to the presence of outliers that do not belong to any cluster. Mixture models are an attractive approach for accommodating the presence of such outliers.

    In addition, clustering methods generally are not very robust to pertur- bations to the data. For instance, suppose that we cluster n observations, and then cluster the observations again after removing a subset of the n observations at random. One would hope that the two sets of clusters ob- tained would be quite similar, but often this is not the case!

  - a tempered approach to interpreting the results of clustering

    Therefore, we recommend performing clustering with different choices of these parameters, and looking at the full set of results in order to see what patterns consistently emerge.
