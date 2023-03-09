---
title: Machine Learning Interview Summary
author: Kai Mo
date: 2023-03-09 12:22:00 +0800
categories: [reading, machine learning, deep learning]
tags: [reading]
math: true
mermaid: true
---

- [特征工程](#特征工程)
  - [特征归一化](#特征归一化)
  - [类别型特征](#类别型特征)
  - [高维组合特征的处理](#高维组合特征的处理)
  - [特征组合](#特征组合)
  - [文本表示模型](#文本表示模型)
  - [Word2Vec](#word2vec)
  - [图像数据不足时的处理方法](#图像数据不足时的处理方法)
- [模型评估](#模型评估)
  - [评估指标的局限性](#评估指标的局限性)
  - [ROC(Receiver operating Characteristic Curve)曲线](#rocreceiver-operating-characteristic-curve曲线)
  - [余弦距离的应用](#余弦距离的应用)
  - [A/B测试的陷阱](#ab测试的陷阱)
  - [模型评估的方法](#模型评估的方法)
  - [超参数调优](#超参数调优)
  - [过拟合和欠拟合](#过拟合和欠拟合)
- [经典算法](#经典算法)
  - [支持向量机](#支持向量机)
  - [逻辑回归](#逻辑回归)
  - [决策树](#决策树)

# 特征工程

## 特征归一化

- z-score normalization
- standard normalization

## 类别型特征

- ordinal encoding

  sklearn LableEncoding

- one-hot ecncoding
- binary encoding

  $2^n$ to represent all unique values of features, that is, only use $n$ bit to represent all features values. E.g. use $n=2$ to represent 4 values.

## 高维组合特征的处理

从一阶特征进行组合变为多阶特征, e.g. 特征1有两种数值,特征2有三种数值,进行组合之后$2 \times 3 = 6$种数值. 遇到数值种数特别多的情况,可以使用矩阵分解的方式降维.

## 特征组合

如何有效的找到特征组合: 使用决策树,root到leaf的特征可以作为特征组合. 对这些新的特征组合经binary encoding, 1表示拥有这个特征, 0代表没有这种特征.

## 文本表示模型

- Bag of Words

  N-grams

- Term Frequency-Inverse Document Frequency (TF-IDF)

  $IDF = \log{\frac{\#doc}{\#doc\_include\_term + 1}}$

- Topic Model

  主题模型用于从文本库中发现有代表性的主题(得到每个主题上面单词的分布特性)

- Word Embedding

  将每个单词都映射到低维空间上的一个dense vector, 这个$K$ 维向量空间的每个维度方向也可以看作一个隐含的主题, 但是没有主题模型那么直观

## Word2Vec

- Word2Vec
- CBOW

  根据上下文出现的词语来预测当前单词的生成概率

  use Hierarchical Softmax and Negative Sampling to boost training speed(原因时普通的softmax需要对所有单词进行遍历)

- Skip-gram

  根据当前单词来预测上下文中各单词的生成概率

- 隐狄利克雷模型(LDA)

  LDA是利用文档中单词的共现关系来对单词按主题聚类. 主题模型和词嵌入最大的不同在于模型本身, 主题模型是一种基于概率图模型的生成式模型, 其似然函数可以写成若干条件概率相乘的形式.

## 图像数据不足时的处理方法

- 基于模型的方法: 简化模型, non-linear to linear, regularization, ensemble learning, Dropout
- 基于数据的方法: Data Augmentation
  - 旋转, 平移, 缩放, 裁剪, 填充, 左右翻转
  - 加入噪声
  - 颜色变换
  - 改变亮度, 清晰度, 对比度, 锐度
  - 也可以对图像进行特征提取, 然后在图像的特征空间内进行变换, 利用一些通用的数据扩充或者上采样技术. E.g. SMOTE(Synthetic Minority Over-sampling Technique)
  - 或者使用GAN
- 或者使用迁移学习, 不需要重复训练, only fine-tune

# 模型评估

## 评估指标的局限性

- accuracy

  be aware of imbalanced data

- precision

  use top N, precision@N

- recall

  use top N, recall@N

- recall-precision curve

- root mean squared error (RMSE)

  outliers may make RMSE looks bad, so we can use another metric: Mean Absolute Percent Error(MAPE)

  $$MAPE = \sum_{i=1}^{n} |\frac{y_i - \hat{y}_i}{y_i}| \times \frac{100}{n}$$

## ROC(Receiver operating Characteristic Curve)曲线

False Positive Rate(fpr) and True Positive Rate(tpr)

- AUC

- ROC and P-R curve

  ROC曲线有一个特点, 当正负样本分布发生变化时, ROC曲线的形状能够保持基本不变, 而P-R曲线的形状会发生较为剧烈的变化. ROC去想能够尽量降低不同测试集带来的干扰, 更加客观的衡量模型本身的性能

## 余弦距离的应用

总体来说, 余弦距离体现方向上的相对差异, 欧式距离体现的是数值上的绝对差异

## A/B测试的陷阱

- 要进行线上测试
- 如何进行分桶-A/B组

## 模型评估的方法

- holdout
- cross-validation
  - k-fold
  - leave-one-out
- Bootstrap: about $\frac{1}{e} \approx 0.368$

## 超参数调优

- grid search
- random search
- Bayes optimization

## 过拟合和欠拟合

- how to avoid overfitting
  - more data, avoid noise
  - reduce model complexity/flexibility
  - regularization
  - ensemble learning, Bagging
- how to avoid underfitting
  - add new features, methods: 因子分解机, 梯度提升决策树, Deep-crossing
  - increase model complexity/flexibility
  - decrease regularization

# 经典算法

## 支持向量机

- 在空间上线性可分的两类点，分别向SVM分类的超平面上做投影, 这些点在超平面上不是线性可分的
- 一定存在这么一组参数能够使SVM训练误差为0
- 加入松弛变量的SVM的训练误差不一定能够为0, 因为我们的目标函数改变了, 不再是使得训练误差最小.

## 逻辑回归

## 决策树

- ID3 最大信息增益

  对于样本集合D, 类别数为k, 数据集D

  $$H(D) = - \sum_{k=1}^{K} \frac{|C_k|}{|D|} \log_{2} \frac{|C_k|}{|D|}$$

  然后计算某个特征A对于数据集D的经验条件熵$H(D|A)$

  $$H(D|A) = \sum_{i=1}^{n} \frac{|D_i|}{|D|} H(D_i)$$

  $D_i$表示D中特征A取$i$个数值的样本子集

  $$g(D, A) = H(D) - H(D|A)$$

- C4.5 最大信息增益比

  $$g_R(D,A) = \frac{g(D,A)}{H_A(D)}$$

  $$H_A(D) = - \sum_{i=1}^{n} \frac{|D_i|}{|D|} \log_{2} \frac{|D_i|}{|D|}$$

- CART 最大基尼系数(Gini)

  $$Gini(D) = 1 - \sum_{k=1}^{n} \frac{|C_k|}{|D|}$$

  $$Gini(D|A) = \sum_{i=1}^{n} \frac{|D_i|}{D} Gini(D_i)$$

- Pruning
  - Pre-Pruning
  - Post-Pruning
    - reduced error pruning REP
    - pessimistic error pruning PEP
    - cost complexity pruning CCP
    - minimum error pruning MEP
    - critical value pruning CVP
    - optimal pruning OPP
