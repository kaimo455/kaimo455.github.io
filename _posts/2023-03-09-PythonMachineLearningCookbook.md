---
title: Python Machine Learning Cookbook
author: Kai Mo
date: 2023-03-09 00:00:00 +0800
categories: [reading, machine learning, deep learning]
tags: [reading]
math: true
mermaid: true
---

- [Machine Learning with Python cookbook](#machine-learning-with-python-cookbook)
  - [1. vectors, matrices and arrays](#1-vectors-matrices-and-arrays)
    - [1.1 Creating a vecor](#11-creating-a-vecor)
    - [1.2 Creating a matrix](#12-creating-a-matrix)
    - [1.3 Creating a sparse matrix](#13-creating-a-sparse-matrix)
    - [1.4 Selecting elements](#14-selecting-elements)
    - [1.5 Describing a matrix](#15-describing-a-matrix)
    - [1.6 Applying operations to elements](#16-applying-operations-to-elements)
    - [1.7 Finding the maximum and minimum values](#17-finding-the-maximum-and-minimum-values)
    - [1.8 Calculating the average, variance, and standard deviation](#18-calculating-the-average-variance-and-standard-deviation)
    - [1.9 Reshaping arrays](#19-reshaping-arrays)
    - [1.10 Transposing a vector or matrix](#110-transposing-a-vector-or-matrix)
    - [1.11 Flattening a matrix](#111-flattening-a-matrix)
    - [1.12 Finding the rank of a matrix](#112-finding-the-rank-of-a-matrix)
    - [1.13 Calculating the determinant](#113-calculating-the-determinant)
    - [1.14 Getting the Diagonal of a Matrix](#114-getting-the-diagonal-of-a-matrix)
    - [1.15 Calculating the Trace of a Matrix](#115-calculating-the-trace-of-a-matrix)
    - [1.16 Finding Eigenvalues and Eigenvectors](#116-finding-eigenvalues-and-eigenvectors)
    - [1.17 Calculating Dot Products](#117-calculating-dot-products)
    - [1.18 Adding and Subtracting Matrices](#118-adding-and-subtracting-matrices)
    - [1.19 Multiplying Matrices](#119-multiplying-matrices)
    - [1.20 Inverting a Matrix](#120-inverting-a-matrix)
    - [1.21 Generating Random Values](#121-generating-random-values)
  - [2. Loading data](#2-loading-data)
    - [2.1 Loading a Sample Dataset](#21-loading-a-sample-dataset)
    - [2.2 Creating a Simulated Dataset](#22-creating-a-simulated-dataset)
    - [2.3 Loading a CSV File](#23-loading-a-csv-file)
    - [2.4 Loading an Excel File](#24-loading-an-excel-file)
    - [2.5 Loading a JSON File](#25-loading-a-json-file)
    - [2.6 Querying a SQL Database](#26-querying-a-sql-database)
  - [3. Data wrangling](#3-data-wrangling)
    - [3.1 Creating a Data Frame](#31-creating-a-data-frame)
    - [3.2 Describing the Data](#32-describing-the-data)
    - [3.3 Navigating DataFrames](#33-navigating-dataframes)
    - [3.4 Selecting Rows Based on Conditionals](#34-selecting-rows-based-on-conditionals)
    - [3.5 Replacing Values](#35-replacing-values)
    - [3.6 Renaming Columns](#36-renaming-columns)
    - [3.7 Finding the Minimum, Maximum, Sum, Average, and Count](#37-finding-the-minimum-maximum-sum-average-and-count)
    - [3.8 Finding Unique Values](#38-finding-unique-values)
    - [3.9 Handling Missing Values](#39-handling-missing-values)
    - [3.10 Deleting a Column](#310-deleting-a-column)
    - [3.11 Deleting a Row](#311-deleting-a-row)
    - [3.12 Dropping Duplicate Rows](#312-dropping-duplicate-rows)
    - [3.13 Grouping Rows by Values](#313-grouping-rows-by-values)
    - [3.14 Grouping Rows by Time](#314-grouping-rows-by-time)
    - [3.15 Looping Over a Column](#315-looping-over-a-column)
    - [3.16 Applying a Function Over All Elements in a Column](#316-applying-a-function-over-all-elements-in-a-column)
    - [3.17 Applying a Function to Groups](#317-applying-a-function-to-groups)
    - [3.18 Concatenating DataFrames](#318-concatenating-dataframes)
    - [3.19 Merging DataFrames](#319-merging-dataframes)
  - [4. Handling numeric data](#4-handling-numeric-data)
    - [4.1 Rescaling a Feature](#41-rescaling-a-feature)
    - [4.2 Standardizing a Feature](#42-standardizing-a-feature)
    - [4.3 Normalizing Observations](#43-normalizing-observations)
    - [4.4 Generating Polynomial and Interaction Features](#44-generating-polynomial-and-interaction-features)
    - [4.5 Transforming Features](#45-transforming-features)
    - [4.6 Detecting Outliers](#46-detecting-outliers)
    - [4.7 Handling Outliers](#47-handling-outliers)
    - [4.8 Discretizating Features](#48-discretizating-features)
    - [4.9 Grouping Observations Using Clustering](#49-grouping-observations-using-clustering)
    - [4.10 Deleting Observations with Missing Values](#410-deleting-observations-with-missing-values)
    - [4.11 Imputing Missing Values](#411-imputing-missing-values)
  - [5. Handling categorical data](#5-handling-categorical-data)
    - [5.1 Encoding Nominal Categorical Features](#51-encoding-nominal-categorical-features)
    - [5.2 Encoding Ordinal Categorical Features](#52-encoding-ordinal-categorical-features)
    - [5.3 Encoding Dictionaries of Features](#53-encoding-dictionaries-of-features)
    - [5.4 Imputing Missing Class Values](#54-imputing-missing-class-values)
    - [5.5 Handling Imbalanced Classes](#55-handling-imbalanced-classes)
  - [6. Handling Text](#6-handling-text)
    - [6.1 Cleaning Text](#61-cleaning-text)
    - [6.2 Parsing and Cleaning HTML](#62-parsing-and-cleaning-html)
    - [6.3 Removing Punctuation](#63-removing-punctuation)
    - [6.4 Tokenizing Text](#64-tokenizing-text)
    - [6.5 Removing Stop Words](#65-removing-stop-words)
    - [6.6 Stemming Words](#66-stemming-words)
    - [6.7 Tagging Parts of Speech](#67-tagging-parts-of-speech)
    - [6.8 Encoding Text as a Bag of Words](#68-encoding-text-as-a-bag-of-words)
    - [6.9 Weighting Word Importance](#69-weighting-word-importance)
  - [7. Handling dates and times](#7-handling-dates-and-times)
  - [8. Handling Images](#8-handling-images)
  - [9. Dimensionality reduction using feature extraction](#9-dimensionality-reduction-using-feature-extraction)
    - [9.1 reducing features using principal components](#91-reducing-features-using-principal-components)
    - [9.2 reducing features when data is linearly inseparable](#92-reducing-features-when-data-is-linearly-inseparable)
    - [9.3 reducing features by maximizing class separability](#93-reducing-features-by-maximizing-class-separability)
    - [9.4 reducing features using matrix factorization](#94-reducing-features-using-matrix-factorization)
    - [9.5 reducing features on sparse data](#95-reducing-features-on-sparse-data)
  - [10. Dimensionality reduction suing feature selection](#10-dimensionality-reduction-suing-feature-selection)
  - [11. Model evaluation](#11-model-evaluation)
    - [11.1 cross-validating models](#111-cross-validating-models)
    - [11.2 creating a baseline regression model](#112-creating-a-baseline-regression-model)
    - [11.3 creating a baseline classification model](#113-creating-a-baseline-classification-model)
    - [11.4 evaluating binary classifier predictions](#114-evaluating-binary-classifier-predictions)
    - [11.5 evaluating binary classifier thresholds](#115-evaluating-binary-classifier-thresholds)
    - [11.6 evaluating multiclass classifier predictions](#116-evaluating-multiclass-classifier-predictions)
    - [11.7 visualizing a classifier's performance](#117-visualizing-a-classifiers-performance)
    - [11.8 evaluating regression models](#118-evaluating-regression-models)
    - [11.9 evaluating clustering models](#119-evaluating-clustering-models)
    - [11.10 creatiwng a custom evaluation metric](#1110-creatiwng-a-custom-evaluation-metric)
    - [11.11 visualizing the effect of training set size](#1111-visualizing-the-effect-of-training-set-size)
    - [11.12 creating a text report of evalution metrics](#1112-creating-a-text-report-of-evalution-metrics)
    - [11.13 visualizing the effect of hyperparameter values](#1113-visualizing-the-effect-of-hyperparameter-values)
  - [12. Model selection](#12-model-selection)
    - [12.1 selecting best models using exhaustive search](#121-selecting-best-models-using-exhaustive-search)
    - [12.2 selecting best models using randomized search](#122-selecting-best-models-using-randomized-search)
    - [12.3 selecting best models from multiple learning algorithms](#123-selecting-best-models-from-multiple-learning-algorithms)
    - [12.4 selecting best models when preprocessing](#124-selecting-best-models-when-preprocessing)
    - [12.5 speeding up model selection with parallelization](#125-speeding-up-model-selection-with-parallelization)
    - [12.7 evaluating performance after model selection](#127-evaluating-performance-after-model-selection)
  - [13. Linear regression](#13-linear-regression)
    - [13.1 fitting a line](#131-fitting-a-line)
    - [13.2 handling interactive effects](#132-handling-interactive-effects)
    - [13.2 fitting a nonlinear relationship](#132-fitting-a-nonlinear-relationship)
    - [13.4 reading variance with regularization](#134-reading-variance-with-regularization)
    - [13.5 reducing features with Lasso regression](#135-reducing-features-with-lasso-regression)
  - [14. Trees and Forests](#14-trees-and-forests)
    - [14.1 training a decision tree classifier](#141-training-a-decision-tree-classifier)
    - [14.2 training a decision tree regressor](#142-training-a-decision-tree-regressor)
    - [14.3 visualizing a decision tree model](#143-visualizing-a-decision-tree-model)
    - [14.4 training a random forest classifier](#144-training-a-random-forest-classifier)
    - [14.5 training a random forest regressor](#145-training-a-random-forest-regressor)
    - [14.6 identifying important features in random forests](#146-identifying-important-features-in-random-forests)
    - [14.7 selecting important features in random forests](#147-selecting-important-features-in-random-forests)
    - [14.8 handling imbalanced classes](#148-handling-imbalanced-classes)
    - [14.9 controlling tree size](#149-controlling-tree-size)
    - [14.10 improving performance through boosting](#1410-improving-performance-through-boosting)
    - [14.11 evaluating random forests with Oout-of-Bag errors](#1411-evaluating-random-forests-with-oout-of-bag-errors)
  - [15. K-Nearest Neighbors](#15-k-nearest-neighbors)
    - [15.1 finding an observation's nearest neighbors](#151-finding-an-observations-nearest-neighbors)
    - [15.2 creating K-Nearest neighbor classifier](#152-creating-k-nearest-neighbor-classifier)
    - [15.3 identifying the best neighborhood size](#153-identifying-the-best-neighborhood-size)
    - [15.3 creating a redius-based nearest neighbor classifier](#153-creating-a-redius-based-nearest-neighbor-classifier)
  - [16. Logistic Regression](#16-logistic-regression)
    - [16.1 training a binary classifier](#161-training-a-binary-classifier)
    - [16.2 training a multiclass classifier](#162-training-a-multiclass-classifier)
    - [16.3 reducing variance through regularization](#163-reducing-variance-through-regularization)
    - [16.4 training a classifier on very large data](#164-training-a-classifier-on-very-large-data)
    - [16.5 handling imbalanced classes](#165-handling-imbalanced-classes)
  - [17 Support Vector Machines](#17-support-vector-machines)
    - [17.1 training a linear classifier](#171-training-a-linear-classifier)
    - [17.2 handling linearly inseparable classes using kernels](#172-handling-linearly-inseparable-classes-using-kernels)
    - [17.3 creating predicted probabilities](#173-creating-predicted-probabilities)
    - [17.4 identifying support vectors](#174-identifying-support-vectors)
    - [17.5 handling imbalanced classes](#175-handling-imbalanced-classes)
  - [18 Naive Bayes](#18-naive-bayes)
    - [18.1 training a classifier for continuous features](#181-training-a-classifier-for-continuous-features)
    - [18.2 training a classifier for discrete and count features](#182-training-a-classifier-for-discrete-and-count-features)
    - [18.3 training a Naice Bayes classifier for binary features](#183-training-a-naice-bayes-classifier-for-binary-features)
    - [18.4 calibrating predicted probabilities](#184-calibrating-predicted-probabilities)
  - [19 Clustering](#19-clustering)
    - [19.1 clustering using K-Means](#191-clustering-using-k-means)
    - [19.2 speeding up K-Means clustering](#192-speeding-up-k-means-clustering)
    - [19.3 clustering using Meanshift](#193-clustering-using-meanshift)
    - [19.4 clustering using DBSCAN](#194-clustering-using-dbscan)
    - [19.5 clustering using hierarchical merging](#195-clustering-using-hierarchical-merging)

# Machine Learning with Python cookbook

## 1. vectors, matrices and arrays

### 1.1 Creating a vecor

```python
import numpy as np
# Create a vector as a row
vector_row = np.array([1, 2, 3])
# Create a vector as a column
vector_column = np.array([[1], [2], [3]])
```

### 1.2 Creating a matrix

```python
import numpy as np
matrix = np.array([[1, 2], [1, 2], [1, 2]])
matrix_object = np.mat([[1, 2], [1, 2],
```

### 1.3 Creating a sparse matrix

```python
import numpy as np
from scipy import sparse
matrix = np.array([[0, 0], [0, 1], [3, 0]])
matrix_sparse = sparse.csr_matrix(matrix)
```

### 1.4 Selecting elements

### 1.5 Describing a matrix

```python
import numpy as np
matrix = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
matrix.shape => (3,4)
matrix.size => 12
matrix.ndim => 2
```

### 1.6 Applying operations to elements

```python
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
add_100 = lambda i: i + 100
vectorized_add_100 = np.vectorize(add_100)
vectorized_add_100(matrix)
# array([[101, 102, 103],
#        [104, 105, 106],
#        [107, 108, 109]])
```

### 1.7 Finding the maximum and minimum values

### 1.8 Calculating the average, variance, and standard deviation

### 1.9 Reshaping arrays

### 1.10 Transposing a vector or matrix

### 1.11 Flattening a matrix

```python
matrix.flatten()
```

### 1.12 Finding the rank of a matrix

```python
matrix = np.array([[1, 1, 1], [1, 1, 10], [1, 1, 15]]) 
# Return matrix rank
np.linalg.matrix_rank(matrix)
# 2
```

### 1.13 Calculating the determinant

```python
np.linalg.det(matrix)
```

### 1.14 Getting the Diagonal of a Matrix

```python
matrix.diagonal()
```

### 1.15 Calculating the Trace of a Matrix

```python
matrix.trace()
```

### 1.16 Finding Eigenvalues and Eigenvectors

```python
eigenvalues, eigenvectors = np.linalg.eig(matrix)
```

### 1.17 Calculating Dot Products

```python
vector_a = np.array([1,2,3])
vector_b = np.array([4,5,6])
# Calculate dot product
np.dot(vector_a, vector_b)
# 32
```

### 1.18 Adding and Subtracting Matrices

```python
np.add()
np.subtract()
```

### 1.19 Multiplying Matrices

```python
# Create matrix
matrix_a = np.array([[1, 1], [1, 2]])
# Create matrix
matrix_b = np.array([[1, 3], [1, 2]])
# Multiply two matrices
np.dot(matrix_a, matrix_b)
# array([[2, 5], [3, 7]])
```

### 1.20 Inverting a Matrix

```python
np.linalg.inv(matrix)
```

### 1.21 Generating Random Values

## 2. Loading data

### 2.1 Loading a Sample Dataset

### 2.2 Creating a Simulated Dataset

```python
from sklearn.datasets import make_regression
from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
```

### 2.3 Loading a CSV File

### 2.4 Loading an Excel File

### 2.5 Loading a JSON File

### 2.6 Querying a SQL Database

```python
from sqlalchemy import create_engine

# Create a connection to the database
database_connection = create_engine('sqlite:///sample.db')

# Load data
dataframe = pd.read_sql_query('SELECT * FROM data', database_connection)
```

## 3. Data wrangling

### 3.1 Creating a Data Frame

### 3.2 Describing the Data

### 3.3 Navigating DataFrames

### 3.4 Selecting Rows Based on Conditionals

```python
# Filter rows
dataframe[(dataframe['Sex'] == 'female') & (dataframe['Age'] >= 65)]
```

### 3.5 Replacing Values

```python
dataframe['Sex'].replace("female", "Woman")
```

### 3.6 Renaming Columns

```python
dataframe.rename(columns={'PClass': 'Passenger Class'})
```

### 3.7 Finding the Minimum, Maximum, Sum, Average, and Count

### 3.8 Finding Unique Values

```python
df.unique()
df.value_counts()
df.nunique()
```

### 3.9 Handling Missing Values

### 3.10 Deleting a Column

### 3.11 Deleting a Row

### 3.12 Dropping Duplicate Rows

```python
df.drop_duplicates()
```

### 3.13 Grouping Rows by Values

### 3.14 Grouping Rows by Time

```python
# Create date range
time_index = pd.date_range('06/06/2017', periods=100000, freq='30S')
# Create DataFrame
dataframe = pd.DataFrame(index=time_index)
# Create column of random values
dataframe['Sale_Amount'] = np.random.randint(1, 10, 100000)
# Group rows by week, calculate sum per week
dataframe.resample('W').sum()
# Group by two weeks, calculate mean
dataframe.resample('2W').mean()
```

### 3.15 Looping Over a Column

### 3.16 Applying a Function Over All Elements in a Column

### 3.17 Applying a Function to Groups

### 3.18 Concatenating DataFrames

### 3.19 Merging DataFrames

## 4. Handling numeric data

### 4.1 Rescaling a Feature

### 4.2 Standardizing a Feature

### 4.3 Normalizing Observations

```python
from sklearn.preprocessing import Normalizer

# Create feature matrix
features = np.array([[0.5, 0.5], [1.1, 3.4], [1.5, 20.2], [1.63, 34.4], [10.9, 3.3]])
# Create normalizer
normalizer = Normalizer(norm="l2")
normalizer.transform(features)
#  array([[ 0.70710678, 0.70710678], 
#         [ 0.30782029, 0.95144452], 
#         [ 0.07405353, 0.99725427], 
#         [ 0.04733062, 0.99887928],
#         [ 0.95709822, 0.28976368]])
```

### 4.4 Generating Polynomial and Interaction Features

```python
from sklearn.preprocessing import PolynomialFeatures
 # Create feature matrix
features = np.array([[2, 3], [2, 3], [2, 3]])
polynomial_interaction = PolynomialFeatures(degree=2, include_bias=False)
polynomial_interaction.fit_transform(features)
#array([[ 2., 3., 4., 6., 9.], 
#       [ 2., 3., 4., 6., 9.],
#       [ 2., 3., 4., 6., 9.]])
```

### 4.5 Transforming Features

```python
from sklearn.preprocessing import FunctionTransformer
features = np.array([[2, 3], [2, 3], [2, 3]])
# Define a simple function
def add_ten(x): return x + 10
# Create transformer
ten_transformer = FunctionTransformer(add_ten)
ten_transformer.transform(features)
# array([[12, 13],
#        [12, 13],
#        [12, 13]])
```

### 4.6 Detecting Outliers

```python
from sklearn.covariance import EllipticEnvelope
outlier_detector = EllipticEnvelope(contamination=.1)
outlier_detector.fit(features)
outlier_detector.predict(features)
# A major limitation of this approach is the need to specify a contamination parame‐ ter, which is the proportion of observations that are outliers—a value that we don’t know. Think of contamination as our estimate of the cleanliness of our data. If we expect our data to have few outliers, we can set contamination to something small. However, if we believe that the data is very likely to have outliers, we can set it to a higher value.
```

### 4.7 Handling Outliers

```python
# Create feature based on boolean condition
houses["Outlier"] = np.where(houses["Bathrooms"] < 20, 0, 1)
# or log feature
houses["Log_Of_Square_Feet"] = [np.log(x) for x in houses["Square_Feet"]]
```

### 4.8 Discretizating Features

```python
from sklearn.preprocessing import Binarizer

# Create feature
age = np.array([[6], [12], [20], [36], [65]])
# Create binarizer
binarizer = Binarizer(18)
# Transform feature
binarizer.fit_transform(age)
#     array([[0],
#            [0],
#            [1],
#            [1],
#            [1]])
## Second, we can break up numerical features according to multiple thresholds:
np.digitize(age, bins=[20,30,64])
#  array([[0],
#         [0],
#         [1],
#         [2],
#         [3]])
```

### 4.9 Grouping Observations Using Clustering

Create new features with cluster labels

### 4.10 Deleting Observations with Missing Values

```python
df.dropna()
```

### 4.11 Imputing Missing Values

```python
# Load library
from sklearn.preprocessing import Imputer
mean_imputer = Imputer(strategy="mean", axis=0)
features_mean_imputed = mean_imputer.fit_transform(features)
```

## 5. Handling categorical data

### 5.1 Encoding Nominal Categorical Features

You have a feature with nominal classes that has no intrinsic ordering (e.g., apple, pear, banana).

```python
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
# Create one-hot encoder
one_hot = LabelBinarizer()
# One-hot encode feature
one_hot.fit_transform(feature)

# Create multiclass feature
multiclass_feature = [("Texas", "Florida"), ("California", "Alabama"),
                          ("Texas", "Florida"),
                          ("Delware", "Florida"),
                          ("Texas", "Alabama")]
one_hot_multiclass = MultiLabelBinarizer()
one_hot_multiclass.fit_transform(multiclass_feature)
```

### 5.2 Encoding Ordinal Categorical Features

```python
from sklearn.preprocessing import LabelEncoder
```

### 5.3 Encoding Dictionaries of Features

```python
from sklearn.feature_extraction import DictVectorizer
# Create dictionary
data_dict = [{"Red": 2, "Blue": 4}, {"Red": 4, "Blue": 3},
                 {"Red": 1, "Yellow": 2},
                 {"Red": 2, "Yellow": 2}]
dictvectorizer = DictVectorizer(sparse=False)
# Convert dictionary to feature matrix
features = dictvectorizer.fit_transform(data_dict)
# array([[ 4., 2., 0.],
#        [ 3., 4., 0.],
#        [ 0., 1., 2.],
#        [ 0., 2., 2.]])
```

### 5.4 Imputing Missing Class Values

The ideal solution is to train a machine learning classifier algorithm to predict the missing values, commonly a k-nearest neighbors (KNN) classifier:

```python
from sklearn.neighbors import KNeighborsClassifier
# Create feature matrix with categorical feature
X = np.array([[0, 2.10, 1.45], 
              [1, 1.18, 1.33], 
              [0, 1.22, 1.27],
              [1, -0.21, -1.19]])
# Create feature matrix with missing values in the categorical feature
X_with_nan = np.array([[np.nan, 0.87, 1.31], 
                      [np.nan, -0.67, -0.22]])
# Train KNN learner
clf = KNeighborsClassifier(3, weights='distance')
trained_model = clf.fit(X[:,1:], X[:,0])
# Predict missing values' class
imputed_values = trained_model.predict(X_with_nan[:,1:])
# Join column of predicted class with their other features
X_with_imputed = np.hstack((imputed_values.reshape(-1,1), X_with_nan[:,1:]))
# Join two feature matrices
np.vstack((X_with_imputed, X))
# array([[ 0. , 0.87, 1.31], 
#       [ 1. , -0.67, -0.22], 
#       [ 0. , 2.1 , 1.45], 
#       [ 1. , 1.18, 1.33], 
#       [ 0. , 1.22, 1.27],
#       [ 1. , -0.21, -1.19]])
```

An alternative solution is to fill in missing values with the feature’s most frequent value:

```python
from sklearn.preprocessing import Imputer
imputer = Imputer(strategy='most_frequent', axis=0)
imputer.fit_transform(X_complete)
```

### 5.5 Handling Imbalanced Classes

```python
from sklearn.ensemble import RandomForestClassifier
weights = {0: .9, 1: 0.1}
RandomForestClassifier(class_weight=weights)
# or
RandomForestClassifier(class_weight="balanced")
```

Or can use downsample or upsample

```python
# downsample
i_class1_downsampled = np.random.choice(i_class1, size=n_class0, replace=False)
np.hstack((target[i_class0], target[i_class1_downsampled]))
# Join together class 0's feature matrix with the
# downsampled class 1's feature matrix
np.vstack((features[i_class0,:], features[i_class1_downsampled,:]))

# upsample
i_class0_upsampled = np.random.choice(i_class0, size=n_class1, replace=True)
np.concatenate((target[i_class0_upsampled], target[i_class1]))
# Join together class 0's upsampled feature matrix with class 1's feature matrix
np.vstack((features[i_class0_upsampled,:], features[i_class1,:]))
```

## 6. Handling Text

### 6.1 Cleaning Text

```python
str.strip()
str.replace(',', '')
```

### 6.2 Parsing and Cleaning HTML

```python
from bs4 import BeautifulSoup

soup = BeautifulSoup(html_content, 'lxml')
soup.find('div', {'class': "full_name"}).text
```

### 6.3 Removing Punctuation

```python
import unicodedata
import sys

# Create text
text_data = ['Hi!!!! I. Love. This. Song....', '10000% Agree!!!! #LoveIT', 'Right?!?!']

# create a dictionary of punctuation characters
punctuation = dict.fromkeys(i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P'))

[string.translate(punctuation) for string in data]
```

### 6.4 Tokenizing Text

```python
from nltk.tokenize import word_tokenize
word_tokenize(string)

# we can also tokenize into sentences
from nltk.tokenize import sent_tokenize
sent_tokenzie(string)
```

### 6.5 Removing Stop Words

```python
from nltk.corpus import stopwords

stop_words = stopwords.words('english')
[word for word in tokenized_words if word not in stop_words]
```

### 6.6 Stemming Words

```python
from nltk.stem.porter import PorterStemmer

porter = PorterStemmer()
[porter.stem(word) for word in tokenized_words]
```

### 6.7 Tagging Parts of Speech
```python
from nltk import pos_tag
from nltk import word_tokenize

# text_tagged is a tuple of (word, POS)
text_tagged = pos_tag(tokenized_words)
```

Tag Part of speech
  - NNP Proper noun, singular NN Noun, singular or mass RB Adverb
  - VBD Verb, past tense
  - VBG Verb, gerund or present participle JJ Adjective
  - PRP Personal pronoun

```python

# Create text
tweets = ["I am eating a burrito for breakfast",
          "Political science is an amazing field",
          "San Francisco is an awesome city"]
# Create list
tagged_tweets = []
# Tag each word and each tweet
for tweet in tweets:
  tweet_tag = nltk.pos_tag(word_tokenize(tweet)) tagged_tweets.append([tag for word, tag in tweet_tag])
# Use one-hot encoding to convert the tags into features
one_hot_multi = MultiLabelBinarizer()
one_hot_multi.fit_transform(tagged_tweets)
```

### 6.8 Encoding Text as a Bag of Words

```python
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

text_data = np.array(['I love Brazil. Brazil!',
'Sweden is best',
'Germany beats both'])

counter = CountVectorizer()
bow = counter.fit_transform(text_data)

# show feature names
counter.get_feature_names()
```

### 6.9 Weighting Word Importance

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

text_data = np.array(['I love Brazil. Brazil!',
'Sweden is best',
'Germany beats both'])
tfidf = TfidfVectorizer()
feature_matrix = tfidf.fit_transform(text_data)
```

$$
idf(t) = log \frac{1 + h_d}{1+df(d,t)} + 1
$$

Then use L2-norm to normalize the weights

## 7. Handling dates and times

## 8. Handling Images

## 9. Dimensionality reduction using feature extraction

### 9.1 reducing features using principal components

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=0.99, whiten=True)
pca.fit_transform(features)
## whiten=True transforms the values of each principal component so that they have zero mean and unit variance
```

### 9.2 reducing features when data is linearly inseparable

```python
from sklearn.decomposition import KernelPCA

kpca = KernelPCA(kernel='rbf'), gamma=15, n_components=1)
kpca.fit_transform(features)
```

### 9.3 reducing features by maximizing class separability

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(n_components=1)
features_lda = lda.fit(features, target).transform(features)

 # Create function
def select_n_components(var_ratio, goal_var: float) -> int: # Set initial variance explained so far
  total_variance = 0.0
          # Set initial number of features
  n_components = 0
  # For the explained variance of each feature:
  for explained_variance in var_ratio:
  # Add the explained variance to the total
  total_variance += explained_variance # Add one to the number of components
  n_components += 1
  # If we reach our goal level of explained variance
  if total_variance >= goal_var: # End the loop
  break
          # Return the number of components
  return n_components # Run function
      select_n_components(lda_var_ratios, 0.95)
```

### 9.4 reducing features using matrix factorization

```python
## NMF(non-negative matrix fatorization)
from sklearn.decomposition import NMF

# Create, fit, and apply NMF
nmf = NMF(n_components=10, random_state=1)
features_nmf = nmf.fit_transform(features)
```

### 9.5 reducing features on sparse data

```python
## use truncated Singular Value Decomposition (SVD)

from sklearn.decomposition import TruncatedSVD
from sklearn.sparse import csr_matrix

# Make sparse matrix
features_sparse = csr_matrix(features)

# Create a TSVD
tsvd = TruncatedSVD(n_components=10)

# conduct TSVD on sparse matrix
features_sparse_tsvd = tsvd.fit(features_sparse).transform(features_sparse)
```

## 10. Dimensionality reduction suing feature selection

## 11. Model evaluation

### 11.1 cross-validating models

```python
from sklearn.model_selection import Kfold, cross_val_score

cv_results = cross_val_score(estimator, features, target, cv=Kold(n_splits=n), scoring='accuracy')
```

### 11.2 creating a baseline regression model

```python
from sklearn.dummy import DummyRegressor

dummy = DummyRegressor(strategy='mean')
# or
dummy = DummyRegressor(strategy='constant', constant=20)
```

### 11.3 creating a baseline classification model

```python
from sklearn.dummy import DummyClassifier

dummy = DummyClassifier(strategy='uniform', random_state=1)
```

### 11.4 evaluating binary classifier predictions

$$
\text {Accuracy}=\frac{T P+T N}{T P+T N+F P+F N}
$$

$$
\text { Precision }=\frac{T P}{T P+F P}
$$

$$
\text { Recall }=\frac{T P}{T P+F N}
$$

$$
F_{1}=2 \times \frac{\text { Precision } \times \text { Recall }}{\text { Precision }+\text { Recall }}
$$

### 11.5 evaluating binary classifier thresholds

The $Receiving \, Operating \, Characterisitc (ROC)$ curve is a commom method for evaluating the quality of a binary classifier, ROC compares the presence of true positives(TP) and false positives(FP) at every probability threshold. ($i.e. the probability at which an observation is predicted to be a class$)

$$
\mathrm{TPR}=\frac{\text { True Positives }}{\text { True Positives+False Negatives }}
$$

$$
\mathrm{FPR}=\frac{\text { False Positives }}{\text { False Positivest True Negatives }}
$$

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score

logit = LogisticRegression()
logit.fit(features_train, target_train)

# function - predict_proba() returns the probability of the sample for each class in the model, where classes are ordered as they are in self.classes_
target_probabilities = logit.predict_proba(feature_test)[:, 1]

false_positive_rate, true_positive_rate, threshold = roc_curve(target_test, target_probabilities)

plt.plot(false_positive_rate, true_positive_rate)
```

### 11.6 evaluating multiclass classifier predictions

### 11.7 visualizing a classifier's performance

```python
from sklearn.metrics import confusion_matrix
import pandas as pd

matrix = confusion_matrix(target_test, target_predicted)
dataframe = pd.DataFrame(matrix, index=class_names, columns=class_names)
sns.heatmap(dataframe, annot=True, cbar=None, cmap="Blues")
```

### 11.8 evaluating regression models

use mean squared error(MSE)

$$
\mathrm{MSE}=\frac{1}{n} \sum_{i=1}^{n}\left(\widehat{y_{i}}-y_{i}\right)^{2}
$$

$$
R^{2}=1-\frac{\sum_{i=1}^{n}\left(y_{i}-\hat{y}_{i}\right)^{2}}{\sum_{i=1}^{n}\left(y_{i}-\overline{y}\right)^{2}}
$$

### 11.9 evaluating clustering models

$$
s_{i}=\frac{b_{i}-a_{i}}{\max \left(a_{i}, b_{i}\right)}
$$

$$
\begin{array}{l}{\text { where } s_{i} \text { is the silhouette coefficient for observation } i, a_{i} \text { is the mean distance between }} \\ {i \text { and all observations of the same class, and } b_{i} \text { is the mean distance between } i \text { and all }} \\ {\text { observations from the closest cluster of a different class. The value returned by sil }} \\ {\text { houette }_{-} \text {score is the mean silhouette coefficient for all observations. Silhouette coef- }} \\ {\text { ficients range between }-1 \text { and } 1, \text { with } 1 \text { indicating dense, well-separated clusters. }}\end{array}
$$

```python
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

silhouette_score(features, target_predicted)
```

### 11.10 creatiwng a custom evaluation metric

### 11.11 visualizing the effect of training set size

```python
from sklearn.ensemble import RandomForestClassifier
from sklean.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(RandomForestClassifier(), features, target, cv=n, scoring='accuracy', train_sizes=np.linspace(0.01, 1, 50))

# Create means and standard deviations of training set scores
train_maen = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

# Create means and standard deviations of test set scores
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Draw lines
plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")
plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")

# Draw bands
plt.fill_between(train_sizes, train_mean - train_std,
                 train_mean + train_std, color="#DDDDDD")
plt.fill_between(train_sizes, test_mean - test_std,
                 test_mean + test_std, color="#DDDDDD")
```

### 11.12 creating a text report of evalution metrics

```python
from sklearn.metrics import classification_report

print(classification_report(target_test, target_predicted, target_names=class_names))
```

### 11.13 visualizing the effect of hyperparameter values

```python
from sklearn.model_selection import validation_curve

param_range = np.arange(1, 250, 2)
train_scores, test_scores = validation_curve(estimator, features, target, param_name='n_estimators', param_range=param_range, cv=n, scoring='accuracy')

# Calculate mean and standard deviation for training set scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

# Calculate mean and standard deviation for test set scores
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
```

## 12. Model selection

### 12.1 selecting best models using exhaustive search

```python
import numpy as np
from sklearn import Linear_model
from sklearn.model_selection import GridSearchCV

logistic = linear_model.LogisticRegression()

# Create range of candidate penalty hyperparameter values
penalty = ['l1', 'l2']
# Create range of candidate regularization hyperparameter values
C = np.logspace(0, 4, 10)
# Create dictionary hyperparameter candidates
hyperparameters = dict(C=C, penalty=penalty)

gridsearch = GridSearchCV(logistic, hyperparameters, cv=5, verbose=0)
best_model = gridsearch.fit(features, target)
```

### 12.2 selecting best models using randomized search

```python
from scipy.stats import uniform
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV

logistic = LinearRegression()

# Create range of candidate regularization penalty hyperparameter values
penalty = ['l1', 'l2']
# Create distribution of candidate regularization hyperparameter values
C = uniform(loc=0, scale=4)
# Create hyperparameter options
hyperparameters = dict(C=C, penalty=penalty)

randomizedsearch = RandomizedSearchCV(logistic, hyperparameters, random_state=1, n_iter=100, cv=5, verbose=0)
best_model = randomizedsearch.fit(features, target)
```

### 12.3 selecting best models from multiple learning algorithms

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

pipe = Pipeline([('classifier', RandomForestClassifier())])

search_space = [{'classifier': [LogisiticRegression()], \
                'classifier__penalty': ['l1', 'l2'], \
                'classifier__C': np.logspace(0, 4, 10)}, \
                \
                {'classifier': [RandomForestClassifier()], \
                'classifier__n_estimators': [10, 100, 1000], \
                'classifier__max_featrures': [1, 2, 3]}]

gridsearch = GridSearchCV(pipe, search_space, cv=5, verbose=0)

best_model = gridsearch.fit(features, target)
```

### 12.4 selecting best models when preprocessing

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Create a preprocessing object that includes StandardScaler features and PCA
preprocess = FeatureUnion([("std", StandardScaler()), ("pca", PCA())])

# create a pipeline
pipe = Pipeline([("preprocess", preprocess),
                 ("classifier", LogisticRegression())])

# Create space of candidate values
search_space = [{"preprocess__pca__n_components": [1, 2, 3],
                 "classifier__penalty": ["l1", "l2"],
                 "classifier__C": np.logspace(0, 4, 10)}]

clf = GridSearchCV(pipe, search_space, cv=5, verbose=0)

best_model = clf.fit(features, target)
```

### 12.5 speeding up model selection with parallelization

```python
# Load libraries
from sklearn import linear_model, datasets
# Load data
iris = datasets.load_iris()
features = iris.data
target = iris.target
# Create cross-validated logistic regression
logit = linear_model.LogisticRegressionCV(Cs=100) # Train model, C(s) the penlty strengths of regularization
logit.fit(features, target)
```

### 12.7 evaluating performance after model selection

```python
from sklearn.linear_model import LogisticRegression()
from sklearn.model_selection import GridSearchCV, cross_val_score

logistic = LogisticRegression()

# Create range of 20 candidate values for C
C = np.logspace(0, 4, 20)
# Create hyperparameter options
hyperparameters = dict(C=C)

gridsearch = GridSearchCV(logistic, hyperparameters, cv=5, verbose=0)

cross_val_score(gridsearch, features, target).mean()

best_model = gridsearch.fit(features, target)

scores = cross_val_score(gridsearch, features, target)
```

## 13. Linear regression

### 13.1 fitting a line

```python
from sklearn.linear_model import LinearRegression

model.fit(features, target)
model.intercept_
model.coef_
```

### 13.2 handling interactive effects

```python
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

interaction = PolynomialFeatures(degree=3, include_bias=False, interaction_only=True)
features_interaction = interaction.fit_transform(features)

regression = LinearRegression()
regression.fit(features_interaction, target)
```

### 13.2 fitting a nonlinear relationship

```python
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

polynomial = PolynomialFeatures(degree=3, include_bias=False)
features_polynomail = polynomial.fit_transform(features)

regression = LinearRegression()
regression.fit(features_polynomial, target)
```

### 13.4 reading variance with regularization

$$
R S S=\sum_{i=1}^{n}\left(y_{i}-\hat{y}_{i}\right)^{2}
$$

$$
\mathrm{RSS}+\alpha \sum_{j=1}^{p} \hat{\beta}_{j}^{2}
$$

$$
\frac{1}{2 n} \mathrm{RSS}+\alpha \sum_{j=1}^{p}\left|\hat{\beta}_{j}\right|
$$

```python
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sclaer = StandardScaler()
features_standardized = scaler.fit_transform(features)

regression = Ridge(alpha=0.5)
regression.fit(features_standardized, target)
```

```python
from sklearn.linear_model import RidgeCV

redge_cv = RidgeCV(alpha=[0.01, 0.1, 1, 10])
model_cv = redge_cv.fit(features_standardized, target)
model_cv.corf_
model_cv.alpha_
```

### 13.5 reducing features with Lasso regression

```python
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

regression = Lasso(alpha=0.5)
regression.fit(features_standardized, target)

regression.coef_
```

## 14. Trees and Forests

### 14.1 training a decision tree classifier

```python
from sklearn.tree import DecisionTreeClassifier
decisiontree = DecisionTreeClassifier(random_state=0)
decisiontree.fit(features, target)
```

### 14.2 training a decision tree regressor

```python
from sklearn.tree import DecisionTreeRegressor

decisiontree = DecisionTreeRegressor(random_state=0)
decisiontree.fit(features, target)
```

### 14.3 visualizing a decision tree model

```python
import pydotplus
from sklearn.tree import DecisionTreeClassifier
from IPython.display import Image
from sklearn import tree

decisiontree = DecisionTreeClassifier(random_state=0)
decisiontree.fit(features, target)

# create DOT data
dot_data = tree.export_graphviz(decisiontree, out_file=None, feature_names=iris.feature_names, class_names=iris.target_names)

# draw graph
graph = pydotplus.graph_from_dot_data(dot_data)

# show graph
Image(graph.create_png())

# create PDF or PNG
graph.write_pdf()
graph.wirte_png()
```

### 14.4 training a random forest classifier

```python
from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier(random_state=0)
randomforest.fit(features, target)
```

### 14.5 training a random forest regressor

```python
from sklearn.ensemble import RandomForestRegressor

randonforest = RandomForestRegressor(random_state=0)
randomforest.fit(features, target)
```

### 14.6 identifying important features in random forests

```python
imoprtances = randomforests.feature_importances_

indices = np.argsort(importances)[::-1]
```

### 14.7 selecting important features in random forests

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

randomforest = RandomForestClassifier(random_state=0)
selector = SelectFromModel(randomforest, threshold=0.3)

# select the important features
features_important = selector.fit_transform(features, target)

model = randomforest.fit(features_important, target)
```

### 14.8 handling imbalanced classes

```python
randomforest = RandomForestClassifier(random_state=0, class_weight='balanced')
randomforest.fit(features. target)
```

### 14.9 controlling tree size

```python
decisiontree = DecisionTreeClassifier(random_state=0, \
                                      max_depth=None, \
                                      min_sample_split=2, \
                                      min_sample_leaf=1, \
                                      min_weight_fraction_leaf=0, \
                                      max_leaf_nodes=None, \
                                      min_impurity_decrease=0)

decisiontree.fit(features, target)
```

### 14.10 improving performance through boosting

```python
from sklearn.ensemble import AdaBoostClassifier

adaboost = AdaBoostClassifier(random_state=0)
adaboost.fit(features, target)
```

### 14.11 evaluating random forests with Oout-of-Bag errors

For every tree there is a separate subset of observations not being used to train that tree. These are called out-of-bag(OOB) observations. We can use OOB observations as a test set to evaluate the performance of our random forest.

```python
randomforest.oob_score_
```

## 15. K-Nearest Neighbors

### 15.1 finding an observation's nearest neighbors

```python
from sklearn.neighbors import NearestNeighbors

nearest_neighbors = NearestNeighbors(n_neighbors=n).fit(features_standardized)

new_observations = [1, 1, 1, 1]
distance, indices = nearest_neighbors.kneighbors([new_observation])
```

$$
d_{\text {eudidean}}=\sqrt{\sum_{i=1}^{n}\left(x_{i}-y_{i}\right)^{2}}
$$

$$
d_{\text {manhattan}}=\sum_{i=1}^{n}\left|x_{i}-y_{i}\right|
$$

$$
d_{\text {minkowski}}=\left(\sum_{i=1}^{n}\left|x_{i}-y_{i}\right|^{p}\right)^{1 / p}
$$

In addition, we can use `kneighbors_graph` to create a matrix indicating each observation's nearest neighbors.

### 15.2 creating K-Nearest neighbor classifier

```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=n)
knn.fit(x_std, y)

knn.predict(new_observations)

# or predict probability
knn.pred_proba(new_observations)
```

### 15.3 identifying the best neighborhood size

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV

knn = KNeighborsClassifier(n_neighbors=n)

pipe = Pipeline([('standardizer', standardizer), ('knn', knn)])

search_space = [{'knn__n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}]

clf = GridSearchCV(pipe, search_space, cv=n, verbose=0).fit(features_standardized, target)
```

### 15.3 creating a redius-based nearest neighbor classifier

```python
from sklearn.neighbors import RadiusNeighborsClassifier

rnn = RadiusNeighborsClassifier(radius=0.5).fit(features_standardized, target)

rnn.predict(new_observations)
```

## 16. Logistic Regression

### 16.1 training a binary classifier

```python
# scaling features
# fit model
logistic_regression = LogisticRegression(random_state=0)
logistic_regression.fit(features_standardize, target)
```

### 16.2 training a multiclass classifier

using one-vs-rest or multinomial methods.

For ‘multinomial’ the loss minimised is the multinomial loss fit across the entire probability distribution, even when the data is binary.

```python
# scaling features
# fit model
logistic_regression = LogisticRegression(random_state=0, multi_class='ovr)
logistic_regression.fit(features_standardized, target)
```

### 16.3 reducing variance through regularization

```python
from sklearn.linear_model import LogisticRegressionCV

# scaling features
# fit model
logistic_regression = LogisticRegressionCV(penalty='l1/l2', Cs=10, random_state=0)
logistic_regression.fit(features_standardized, target)
```

### 16.4 training a classifier on very large data

Training a logistic regression in scikit-learn with LogisticRegression using the stochastic average gradient (SAG) solver.

```python
# scaling features
# fit model
logistic_regression = LogisticRegression(solver='sag')
logistic_regression.fit(features_standardized, target)
```

### 16.5 handling imbalanced classes

The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as: $w_{j}=\frac{n}{k n_{j}}$. where wj is the weight to class j, n is the number of observations, nj is the number of observations in class j, and k is the total number of classes.
```python
# scaling features
# fit model use class_weight='balanced'
logistic_regression = LogisticRegression(random_state=0, class_weight='balanced')
```

## 17 Support Vector Machines

### 17.1 training a linear classifier

```python
from sklearn.svm import LinearSVC

svc = LinearSVC(C=1.0).fit(features_std, target)

## visualize the boundary
w = svc.coef_[0]
a = - w[0]
xx = np.linspace(-2.5, 2.5)
yy = ( a * xx - (svc.intercept_[0]) ) / w[1]

plt.plot(xx, yy)
```

### 17.2 handling linearly inseparable classes using kernels

```python
# Create a support vector machine with a radial basis function kernel
svc = SVC(kernel="rbf", random_state=0, gamma=1, C=1).fit(features, target)
```

```python
# Plot observations and decision boundary hyperplane
from matplotlib.colors import ListedColormap import matplotlib.pyplot as plt
def plot_decision_regions(X, y, classifier):
cmap = ListedColormap(("red", "blue"))
xx1, xx2 = np.meshgrid(np.arange(-3, 3, 0.02), np.arange(-3, 3, 0.02))
Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
Z = Z.reshape(xx1.shape)
plt.contourf(xx1, xx2, Z, alpha=0.1, cmap=cmap)
for idx, cl in enumerate(np.unique(y)):
  plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
              alpha=0.8, c=cmap(idx),
              marker="+", label=cl)
```

### 17.3 creating predicted probabilities

```python
# Create support vector classifier object
svc = SVC(kernel="linear", probability=True, random_state=0).fit(features_standardized, target)

model.predict_proba(new_observation)
```

### 17.4 identifying support vectors

```python
# View support vectors
model.support_vectors_

## we can view the indices of the support vectors using:
model.support_

## we can use this to find the number of support vectors belonging to each class:
model.n_support_
```

### 17.5 handling imbalanced classes

```python
# Create support vector classifier
svc = SVC(kernel="linear", class_weight="balanced", C=1.0, random_state=0).fit(features_standardized, target)
```

## 18 Naive Bayes

### 18.1 training a classifier for continuous features

The most common type of naive Bayes classifier is the Gaussian naive Bayes. In Gaussian naive Bayes, we assume that the likelihood of the feature values, $x$, given an observation is of class $y$, follows a normal distribution.:

$$
p\left(x_{j} | y\right)=\frac{1}{\sqrt{2 \pi \sigma_{y}^{2}}} e^{-\frac{\left(x_{j}-\mu_{y}\right)^{2}}{2 \sigma_{y}^{2}}}
$$

```python
from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()
clf.fit(features, target)
```

### 18.2 training a classifier for discrete and count features

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVecterizer

# create bag of words
count = CountVecterizer()
bow = count.fit_transform(text_data)

# create feature matrix
features = bow.toarray()

# create multinomial naive Bayes object with prior probabilities of each class
clf = MultinomialNB(class_prior=[0.25, 0.5])
clf.fit(features, target)
```

### 18.3 training a Naice Bayes classifier for binary features

```python
from sklearn.naive_bayes import BernoulliNB

clf = BernoulliNB(class_prior=[0.25, 0.5])
clf.fit(features, target)
```

### 18.4 calibrating predicted probabilities

In calibratedClassifierCV the training sets are used to train the model and the test set is used to calibrate the predicted probabilities. The returned predicted probabilities are the average of the k-folds.

```python
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassiferCV

# create gaussian Naive Bayes object
clf = GaussianNB()

# create calibrated cross-validation with sigmoid calibration
clf_sigmoid = CalibratedClassifierCV(clf, cv=2, method='sigmoid')

# calibrate probabilities
clf_sigmoid.fit(features, target)
```

```python
# Train a Gaussian naive Bayes then predict class probabilities
classifer.fit(features, target).predict_proba(new_observation)
## output: array([[  2.58229098e-04,   9.99741447e-01,   3.23523643e-07]])

# View calibrated probabilities
classifer_sigmoid.predict_proba(new_observation)
## output: array([[ 0.31859969,  0.63663466,  0.04476565]])
```

## 19 Clustering

### 19.1 clustering using K-Means

```python
from sklearn.cluster import KMeans

cluser = KMeans(n_clusters=n, random_state=0)
model = cluster.fit(features_std)
```

### 19.2 speeding up K-Means clustering

```python
from sklearn.cluster MiniBatchKMeans

cluster = MiniBatchKMeans(n_clusters=n, random_state=0, batch_size=100)
model = cluster.fit(features_std)
```

### 19.3 clustering using Meanshift

```python
from sklearn.cluster import MeanShift

# Create meanshift object
cluster = MeanShift(n_jobs=-1)

# Train model
model = cluster.fit(features_std)
```

### 19.4 clustering using DBSCAN

```python
from sklearn.cluster import DBSCAN

# Create meanshift object
cluster = DBSCAN(n_jobs=-1)

# Train model
model = cluster.fit(features_std)
```

### 19.5 clustering using hierarchical merging

```python
from sklearn.cluster import AgglomeractiveClustering

# Create meanshift object
cluster = AgglomerativeClustering(n_clusters=3)

# Train model
model = cluster.fit(features_std)
```

