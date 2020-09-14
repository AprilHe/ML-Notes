## Contents

- [1. Random Forest](#1-Random-Forest)
   - [1.1 Bagging](#11-Bagging)
   - [1.2 Random Forest](#12-Random-Forest)
- [2. Out of Bag Error](#2-Out-of-Bag-Error)
- [3. Implementation](#3-Implementation)
- [4. Reference](#3-Reference)



## 1. Random Forest
### 1.1 Bagging
As we described in Decision Tree, it is very easy to overfit, so to avoid it and high lower variance, we introduce bagging. Bagging is bootstrap aggregating, it create several subsets of data from training sample chosen randomly with replacement. Each collection of subset data is used to train their models. As a result, we end up with an ensemble of different models. Average of all the predictions from different results so that improve the accuracy and avoid noise.


### 1.2 Random Forest
Random Forest is an extension over bagging algorithm with Decision tree (CART) model. It takes one extra step where in addition to taking the random subset of data, it also takes the random selection of features rather than using all features to grow trees.

Suppose there are N observations and M features in training data set, the random forest is developed:
- A sample with N observation from training data set is taken randomly with replacement.
- A subset of M features (m) are selected randomly and whichever feature gives the best split is used to split the node iteratively.
- The tree is grown to the largest.
- Above steps are repeated and prediction is given based on the aggregation of predictions from n number of trees.

**Advantages**
- Handles higher dimensionality data very well, no feature selection is required
- Handles missing values and maintains accuracy for missing data
   - fills in the median value for continuous variables, or the most common non-missing value by class.
   - fills in missing values, then runs RF, then for missing continuous values, RF computes the proximity-weighted average of the missing values. Repeat this process several times and the model is trained a final time using the RF-imputed data set.
- Balance errors in data sets where classes are imbalanced.

**Disadvantages**
- For regression, the final prediction is based on the mean predictions from subset trees, it won’t give precise values for the regression model.
- Overfitting when there are rare outcomes or rare predictors

## 2. Out of Bag Error
Each tree is constructed using a different bootstrap sample from the original data. The cases are left out of the bootstrap sample and not used in the construction of the each tree are consider to be the out of bag sample. With the model constructed by sample in bag, apply it to out of bag sample and calculate the error. This has proven to be unbiased in many tests.


## 3. Implementation
[Random Forest Implementation with Scikit Learn.](https://github.com/AprilHe/ML-Notes/blob/master/MachineLearning/5.%20Ensemble%20/5.2%20Random%20Forest/Random%20Forest.ipynb)

Commonly used hyper parameters:  
- n_estimators: number of trees in the foreset
- max_features: max number of features considered for splitting a node
- max_depth: max number of levels in each decision tree
- min_samples_split: min number of data points placed in a node before the node is split
- min_samples_leaf: min number of data points allowed in a leaf node
- bootstrap: method for sampling data points (with or without replacement)



## 4. Reference
1. [Random Forests Leo Breiman and Adele Cutler](https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm#missing1)
2. [Scikit Learn RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
