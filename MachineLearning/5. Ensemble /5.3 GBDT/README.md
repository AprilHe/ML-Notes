## Contents

- [1. Gradient Boosting Decision Tree](#1-Gradient-Boosting-Decision-Tree)
   - [1.1 Boosting](#11-Boosting)
   - [1.2 Gradient Boosting Decision Tree](#212-Gradient-Boosting-Decision-Tree)
- [2. Random Forest v.s. GBDT](#3-Random-Forest-v.s.-GBDT)
- [3. Implementation](#3-Implementation)
- [4. Reference](#4-Reference)



## 1. Gradient Boosting Decision Tree
### 1.1 Boosting
Boosting is another ensemble technique to create a collection of predictors. Models are learned sequentially with **increasing weights on misclassified predictors** at early learners. In other words, we fit consecutive trees and at every step, the goal is to reduce error from the prior steps. It converts weak learner to strong learners sequentially.



### 1.2 Gradient Boosting Decision Tree
Gradient Boosting is an extension over boosting method with **Gradient Descent** applied to optimize any differentiable **loss function** in each step of learning a decision tree.  There are typically three parameters - number of trees, depth of trees and learning rate.


**Advantages**
- Supports different loss function.
- Works well with interactions.

**Limitations**
- performance is not good on high sparsity data.
- Requires careful tuning of different hyper-parameters
- Prone to overfitting

## 2. Random Forest v.s. GBDT
Both Random Forest and GBDT are ensemble learning methods and predict (regression or classification) by combining the outputs from individual trees. They differ in the way the trees are built and the way the results are combined.
- RF trains each tree independently while GBDT trains one tree at a time with the help of previous tree to correct error in previous stage
- RF take the vote of each tree while GBDT add the results from each tree
- RF is not sensitive to outliers while opposite in GBDT
- RF aims for reducing variances while GBDT aims for reducing bias



## 3. Implementation
[GBM Implementation with Scikit Learn.](https://github.com/AprilHe/ML-Notes/blob/master/MachineLearning/5.%20Ensemble%20/5.3%20GBDT/GBM.ipynb)

Commonly used hyper parameters:  
- learning_rate： learning rate shrinks the contribution of each tree.
- N_estimators: number of trees in the forest. Usually the higher the number of trees the better to learn the data.
- max_depth: deep of the tree. The deeper the tree, the more splits it has and it captures more information about how the data.
- min_samples_split:  minimum number of samples required to split an internal node.
- min_samples_leaf: minimum number of samples required to be at a leaf node.
- max_features: represents the number of features to consider when looking for the best split.



## 4. Reference
1. [Friedman, J. H.  "Greedy Function Approximation: A Gradient Boosting Machine"](https://statweb.stanford.edu/~jhf/ftp/trebst.pdf)
