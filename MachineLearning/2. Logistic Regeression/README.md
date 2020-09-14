------
## Contents

- [1. Logistic Regression](#1-Logistic-Regression)
- [2. Cost Function](#2-Cost-Function)
   - [2.1 Gradient Descent](##21-Gradient-Descent)
   - [2.2 Other Optimization Method](##22-Other-Optimization-Method)
- [3. Regularization](#3-Regularization)
- [4. Multiclass classification problems](#4-Multiclass-Classification-Problems)
- [5. Implementation](#5-Implementation)


## 1. Logistic Regression
A statistical model to fit the probability of a certain class or event existing to determine what class a new input should fall into, such as binary results - pass or fail in the exam, with more complex extension usage.


**Classification Probelms**
- Credit Risk
- Click through rate
- Medical treatment
...


**Basic Information**
- Actual event (Dependent variables Y) is a discrete value, with two possible values, 0/1 (multiple class in extended situation)
- The logit function of predicted probability of actual event $\hat{Y}$ is a linear combination of independent variables (X)
- Decide the cut-off (decision boundary) on $\hat{Y}$ to predict/fit the actual class $Y$ with the help of some metrics (Confustion metrics, AUC-ROC plot, F1-score, accuracy, GINI)

**Model**

<img src="https://latex.codecogs.com/gif.latex?logit(P(Y=k|X))=ln(\frac{P(Y=k|X)}{1-P(Y=k|X)})= \theta_{0}+\theta_{1}X_{1}+...+\theta_{p}X_{p}" /></a>

<img src="https://latex.codecogs.com/gif.latex?\hat{Y} = P(Y=k | X) = \frac{1}{1+{{e}^{-{{\theta }^{T}}X}}}" /></a>


where <img src="https://latex.codecogs.com/gif.latex?\theta" /> is the parameter (or weight) of each independent variables

<img src="https://latex.codecogs.com/gif.latex?g(z) = \frac{1}{1+{{e}^{-z}}}" /> is *sigmond function (logistic function)* with S shape

<img src="../images/LR-sigmod.png" width="300">






## 2. Cost Function
<img src="https://latex.codecogs.com/gif.latex?J\left( \theta  \right)=\frac{1}{m}\sum\limits_{i=1}^{m}{[-{{y}^{(i)}}\log \left( {{h}_{\theta }}\left( {{x}^{(i)}} \right) \right)-\left( 1-{{y}^{(i)}} \right)\log \left( 1-{{h}_{\theta }}\left( {{x}^{(i)}} \right) \right)]}" /></a>


- If <img src="https://latex.codecogs.com/gif.latex?{y}^{(i)} = 1" />, cost function is <img src="https://latex.codecogs.com/gif.latex?-\log( {{h}_{\theta }}( {{x}^{(i)}})" />.
- If <img src="https://latex.codecogs.com/gif.latex?{y}^{(i)} = 0" />, cost function is <img src="https://latex.codecogs.com/gif.latex?-\log( 1 -  {{h}_{\theta }}( {{x}^{(i)}})" />


<img src="../images/LR-cost.png" width="600"></a>

- If hypothesis predicts exactly correct then that cost corresponds to 0
- otherwise, cost goes to infinity and penalize the learning algorithm with a massive cost

### 2.1 Gradient Descent

Partial derivative of cost function:

<img src="https://latex.codecogs.com/gif.latex?\frac{1}{m} X^T( h_{\theta}(x) - y )" /></a>

<img src="https://latex.codecogs.com/gif.latex?\frac{\partial J\left( \theta  \right)}{\partial {{\theta }_{j}}}=\frac{1}{m}\sum\limits_{i=1}^{m}{({{h}_{\theta }}\left( {{x}^{(i)}} \right)-{{y}^{(i)}})x_{_{j}}^{(i)}}" /></a>

Repeat until convergence:
<img src="https://latex.codecogs.com/gif.latex?{{\theta }_{j}}:={{\theta }_{j}}-a\frac{1}{m}\sum\limits_{i=1}^{m}{[{{h}_{\theta }}\left( {{x}^{(i)}} \right)-{{y}^{(i)}}]x_{j}^{(i)}}, j = 0, 1, 2, 3, ... " /></a>

Add <img src="https://latex.codecogs.com/gif.latex?\frac{\lambda }{m}{{\theta }_{j}}" /> if we have regularization on parameters for <img src="https://latex.codecogs.com/gif.latex?{{\theta }_{j}}, j = 1,2,3, ..." />

### 2.2 Other Optimization Method
- Conjugate gradient
- BFGS (Broyden-Fletcher-Goldfarb-Shanno)
- L-BFGS (Limited memory - BFGS)

## 3. Regularization

To solve the problem of overfitting (high variance), which is good performance in training sample but bad results on testing sample, we can reduce number of features or apply Regularization.

With regularization:
- Keep all features, but reduce magnitude of parameters θ (magnitude might be 0 depends on regularization method)
- Works well when we have a lot of features, each of which contributes a bit to predicting y

<img src="https://latex.codecogs.com/gif.latex?J\left( \theta  \right)=\frac{1}{m}\sum\limits_{i=1}^{m}{[-{{y}^{(i)}}\log \left( {{h}_{\theta }}\left( {{x}^{(i)}} \right) \right)-\left( 1-{{y}^{(i)}} \right)\log \left( 1-{{h}_{\theta }}\left( {{x}^{(i)}} \right) \right)]}+\frac{\lambda }{2m}\sum\limits_{j=1}^{n}{\theta _{j}^{2}}" /></a>



Gradient descent optimization with regularization ,

<img src="https://latex.codecogs.com/gif.latex?J{{\theta }_{0}}:={{\theta }_{0}}-a\frac{1}{m}\sum\limits_{i=1}^{m}{[{{h}_{\theta }}\left( {{x}^{(i)}} \right)-{{y}^{(i)}}]x_{_{0}}^{(i)}} " /></a>

<img src="https://latex.codecogs.com/gif.latex?{{\theta }_{j}}:={{\theta }_{j}}-a\frac{1}{m}\sum\limits_{i=1}^{m}{[{{h}_{\theta }}\left( {{x}^{(i)}} \right)-{{y}^{(i)}}]x_{j}^{(i)}}+\frac{\lambda }{m}{{\theta }_{j}}" /></a>

## 4. Multiclass classification problems

One vs. all classification - Suppose we have three classes instead of binary value, we split the training set into three separate binary classification problems:
- Class 1 vs class 2 and 3 - <img src="https://latex.codecogs.com/gif.latex?P(y=1 | x_{1}; \theta)" />
- Class 2 vs class 1 and 3 - <img src="https://latex.codecogs.com/gif.latex?P(y=1 | x_{2}; \theta)" />
- Class 3 vs class 1 and 2 - <img src="https://latex.codecogs.com/gif.latex?P(y=1 | x_{3}; \theta)" />



## 5. Implementation
[Logistic Regression implementation]()
