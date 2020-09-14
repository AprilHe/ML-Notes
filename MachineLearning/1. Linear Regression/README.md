------
## Contents

- [1. Linear Regression](#1-Linear-Regression)
- [2. Assumptions ](#2-Assumptions)
- [3. Cost Function](#3-Cost-Function)
   - [3.1 Ordinary Least Squares (OLS)](##31-Ordinary-Least-Squares-(OLS))
   - [3.2 Gradient Descent](##32-Gradient-Descent)
- [4. Implementation](#4-Implementation)




## 1. Linear Regression
Linear regression model is the most commonly used model in regression. It attempts to model the relationship between a variable we are interested at (named as dependent variable or response) and one or more explanatory variables (named as independent variables or predictor). This relationship is modelled by estimating the parameters of each independent variables in a linear predictor function.

In particular, we model how the mean, or expectation, of the outcome Y varies as a function of the predictors:

<img src="https://latex.codecogs.com/gif.latex?E(Y|X_{1},..,X_{p}) = \theta_{0}+\theta_{1}X_{1}+...+\theta_{p}X_{p}" /></a>

Equivalently, the linear model can be expressed by:

<img src="https://latex.codecogs.com/gif.latex?Y= {{h}_{\theta }}\left( X \right) = \theta_{0}+\theta_{1}X_{1}+...+\theta_{p}X_{p} + \epsilon" /></a>

where <img src="https://latex.codecogs.com/gif.latex?\epsilon" /> is the error or residual


## 2. Assumptions
There are some assumptions behind the linear model
-  Linearity: The relationship between predictors and the mean of response variables is linear.
-  Homoscedasticity (Constant variance): The variance of errors is the same for any value of X.
-  Independence of errors: Errors of the response variables are uncorrelated with each other.
-  Lack of perfect multicollinearity in the predictors: Perfectly correlated predictor variables or fewer data points than regression coefficients will have no unique solution of parameters (Linear Algebra)
-  Weak exogeneity: The predictors are assumed to be error-free, not contaminated with measurement errors. With all significant predictors, expected value on error is 0.
-  Normality: For any fixed value of X, Y is normally distributed. (Optional)


## 3. Cost Function
To fit a best straight line to our data, we want to minimised the cost function. Cost function quantifies the error between predicted values and expected values and presents it in the form of a single real number.  

We will use MSE (L2) here, it measures the average squared difference between an observation’s actual and predicted values.

<img src="https://latex.codecogs.com/gif.latex?J\left( \theta  \right)=\frac{1}{2m}\sum\limits_{i=1}^{m}{{{\left( {{h}_{\theta }}\left( {{x}^{(i)}} \right)-{{y}^{(i)}} \right)}^{2}}}" />


Numerical solutions for finding a solution of a minimised cost function
- Normal equations method with OLS
- Gradient descent (better to large data sets)


### 3.1 Ordinary Least Squares (OLS)
OLS is a non-iterative method that fits a model such that the sum-of-squares of differences of observed and predicted values is minimised.

To minimise the cost function, we solve the following equation:

<img src="https://latex.codecogs.com/gif.latex?\frac{\partial }{\partial {{\theta }_{j}}}J\left( {{\theta }_{j}} \right)=0" /></a>


The closed form of estimator is <img src="https://latex.codecogs.com/gif.latex?\theta ={{\left( {{X}^{T}}X \right)}^{-1}}{{X}^{T}}y" />


Advantage
- No need to choose hyper parameter (learning rate)
- No iteration, no need to check convergency
- Can be much faster for small sample (O(n3))

Disadvantage
- Much more complicated for large sample

### 3.2 Gradient Descent
Gradient descent finds the linear model parameters iteratively. To run gradient descent on this error function, we first need to compute its gradient. The gradient will act like a compass and always point us downhill.

- Update <img src="https://latex.codecogs.com/gif.latex?\theta_{j}" />
 by setting it to (<img src="https://latex.codecogs.com/gif.latex?\theta_{j} - \alpha" />) times the partial derivative of the cost function with respect to <img src="https://latex.codecogs.com/gif.latex?\theta_{j}" />

- <img src="https://latex.codecogs.com/gif.latex?\alpha" /> is the learning rate, which controls how big a step to iterate

<img src="https://latex.codecogs.com/gif.latex?{{\theta }_{j}}:={{\theta }_{j}}-\alpha \frac{\partial }{\partial {{\theta }_{j}}}J\left( \theta  \right)" /></a>


Note: If there are multiple features, normalise each feature so that they have a similar scale and gradient descent will converge more quickly


## 4. Implementation
[Linear Model Implementation]()
