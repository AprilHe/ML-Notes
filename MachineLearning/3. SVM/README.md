------
## Contents

- [1. What is SVM](#1-What-is-SVM)
- [2. SVM with Hard Margin ](#2-SVM-with-Hard-Margin)
- [3. SVM with Soft Margin ](#3-SVM-with-Soft-Margin)
- [4. Kernel Trick for Non Linear SVM](#4-Kernel-Trick-for-Non-Linear-SVM)
- [5. Summary](#5-Summary)
- [6. Implementation](#6-Implementation)
- [7. Reference](#-Reference)

## 1. What is SVM
Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection. For classification problem, SVM algorithm fit a hyperplane that distinctly classifies the data points with maximum margin, i.e the maximum distance between data points of both classes.


**Terminologies**
Take an example of 2D dimensions:
<p align="center">
<img src="../images/SVM_margin_2.png" width="300"></a>
</p>

- Hyperplane: a function <img src="https://latex.codecogs.com/gif.latex?w^Tx+b=0" /> used to differentiate between features (Red line)
- Support vector points: The points on two dashed lines, which are closest to the hyperplane <img src="https://latex.codecogs.com/gif.latex?w^Tx+b=1" /> or <img src="https://latex.codecogs.com/gif.latex?w^Tx+b=-1" />
- Margins: Two times of the distance of the Support vector points from the hyperplane
- Decision function: <img src="https://latex.codecogs.com/gif.latex?f(x)=\operatorname{sign}\left(w^{*} \cdot x+b^{*}\right)" />

**SVMs model**
- Linear SVM (Hard margin): The data is linearly separable
- Linear SVM (Soft margin): Extension of hard margin SVM, the data is almost linearly separable  
- Non Linear SVM: Apply kernel function to map data points to different feature space so that data points are linearly separable

## 2. SVM with Hard Margin

In this section, I will give detail process of fit hyperplane for linearly separable data.

**Optimize function**
- The distance between two parallel lines <img src="https://latex.codecogs.com/gif.latex?w^Tx+b=1" /> and <img src="https://latex.codecogs.com/gif.latex?w^Tx+b=-1" /> is 2
- The distance from points to the hyperplane: <img src="https://latex.codecogs.com/gif.latex?r=\frac{|w^Tx+b|}{||w||}" /> where <img src="https://latex.codecogs.com/gif.latex?||w||=\sqrt[2]{\sum^m_{i=1}w_i^2}" />
- Support vectors with the minimal distance to hyperplane satisfy the equation <img src="https://latex.codecogs.com/gif.latex?sign(w^Tx+b) = 1   or \quad y_{i}\left(w \cdot x_{i}+b\right)=1" />


① To maximal margin,
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?margin = \rho = \frac{2}{||w||} " /></a>
</p>
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\max_{w, b}(\rho)  \Leftrightarrow \max_{w, b}(\rho^2)  \Leftrightarrow \min_{w, b} \frac{1}{2}\|w\|^{2}" /></a>
</p>
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?s.t. \quad y_{i}\left(w \cdot x_{i}+b\right)-1 \geqslant 0, \quad i=1,2, \cdots, N" /></a>
</p>

This is a quadratic programming - minimising a quadratic function subject to linear constraints.

② **Lagrange multipliers**
To solve above equation, construct a Lagrange function, <img src="https://latex.codecogs.com/gif.latex?\alpha_i" />  is Lagrange multiplier

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?L(w, b, \alpha) = \frac{1}{2}||w||^2+\sum^m_{i=1}\alpha_i(-y_i(w^Tx_i+b)+1)" />
</p>


- **Lagrange duality principle**：
   - Optimization problems may be viewed as primal (in this case minimising over w and b) or dual (in this case, maximising over a).
   - For a convex optimisation problem, the primal and dual have the same optimum solution.

- So the optimisation can be converted as:
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\min \frac{1}{2}||w||^2=\min \max\ L(w, b, \alpha)\geq \max \min\ L(w, b, \alpha)" /></a>
</p>

- Taking partial derivatives with respect to w and b we obtain:
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\frac{\partial }{\partial w}L(w, b, \alpha)=w-\sum\alpha_iy_ix_i=0,\ w=\sum\alpha_iy_ix_i" />
</p>
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\frac{\partial }{\partial b}L(w, b, \alpha)=\sum\alpha_iy_i=0" />
</p>


- Substitute into <img src="https://latex.codecogs.com/gif.latex?L(w, b, \alpha)" />:

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\begin{align*}
\min\  L(w, b, \alpha) & =\frac{1}{2}||w||^2+\sum^m_{i=1}\alpha_i(-y_i(w^Tx_i+b)+1) \\
& =\frac{1}{2}w^Tw-\sum^m_{i=1}\alpha_iy_iw^Tx_i-b\sum^m_{i=1}\alpha_iy_i+\sum^m_{i=1}\alpha_i \\
& =\frac{1}{2}w^T\sum\alpha_iy_ix_i-\sum^m_{i=1}\alpha_iy_iw^Tx_i+\sum^m_{i=1}\alpha_i \\
& =\sum^m_{i=1}\alpha_i-\frac{1}{2}\sum^m_{i=1}\alpha_iy_iw^Tx_i \\
& =\sum^m_{i=1}\alpha_i-\frac{1}{2}\sum^m_{i,j=1}\alpha_i\alpha_jy_iy_j(x_ix_j) \\
\end{align*} " /></a>
</p>

- Transfer the problem from max to min：
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\max\ \sum^m_{i=1}\alpha_i-\frac{1}{2}\sum^m_{i,j=1}\alpha_i\alpha_jy_iy_j(x_ix_j)=\min \frac{1}{2}\sum^m_{i,j=1}\alpha_i\alpha_jy_iy_j(x_ix_j)-\sum^m_{i=1}\alpha_i" />
</p>

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?s.t.\ \sum^m_{i=1}\alpha_iy_i=0,    \alpha_i \geq 0,i=1,2,...,m" />
</p>


We now have an optimisation problem over <img src="https://latex.codecogs.com/gif.latex?\alpha"/>(Apply Sequential Minimal Optimization, SMO to solve it). After optimising our Lagrange multipliers, <img src="https://latex.codecogs.com/gif.latex?\alpha"/>, we can have optimised value of <img src="https://latex.codecogs.com/gif.latex?w"/> and <img src="https://latex.codecogs.com/gif.latex?b"/> so that to classify new data points

## 3. SVM with Soft Margin

Soft Margin allow SVM to make a certain number of mistakes and keep margin as wide as possible so that other points can still be classified correctly.

In the plot below, even the red line can separable the points perfectly, but the green decision boundary has a wider margin that would allow it to generalize well on unseen data. In that sense, soft margin formulation would also help in avoiding the overfitting problem.

<p align="center">
<img src="../images/svm-softmargin.png" width="300">
</p>

Data points that are far away on the wrong side of the decision boundary should have more penalty as compared to the ones that are closer, so the penalty on non-perfectly classified points is **Hinge Loss**:

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\max(0, 1-y_{i}(w \cdot x_{i}+b) )"/>
</p>


In mathematics, the objective function becomes:
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\min _{w, b, \xi} \frac{1}{2}\|w\|^{2}+C \sum_{i=1}^{N} \max(0, 1-y_{i}(w \cdot x_{i}+b) ) "/>
</p>


where C is a hyperparameter that decides the trade-off between maximizing the margin and minimizing the mistakes.
    - When C is small, classification mistakes are given less importance and focus is more on maximizing the margin
    - when C is large, the focus is more on avoiding misclassification at the expense of keeping the margin small

We introduce soft margin as <img src="https://latex.codecogs.com/gif.latex?\xi_{\mathrm{i}}"/>, objective function becomes

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\min _{w, b, \xi} \frac{1}{2}\|w\|^{2}+C \sum_{i=1}^{N} \xi_{i}"/>
</p>

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?s.t. \quad y_{i}\left(w \cdot x_{i}+b\right) \geqslant 1-\xi_{i}, \quad i=1,2, \cdots, N"/>
</p>

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\xi_{i} \geqslant 0, \quad i=1,2, \cdots, N"/>
</p>


Similar process as hard margin, we construct a **Lagrange multipliers**  and take the property of Lagrange duality  
- Taking partial derivatives with respect to w, w and <img src="https://latex.codecogs.com/gif.latex?\xi_{\mathrm{i}}"/>, set it to 0
- Taking partial derivatives with respect to <img src="https://latex.codecogs.com/gif.latex?\alpha"/>, apply SMO to optimise it

The Lagrange dual representation of optimised function is:
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\max\ \sum^m_{i=1}\alpha_i-\frac{1}{2}\sum^m_{i,j=1}\alpha_i\alpha_jy_iy_j(x_ix_j)=\min \frac{1}{2}\sum^m_{i,j=1}\alpha_i\alpha_jy_iy_j(x_ix_j)-\sum^m_{i=1}\alpha_i"/>
</p>

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?s.t.\ C\geq\alpha_i \geq 0,i=1,2,...,m\quad \sum^m_{i=1}\alpha_iy_i=0"/>
</p>

## 4. Kernel Trick for Non Linear SVM

When data is not linearly separable, as plots presented below, we apply Kernel trick so that it can be separable.

<p align="center">
<img src="../images/svm-kernel.png" width="500">
</p>

Kernel functions are generalized functions that take two vectors (of any dimension) as input and output a score that denotes how similar the input vectors are.

The Lagrange dual representation  of optimised function is:

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\max \sum^m_{i=1}\alpha_i-\frac{1}{2}\sum^m_{i=1}\sum^m_{j=1}\alpha_i\alpha_jy_iy_j<\phi(x_i)^T,\phi(x_j)>"/>
</p>

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?s.t.\ \sum^m_{i=1}\alpha_iy_i=0,"/>
</p>

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?s.t.\ C\geq\alpha_i \geq 0,i=1,2,...,m "/>
</p>

## 5. Summary
Pros:
- Effective in the higher dimension, especially when the number of features are more than training examples.
- Suitable for both lieanr and non-linear problems
- The hyperplane is affected by only the support vectors thus outliers have less impact.
- SVM is suited for extreme case binary classification.

Cons:
- For larger dataset, it requires a large amount of time to process.
- Does not perform well in case of overlapped classes.

How to choose kernel function?



## 6. Implementation
[SVM Implementation]()

-----
## 7. Reference

[1] :[Support Vector Notes by Andrew Ng](http://cs229.stanford.edu/notes/cs229-notes3.pdf)

[2] :[Scikit-Learn SVM](https://scikit-learn.org/stable/modules/svm.html)

[3] :[Wikipedia SVN](https://en.wikipedia.org/wiki/Support_vector_machine)

[4] :[Implement SVM with Python ](http://blog.csdn.net/wds2006sdo/article/details/53156589)
