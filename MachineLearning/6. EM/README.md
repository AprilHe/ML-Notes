------
## Outline
- [1. EM Algorithm](#1-EM-Algorithm)
- [2. EM Example](#2-EM-Example)
- [3. Mathematics Behind EM Algorithm](#2-Mathematics-Behind-EM-Algorithm)
- [4. Reference](#4-Reference)


## 1. EM Algorithm
The expectation–maximization (EM) algorithm is an iterative method to find (local) maximum likelihood or maximum a posteriori (MAP) estimates of parameters in statistical models in the presence of latent variables.

1. Create a function for the expectation of the log-likelihood evaluated using the current estimate on the distribution of latent variables.
2. Maximize the expected log-likelihood found on the E step, then estimated parameters are then used to determine the distribution of the latent variables in the next E step.

Iterate steps 1 and 2 until convergence.

**Applications：**



## 2. EM Example
Suppose we have two coins A and B, to estimate the probability of head P(A) and P(B), we flip the coins to get estimated probability. From the results in the table below, we have P(A) = 0.4, P(B) = 0.5.


| Coin | Result      | Summary    |
| ---- | ---------- | ------- |
| A    | HHTHT| 3H-2T |
| B    | TTHHT | 2H-3T |
| A    | HTTTT | 1H-4T |
| B    | HTTHH | 3H-2T |
| A    | THHTT| 2H-3T |

A more complicated situation and we will use EM Algorithm - if we don't know which coin we flipped in every round (Latent Variables-z), how we get the probability of head for two coins.

| Coin    | Result   | Summary |
| ----    | -------- | ------- |
| Unknown | HHTHT | 3H-2T |
| Unknown | TTHHT | 2H-3T |
| Unknown | HTTTT | 1H-4T |
| Unknown | HTTHH | 3H-2T |
| Unknown | THHTT | 2H-3T |

To solve this problem, we give an initiated values of P(A) and P(B) to estimate which coin we flipped in each round (z1, z2, z3, z4, z5).

1. Initiate a random value of probability of head for coin A is 0.2, coin B is 0.7. P(A) = 0.2, P(B)= 0.7
2. E-Step: Calculate the probability of results in every round given initiated probability.


| Round | Results | If Coin is A  | If Coin is B |
| ----  | ------- | ------------  | ------------ |
| 1     | 3H2T    |  0.00512      | 0.03087      |
| 2     | 2H3T    |  0.02048      | 0.01323      |
| 3     | 1H4T    |  0.08192      | 0.08192      |
| 4     | 3H2T    |  0.00512      | 0.03087      |
| 5     | 2H3T    |  0.02048      | 0.01323      |

3. M-Step: Take the maximal probability in each round, we have {z1=B, z2=A, z3=A, z4=B, z5=A}, then we recalculate the probability of head with the estimation on latent variables,  P(A) = 0.33, P(B)=0.6

4. Iterate step 2 and 3, we can finally get P(A) = 0.4, P(B) = 0.5


## 3. Mathematics Behind EM Algorithm
To be updated



## 4. Reference
1. [What is the expectation maximization algorithm?](https://datajobs.com/data-science-repo/Expectation-Maximization-Primer-[Do-and-Batzoglou].pdf)
