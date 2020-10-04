------
# Time Series

## Contents

- [1. Basics of Time Series](#1-Basics-of-Time-Series)



## 1. Basics of Time Series
A time series is a series of data points indexed (or listed or graphed) in time order. Any time series may be split into the following components:

     Base + Trend + seasonal/cyclical + random

- A trend is observed when there is an increasing or decreasing slope observed in the time series.
- Seasonality is observed when there is a distinct repeated pattern observed between regular intervals due to seasonal factors. It could be because of the month of the year, the day of the month, weekdays or even time of the day.
- Trend and seasonality are reasons of non-stationary.

<p align="center">
<img src="../images/tspatterns.png" width="600"></a>
</p>

### 1.1 Stationarity
A stationary series is one where statistical properties of the series like mean, variance and autocorrelation are not function of time, which is constant over time but autocovariance can be a function of lags.

- The mean of the series should not be a function of time. The red graph below is not stationary because the mean increases over time.

<p align="center">
<img src="../images/stationary1.png" width="500"></a>
</p>

- The variance of the series should not be a function of time. This property is known as homoscedasticity. Notice in the red graph the varying spread of data over time.

<p align="center">
<img src="../images/stationary2.png" width="500"></a>
</p>

- Finally, the covariance of the i th term and the (i + m) th term should not be a function of time. In the following graph, you will notice the spread becomes closer as the time increases. Hence, the covariance is not constant with time for the ‘red series’.

<p align="center">
<img src="../images/stationary3.png" width="500"></a>
</p>

In a linear regression, we assume all observations are independent. In a time series, observations are time dependent. It turns out that a lot of nice results that hold for independent random variables (law of large numbers and central limit theorem) hold for stationary random variables. So by making the data stationary, we can actually apply regression techniques to this time dependent variable and easy to predict.

**Stationary is important as**:
- A stationary time series assume future statistical properties are the same or proportional to current statistical properties, which is simple to predict.
- Most of the TS models work on the assumption that the TS is stationary (covariance-stationarity).
- Intuitively, if a TS has a particular behaviour over time, there is a very high probability that it will follow the same in the future.
- Autoregressive forecasting models are essentially linear regression models that utilize the lag(s) of the series itself as predictors. As linear regression works best if the predictors (X variables) are not correlated against each other.


**To check stationary**
- Visualizing the data to identify a changing mean or variation in the data
- Unit Root Tests
   - Augmented Dickey Fuller test (ADF Test): the null hypothesis is the time series possesses a unit root and is non-stationary
   - Kwiatkowski-Phillips-Schmidt-Shin – KPSS test (trend stationary): used to test for trend stationarity. The null hypothesis and the P-Value interpretation is just the opposite of ADF test
   - Philips Perron test (PP Test)


**To transform the data and make it more stationary**

- Differencing the Series (once or more)
- Take the log of the series
- Take the nth root of the series
- Combination of the above


### 1.2 Autocorrelation
As mentioned, a time series is decomposed into three components:

        trend + seasonal/cyclical+ random

The random component is called the residual or error. It is simply the difference between our predicted value(s) and the observed value(s). Autocorrelation (Serial correlation) is when the residuals (errors) of our TS models are correlated with each other. The residuals (errors) of a stationary TS are serially uncorrelated by definition.

### 1.3 White Noise
A time series is a white noise process if
- it has serially uncorrelated errors, or errors are independent and identically distributed (i.i.d.)
- the expected mean of those errors is equal to zero.

If our TSM is appropriate and successful at capturing the underlying process, the residuals of our model will be i.i.d. and resemble a white noise process.


### 1.4 Random Walks
A Random Walk is a time series model Xt such that Xt = Xt-1 + Wt, where Wt is a discrete white noise series. The random walk is non-stationary because the covariance between observations is time-dependent. If the TS we are modelling is a random walk it is unpredictable. To make a random walk stationary, we can use first differences of series, xt - xt-1 = wt, which should equal a white noise process.  

### 1.5 ACF and PACF
Autocorrelation (ACF) is the correlation of a series with its own lags. If a series is significantly autocorrelated, that means, the previous values of the series (lags) may be helpful in predicting the current value.

Partial Autocorrelation (PACF) also conveys similar information but it conveys the pure correlation of a series and its lag, excluding the correlation contributions from the intermediate lags.


## 2. Decompose Time Series
### 2.1  Detrend a time series
Detrending a time series is to remove the trend component from a time series.

- Subtract the line of best fit from the time series. The line of best fit may be obtained from a linear regression model with the time steps as the predictor
- Subtract the mean
- Subtract the trend component obtained from time series decomposition(seasonal_decompose)
- Apply a filter like Baxter-King filter(statsmodels.tsa.filters.bkfilter) or the Hodrick-Prescott Filter (statsmodels.tsa.filters.hpfilter) to remove the moving average trend lines or the cyclical components.

### 2.2 Deseasonalize a time series
- Take a moving average with length as the seasonal window. This will smoothen in series in the process ((Moving average, Exponentially Weighted Moving Average)
- Seasonal difference the series (subtract the value of previous season from the current value)
- Divide the series by the seasonal index obtained from STL decomposition

### 2.3 Treat missing values
Basic ways of missing value imputation in time series:
- Backward Fill
- Linear Interpolation
- Quadratic interpolation
- Mean of nearest neighbors
- Mean of seasonal couterparts



## 3. Basic Models for TS
### 3.1 Linear model
The basic equation for linear model to illustrate a linear trend of time:

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?y_%7Bt%7D%20%3D%20b_%7B0%7D%20&plus;%20b_%7B1%7D*t%20&plus;%20%5Cvarepsilon%20_%7Bt%7D" /></a>
</p>

In this model the value of the dependent variable is determined by the beta coefficients and a singular independent variable, time.

### 3.2 Log-Linear Models
It is similar to linear models except data points form an exponential function that represent a constant rate of change with respect to each time step.

### 3.3 Autoregressive Models - AR(p)
When the dependent variable is regressed against one or more lagged values of itself the model is called autoregressive. The formula looks like this:

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?x_%7Bt%7D%20%3D%20%5Calpha%20_%7B1%7D*x_%7Bt-1%7D%20...%20%5Calpha%20_%7Bp%7D*x_%7Bt-p%7D%20&plus;%20w_%7Bt%7D%20%3D%20%5Csum_%7Bi%3D1%7D%5E%7BP%7D%5Calpha%20_%7Bi%7Dx_%7Bt-i%7D%20&plus;%20w_%7Bt%7D" />
</p>

p is the order of the model, which can be decided by PACF plot, it represents the number of lagged variables used within the model. For example an AR(2) model or second-order autoregressive model:
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?x_%7Bt%7D%20%3D%20%5Calpha%20_%7B1%7D*x_%7Bt-1%7D%20&plus;%20%5Calpha%20_%7B2%7D*x_%7Bt-2%7D%20&plus;%20w_%7Bt%7D" />
</p>

The absolute value of alpha is less than 1 with the assumption of stationary.

### 3.4 Moving Average Models - MA(q)
MA(q) model is a linear combination of past white noise error terms as opposed to a linear combination of past observations like the AR(p) model.

The motivation for the MA model is that we can observe "shocks" in the error process directly by fitting a model to the error terms. In an MA(q) model these shocks are observed indirectly by using the ACF on the series of past observations. The formula for an MA(q) model is:

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?x_%7Bt%7D%20%3D%20w_%7Bt%7D%20&plus;%20%5Cbeta%20_%7B1%7D%20w_%7Bt-1%7D%20&plus;%20...%20&plus;%20%5Cbeta%20_%7Bp%7D%20w_%7Bt-p%7D%20%3D%20w_%7Bt%7D%20&plus;%20%5Csum_%7Bi%3D1%7D%5E%7BP%7D%5Cbeta%20_%7Bi%7D%20w_%7Bt-i%7D" />
</p>


w is white noise with E(wt) = 0 and variance of sigma squared.

The absolute value of beta is less than 1 with the assumption of stationary.

## 4. Autoregressive Moving Average Models - ARMA(p, q)
The ARMA model is simply the merger between AR(p) and MA(q) models. From finance perspective:

- AR(p) models try to capture (explain) the momentum and mean reversion effects often observed in trading markets.
- MA(q) models try to capture (explain) the shock effects observed in the white noise terms. These shock effects could be thought of as unexpected events affecting the observation process e.g. Surprise earnings, A terrorist attack, etc.

ARMA's weakness is that it ignores the volatility clustering effects found in most financial time series.

The model formula is:

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign*%7D%20x_%7Bt%7D%20%26%3D%20%5Calpha%20_%7B1%7D*x_%7Bt-1%7D%20...%20%5Calpha%20_%7Bp%7D*x_%7Bt-p%7D%20&plus;%20w_%7Bt%7D%20&plus;%20%5Cbeta%20_%7B1%7D%20w_%7Bt-1%7D%20&plus;%20...%20&plus;%20%5Cbeta%20_%7Bp%7D%20w_%7Bt-p%7D%20%5C%5C%20%26%3D%20%5Csum_%7Bi%3D1%7D%5E%7BP%7D%5Calpha%20_%7Bi%7Dx_%7Bt-i%7D%20&plus;%20w_%7Bt%7D%20&plus;%20%5Csum_%7Bi%3D1%7D%5E%7BQ%7D%5Cbeta%20_%7Bi%7D%20w_%7Bt-i%7D%20%5Cend%7Balign*%7D" />
</p>

To decide p and q for ARMA model:
- An AR signature corresponds to a PACF plot displaying a sharp cut-off and a more slowly decaying ACF;
- An MA signature corresponds to an ACF plot displaying a sharp cut-off and a PACF plot that decays more slowly.


## 5. Autoregressive Integrated Moving Average Models - ARIMA(p, d, q)
ARIMA is a natural extension to the class of ARMA models. As previously mentioned many of our TS are not stationary, however they can be made stationary by differencing. We saw an example of this when we took the first difference of a Guassian random walk and proved that it equals white noise.

The "d" references the number of times we are differencing the series, which is the minimum differencing required to get a near-stationary series which roams around a defined mean and the ACF plot reaches to zero fairly quick.


## 6. Autoregressive Conditionally Heteroskedastic Models - ARCH(p)
ARCH(p) models can be thought of as simply an AR(p) model applied to the variance of a time series. Another way to think about it, is that the variance of our time series NOW at time t, is conditional on past observations of the variance in previous periods.

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?Var%28y_%7Bt%7D%20%7C%20y_%7Bt-1%7D%29%20%3D%20%5Csigma%20_%7Bt%7D%5E%7B2%7D%3D%5Calpha%20_%7B0%7D%20&plus;%20%5Calpha%20_%7B1%7Dy_%7Bt-1%7D%5E2" />
</p>

Assuming the series has zero mean we can express the model as:

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?y_%7Bt%7D%3D%5Csigma%20_%7Bt%7D%5Cepsilon%20_%7Bt%7D%20%2C%20with%20%5Csigma%20_%7Bt%7D%20%3D%20%5Csqrt%7B%5Calpha%20_%7B0%7D%20&plus;%20%5Calpha%20_%7B1%7Dy_%7Bt-1%7D%5E2%7D%20%2C%20and%20%5Cepsilon%20_%7Bt%7D%20%5Csim%20iid%280%2C1%29" />
</p>

## 7. Generalized Autoregressive Conditionally Heteroskedastic Models - GARCH(p,q)

Simply put GARCH(p, q)  is an ARMA model applied to the variance of a time series i.e., it has an autoregressive term and a moving average term. The AR(p) models the variance of the residuals (squared errors) or simply our time series squared. The MA(q) portion models the variance of the process. The basic GARCH(1, 1) formula is:

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?%5Cepsilon%20_%7Bt%7D%20%3D%20%5Csigma%20_%7Bt%7D%20w_%7Bt%7D%5C%5C%20%5Csigma%20_%7Bt%7D%5E%7B2%7D%20%3D%20%5Calpha%20_%7B0%7D%20&plus;%5Calpha%20_%7B1%7D%5Cepsilon%20_%7Bt-1%7D%5E%7B2%7D%20&plus;%20%5Cbeta%20_%7B1%7D*%5Csigma%20_%7Bt-1%7D%5E%7B2%7D" />
</p>


w is white noise, and alpha and beta are parameters of the model. Also alpha_1 + beta_1 must be less than 1 or the model is unstable. We can simulate a GARCH(1, 1) process below.


## 8. Reference
[1. Data transformations and forecasting models  ](http://people.duke.edu/~rnau/whatuse.htm)
