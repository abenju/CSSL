import numpy as np
from statsmodels.tsa.vector_ar import var_model
from statsmodels.tsa.ar_model import AutoReg
from sklearn.linear_model import LinearRegression


def gauss_fit(data):
    mu = np.average(data, axis=0, keepdims=True).T
    if data.shape[1] == 1:  # p is known to be square
        cov = np.full((1, 1), np.var(data))
    else:
        cov = np.cov(data, rowvar=False)
    
    return mu, cov


def linear_reg_fit(seq, gts):
    model = LinearRegression(fit_intercept=False).fit(gts, np.squeeze(seq))
    a = model.coef_
    cov = np.cov(np.squeeze(seq) - model.predict(gts), rowvar=False)

    return a, cov


def auto_reg1_fit(data):
    if data.shape[1] == 1:
        model = AutoReg(data, lags=1, trend='n')
        results = model.fit()
        a = np.full((1, 1), results.params[0])

        diff = data.T.squeeze()[1:] - results.fittedvalues
        cov = np.full((1, 1), np.var(diff))
    else:
        model = var_model.VAR(data)
        results = model.fit(maxlags=1)
        #TODO
    return a, cov
