from sklearn.datasets import make_regression
import numpy as np

def lm_ridge(X, y, coefficients): 
    alpha = 1
    n, m = X.shape
    I = np.identity(m)
    beta_coef = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X) + alpha * I), X.T), y)
    return(beta_coef)

X, y, coefficients = make_regression(
    n_samples=50,
    n_features=1,
    n_informative=1,
    n_targets=1,
    noise=5,
    coef=True,
    random_state=1
)

lm_ridge(X,y,coefficients)

coefficients


