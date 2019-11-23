import os
import pandas as pd
import numpy as np
from sklearn import linear_model

os.chdir("/Users/tylerharvey/homework3")
iterx = pd.read_excel("iterx.xlsx")
itery = pd.read_excel("itery.xlsx")

n_samples, n_features = (len(itery), len(iterx))
rng = np.random.RandomState(0)
y = rng.randn(n_samples)
X = rng.randn(n_samples, n_features)
linfit = linear_model.SGDRegressor(max_iter=1000, tol=1e-2)
print(linfit.fit(X, y))
print (linfit.intercept_, linfit.coef_)
