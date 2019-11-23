from sklearn.datasets import make_regression
import numpy 

X, y, coefficients = make_regression(
    n_samples=50,
    n_features=1,
    n_informative=1,
    n_targets=1,
    noise=5,
    coef=True,
    random_state=1
)

def cv_ridge (X,y, nfolds, lambdas): 
  n = len(y)
  k = len(lambdas)
  di = y.shape
  p = X.shape
  msp = numpy.zeros((nfolds, k))
  for j in range(1,nfolds): 
    y_test = y[nfolds[[ j ]], ]
    y_train = y[ -nfolds[[ j ]], ]
    my = applymap(y_train, statistics.mean)
    yy = numpy.asmatrix(y_train + my)
    x_train = X[ -nfolds[[ j ]] ]
    x_test = X[nfolds[[ j ]]]
    sa = sd.shape(x_train)
    d  = sa.d
    v  = numpy.tranpose(sa.v)
    tu = numpy.tranpose(sa.u)
    d2 = d**2
    A  = numpy.matmal((d*tu), yy) 
    for i in range(i, k): 
       beta = numpy.cross(v / d+ lambdas.index(i), A)
       est = numpy.matmal(x_test, beta)
       est_2= pandas.data.frame(est)/my
       msp = applymap((y_test - est_2)**2, statistics.mean) 
    return(list(lambdas[min(msp)]))
    
cv_ridge(X, y, 10, [10])
