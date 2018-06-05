""" Simumalte easy X and Y variables for a regression
according to a + bX + e = Y
"""
import random
import numpy as np
import os
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation
import pandas as pd

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

a = 5

b = 3

X = []

rng = 10000

for i in range(0, rng):
    X.append(random.uniform(6, 18))

X_norm = []
for i in range(0, rng):
    X_norm.append((X[i] - min(X))/(max(X) - min(X)))

error = []

for i in range(0, rng):
    error.append(random.gauss(0, 4))

Y = []

for i in range(0, rng):
    number = a + b * X[i] + error[i]
    Y.append(number)

Y_norm = []
for i in range(0, rng):
    Y_norm.append((Y[i] - min(Y))/(max(Y) - min(Y)))

print(np.polyfit(X, Y, 1))

"""Try to fit the same data with Keras
"""

model = Sequential()
model.add(Dense(1, input_dim=1))
model.add(Activation('linear'))
model.add(Dense(100))
model.add(Activation('linear'))
model.add(Dense(100))
model.add(Activation('linear'))
model.add(Dense(1))
model.add(Activation('linear'))

# For a mean squared error regression problem
model.compile(optimizer='nadam',
              loss='mse')

model.fit(X_norm, Y_norm, epochs=10, batch_size=32)
#score = model.evaluate(X, Y, batch_size=32)

Y_pred = model.predict(X_norm)
Y_pred = Y_pred * (max(Y) - min(Y)) + min(Y)
Y_pred = np.reshape(Y_pred, (1, rng))
Y_dif = Y - Y_pred
# from IPython.display import SVG
# from keras.utils.vis_utils import model_to_dot

# useless plot_model(model, to_file='score.png')

"""It can be seen that the results are reasonable, they are all on one line"""
X = np.reshape(X, (1, rng))
Y = np.reshape(Y, (1, rng))
plt.plot(X, Y_pred, "o")
plt.plot(X, Y, "+")

plt.show()


"""Generate data with a given covariance"""

Cov = np.array([[1, 0.3], [0.3, 1]])
L = np.linalg.cholesky(Cov)

# be careful to use randn and NOT rand!
Sim = np.dot(np.random.randn(100, 2), L).T
Cov_sim = np.cov(Sim)

"""Generate Data for the first 2SLS simulation"""

# X and U are correlated:

Cov = np.array([[2, 0.7], [0.7, 2]])
L = np.linalg.cholesky(Cov)
X_U = np.dot(np.random.randn(rng, 2), L).T
Cov_sim = np.cov(Sim)

X = X_U[0]
U = X_U[1]

c = 9
# true Y
Y = (a + b*X + c*U + np.random.randn(1, rng)).T

import statsmodels.formula.api as sm
df = pd.DataFrame({'X':X.T, 'Y':Y[:,0], 'U':U.T})
reg = sm.ols(formula= 'Y ~ X + U', data = df)
fitted = reg.fit()
fitted.summary()

reg = sm.ols(formula= 'Y ~ X ', data = df)
fitted = reg.fit()
fitted.summary()

# simulate Instrumental variable Z, here are some problems regarding the sizing

Z = (X - np.random.randn(1, rng)) / 4
Z = Z[0,:]
df2 = pd.DataFrame({'X':X.T, 'Z':Z.T})
reg = sm.ols(formula= 'X ~ Z ', data = df2)
fitted = reg.fit()
fitted.summary()


# below here is rather random

model.fit(X, Z, epochs=10, batch_size=32)

X_fitted = model.predict(Z)

model.fit(Y, X_fitted, epochs=10, batch_size=32)

Y_fitted = model.predict(X_fitted)


plt.plot(X_fitted, Y_fitted, "o")
plt.plot(X, Y_fitted, "+")

plt.show()

model.fit(Y, X, epochs=10, batch_size=32)

Y_fitted = model.predict(X)