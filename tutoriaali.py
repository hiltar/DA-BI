import numpy as np
from sklearn.linear_model import LinearRegression

#Simple Linear Regression With scikit-learn

x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
y = np.array([5, 20, 14, 32, 22, 38])

model = LinearRegression()

model.fit(x,y)

model = LinearRegression().fit(x, y)

r_sq = model.score(x, y)

print('coefficient of determination:', r_sq)
print('--------------------------')

print('intercept:', model.intercept_)
print('--------------------------')

print('slope:', model.coef_)
print('--------------------------')

print()

new_model = LinearRegression().fit(x, y.reshape((-1, 1)))

print('intercept:', new_model.intercept_)
print('--------------------------')

print('slope:', new_model.coef_)

y_pred = model.predict(x)
print('--------------------------')

print('predicted response:', y_pred, sep='\n')

#Multiple Linear Regression With scikit-learn


