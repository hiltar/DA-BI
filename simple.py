import numpy as np
from sklearn.linear_model import LinearRegression

x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
y = np.array([5, 20, 14, 32, 22, 38])

print(x)
print(y)

##4. Get results
model = LinearRegression().fit(x,y)

r_sq = model.score(x,y)
print('coefficient of determination', r_sq)

print('intercept:', model.intercept_)
print('slope:', model.coef_)

new_model = LinearRegression().fit(x,y.reshape((-1,1)))
print('intercept:', new_model.intercept_)
print('slope:', new_model.coef_)

##5. Predict response
y_pred = model.predict(x)
print('predicted response:', y_pred, sep='\n')

##Nearly identical way to predict the response:
y_pred = model.intercept_ + model.coef_ * x
print('predicted response:', y_pred, sep='\n')

## you can use fitted models to calculate the outputs based on some other, new inputs:
x_new = np.arange(5).reshape((-1, 1))
print(x_new)
y_new = model.predict(x_new)
print(y_new)