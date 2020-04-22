import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
y = np.array([15, 11, 2, 8, 25, 32])

transformer = PolynomialFeatures(degree=2, include_bias=False)
transformer.fit(x)
x_ = transformer.transform(x)
##same fitting and transforming:
x_ = PolynomialFeatures(degree=2, include_bias=False).fit_transform(x)

print(x_)

##Step 3: Create a model and fit it

model = LinearRegression().fit(x_, y)

## Step 4: Get results

r_sq = model.score(x_,y)
print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('coefficients:', model.coef_)

##Same result with different transformation and regression arguments:
x_ = PolynomialFeatures(degree=2, include_bias=True).fit_transform(x)
print(x_)

##when providing fit_intercept=False:
model = LinearRegression(fit_intercept=False).fit(x_,y)
r_sq = model.score(x_,y)
print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('coefficients:', model.coef_)

##Step 5. Predict response

y_pred = model.predict(x_)
print('predicted response:', y_pred, sep='\n')