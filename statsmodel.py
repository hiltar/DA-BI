import numpy as np
import statsmodels.api as sm

##Step 2: Provide data and transform inputs
x = [[0, 1], [5, 1], [15, 2], [25, 5], [35, 11], [45, 15], [55, 34], [60, 35]]
y = [4, 5, 20, 14, 32, 22, 38, 43]
x, y = np.array(x), np.array(y)

##Add the column of ones to the inputs if you want statsmodels to 
##calculate the intercept b0. It doesn't take b0 into account by default.
##This is just one function call: 
x = sm.add_constant(x)

print(x)
print(y)

##Step 3: Create a model and fit it:
## Careful! The first argument is the output, followed with input.
## creating
model = sm.OLS(y, x)
##fitting
results = model.fit()

##Step 4: Get results
print(results.summary())

print('coefficient of determination:', results.rsquared)
print('adjusted coefficient of determination:', results.rsquared_adj)
print('regression coefficients:', results.params)

##Step 5: Predict response
##This is the predicted response for known inputs:
print('predicted response:', results.fittedvalues, sep='\n')
print('predicted response:', results.predict(x), sep='\n')

##Predictions with new regressors, you can also apply .predict()
##with new data as the argument:
x_new = sm.add_constant(np.arange(10).reshape((-1, 2)))
print(x_new)
y_new = results.predict(x_new)
print(y_new)