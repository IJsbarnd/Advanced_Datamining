from sys import version
print(f'Using python version {version.split(" ")[0]}')

from pandas import DataFrame, __version__
print(f'Using pandas version {__version__}')

from sklearn import linear_model, __version__
print(f'Using sklearn version {__version__}')

import model, data


#xs, ys = data.linear('nominal')

my_perceptron = model.Perceptron(dim=2)
skl_perceptron = linear_model.Perceptron(max_iter=1000)

#print(my_perceptron)
#print(f'- bias = {my_perceptron.bias}')
#print(f'- weights = {my_perceptron.weights}')

#yhats = my_perceptron.predict(xs)
#print(DataFrame(xs, columns=['x1', 'x2']).assign(y=ys, ŷ=yhats).head())

#my_perceptron.fit(xs, ys)
#data.scatter(xs, ys, model=my_perceptron)
#print(my_perceptron)
#print(f'- bias = {my_perceptron.bias}')
#print(f'- weights = {my_perceptron.weights}')
#
#
#skl_perceptron.fit(xs, ys)
#data.scatter(xs, ys, model=skl_perceptron)
#print(skl_perceptron)
#print(f'- bias = {skl_perceptron.intercept_[0]}')
#print(f'- weights = {skl_perceptron.coef_[0]}')


xs, ys = data.linear('numeric')

my_linearregression = model.LinearRegression(dim=2)
yhats = my_linearregression.predict(xs)
DataFrame(xs, columns=['x1', 'x2']).assign(y=ys, ŷ=yhats).head()

my_linearregression = model.LinearRegression(dim=2)
my_linearregression.fit(xs, ys)
data.scatter(xs, ys, model=my_linearregression)
print(my_linearregression)
print(f'- bias = {my_linearregression.bias}')
print(f'- weights = {my_linearregression.weights}')

yhats = my_linearregression.predict(xs)
print(DataFrame(xs, columns=['x1', 'x2']).assign(y=ys, ŷ=yhats).head())

skl_linearregression = linear_model.LinearRegression()
skl_linearregression.fit(xs, ys)
data.scatter(xs, ys, model=skl_linearregression)
print(skl_linearregression)
print(f'- bias = {skl_linearregression.intercept_}')
print(f'- weights = {skl_linearregression.coef_}')



























