from sys import version
print(f'Using python version {version.split(" ")[0]}')

from pandas import DataFrame, __version__
print(f'Using pandas version {__version__}')

from sklearn import linear_model, svm, __version__
print(f'Using sklearn version {__version__}')

import model, data

xs, ys = data.linear(outcome='nominal', noise=1.0)

my_neuron = model.Neuron(dim=2, loss=model.hinge)
my_neuron.fit(xs, ys)
data.scatter(xs, ys, model=my_neuron)
print(my_neuron)
print(f'- bias = {my_neuron.bias}')
print(f'- weights = {my_neuron.weights}')

yhats = my_neuron.predict(xs)
print(DataFrame(xs, columns=['x1', 'x2']).assign(y=ys, ŷ=yhats).head())

skl_svm = svm.SVC(kernel='linear')
skl_svm.fit(xs, ys)
data.scatter(xs, ys, model=skl_svm)
print(skl_svm)
print(f'- bias = {skl_svm.intercept_[0]}')
print(f'- weights = {skl_svm.coef_[0]}')

yhats = my_neuron.predict(xs)
print(DataFrame(xs, columns=['x1', 'x2']).assign(y=ys, ŷ=yhats).head())