from sys import version
print(f'Using python version {version.split(" ")[0]}')

from pandas import DataFrame, __version__
print(f'Using pandas version {__version__}')

from numpy import array, __version__
print(f'Using numpy version {__version__}')

#from tensorflow import keras, __version__
#print(f'Using tensorflow version {__version__}')

import model, data


my_network = model.Layer(3, name='Input') + model.Layer(2, name='Hidden') + model.Layer(1, name='Output')
print(my_network)