import numpy as np

def get_data(name):
  data = np.genfromtxt('data/' + str(name) + '.csv',delimiter=',')

  X = data[:,:-1]
  y = data[:, -1]

  return X, y