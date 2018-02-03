# -*- coding: utf-8 -*-
"""
Neural network regression example for JDOT
"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#         Nicolas Courty <ncourty@irisa.fr>
#
# License: MIT License


import numpy as np
import data_helper
import jdot
from keras.utils import np_utils
import keras
import time


X,y=data_helper.get_data('supernova-src')
Xtest,ytest=data_helper.get_data('supernova-tgt')
nclasses=len(np.unique(np.hstack((y, ytest))))+1
n_inputs= X.shape[1]
ytest = np_utils.to_categorical(ytest, num_classes=nclasses)
y=np_utils.to_categorical(y,num_classes=nclasses )


def get_model():
    # simple 1D nn
    model=keras.models.Sequential()
    model.add(keras.layers.Dense(80,activation='tanh', input_dim=n_inputs))
    model.add(keras.layers.Dense(80,activation='tanh'))
    model.add(keras.layers.Dense(y.shape[1],activation='softmax'))
    
    model.compile(optimizer='sgd',loss='categorical_crossentropy', metrics=['accuracy'])

    return model

t=time.process_time()
model=get_model()

fit_params={'epochs':500}

model,loss = jdot.jdot_nn_l2(get_model,X,y,Xtest,ytest=ytest,fit_params=fit_params, reset_model=True, nb_epoch=1)
res=model.evaluate(x=Xtest,y=ytest,batch_size=64)
print('\nResults for metrics ' + str(model.metrics_names) + ' ' + str(res))
print(time.process_time()-t)
