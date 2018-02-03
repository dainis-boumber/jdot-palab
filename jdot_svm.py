# -*- coding: utf-8 -*-
"""
Classification example for JDOT
"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#         Nicolas Courty <ncourty@irisa.fr>
#
# License: MIT License

import numpy as np
import jdot
import classif
import data_helper
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import time

seed=1985
np.random.seed(seed)

X,y=data_helper.get_data('supernova-src')
Xtest,ytest=data_helper.get_data('supernova-tgt')
Y,Yb=classif.get_label_matrix(y)
t=time.process_time()
svm = LinearSVC()
svm.fit(X=X,y=y)
ypred=svm.predict(Xtest)
acc=accuracy_score(y_true=ytest, y_pred=ypred)
print('LinearSVC: ', str(acc))
print(time.process_time()-t)

