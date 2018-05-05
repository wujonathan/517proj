import numpy as np
from sklearn import linear_model as lm
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error as mse, r2_score as r2
from sklearn.model_selection import cross_val_score as cvs

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pyGPs

# trainTestname = '../dataset/debug.csv'
# testTestname = '../dataset/debug.csv'

xFilename = '../dataset/bostonX.csv'
yFilename = '../dataset/bostonY.csv'

XTrain = np.loadtxt(xFilename, delimiter=",")
yTrain = np.loadtxt(yFilename, delimiter=",")

K = 10
scoresR2 = []
scoresMSE = []

for x_train, x_test, y_train, y_test in pyGPs.Validation.valid.k_fold_validation(XTrain, yTrain, K):

	lr = lm.LinearRegression()
	lr.fit(x_train, y_train)

	scoresR2.append(lr.score(x_test, y_test))
	
	y_pred = lr.predict(x_test)
	scoresMSE.append(mse(y_test, y_pred))


with open('linearClassResults.txt', 'w') as o:
	o.write('Cross Val Scores:' + '\n')
	o.write('R2:' + '\n')
	o.write(str(['%.4f' % round(n, 4) for n in scoresR2]) + '\n')
	o.write(str('%.4f' % round(np.mean(scoresR2), 4)) + '\n');
	o.write('MSE:' + '\n')
	o.write(str(['%.4f' % round(n, 4) for n in scoresMSE]) + '\n')
	o.write(str('%.4f' % round(np.mean(scoresMSE), 4)) + '\n');