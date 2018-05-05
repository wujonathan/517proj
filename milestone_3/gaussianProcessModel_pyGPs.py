import numpy as np
import pyGPs
import datetime

dataFilename = '../dataset/debug.csv'

# dataFilename = '../dataset/dota2Smallest.csv'

xFilename = '../dataset/bostonX.csv'
yFilename = '../dataset/bostonY.csv'

XTrain = np.loadtxt(xFilename, delimiter=",")
yTrain = np.loadtxt(yFilename, delimiter=",")


def NLPD(y, MU, S2):
	nlpd = 0.5*np.log(2*np.pi*S2) + 0.5*((y-MU)**2)/S2 
	nlpd = np.mean(nlpd)
	return nlpd


K = 10
RBF_RMSE = []
RBF_NLPD = []

for x_train, x_test, y_train, y_test in pyGPs.Validation.valid.k_fold_validation(XTrain, yTrain, K):
	model = pyGPs.GPR()
	model.optimize(x_train, y_train)
	ymu, ys2, fmu, fs2, lp = model.predict(x_test, ys = y_test)
	RBF_RMSE.append(pyGPs.Validation.valid.RMSE(ymu, y_test))
	RBF_NLPD.append(NLPD(y_test, ymu, ys2))


with open('gaussianProcessResults.txt', 'w') as o:
	o.write('Cross Val Scores:' + '\n')
	o.write('RBF kernel:' + '\n')
	o.write('	RMSE: ' + str(['%.4f' % round(n, 4) for n in RBF_RMSE]) + '\n')
	o.write('	RMSE Average: ' + str('%.4f' % round(np.mean(RBF_RMSE), 4)) + '\n')
	o.write('	RMSE Std: ' + str('%.4f' % round(np.std(RBF_RMSE), 4)) + '\n')
	o.write('	NLPD: ' + str(['%.4f' % round(n, 4) for n in RBF_NLPD]) + '\n')
	o.write('	NLPD Mean: ' + str('%.4f' % round(np.mean(RBF_NLPD), 4)) + '\n')
	o.write('	NLPD Std: ' + str('%.4f' % round(np.std(RBF_NLPD), 4)) + '\n')
