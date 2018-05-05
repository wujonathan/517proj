import numpy as np
import pyGPs
import datetime

dataFilename = '../dataset/debug.csv'

# dataFilename = '../dataset/dota2Smallest.csv'

numpyArr = np.loadtxt(dataFilename, delimiter=",")
yTrain, XTrain = np.hsplit(numpyArr, [1])


def NLPD(y, MU, S2):
	nlpd = 0.5*np.log(2*np.pi*S2) + 0.5*((y-MU)**2)/S2 
	nlpd = np.mean(nlpd)
	return nlpd


K = 10
RBF_ACC = []
RBF_NLPD = []
Matern_ACC = []
Matern_NLPD = []

for x_train, x_test, y_train, y_test in pyGPs.Validation.valid.k_fold_validation(XTrain, yTrain, K):
	print (datetime.datetime.now())
	
	model = pyGPs.GPC()
	model.optimize(x_train, y_train)
	ymu, ys2, fmu, fs2, lp = model.predict(x_test, ys = y_test)
	ymu_class = np.sign(ymu)
	RBF_ACC.append(pyGPs.Validation.valid.ACC(ymu_class, y_test))
	RBF_NLPD.append(NLPD(y_test, ymu, ys2))

	model.setPrior(kernel = pyGPs.cov.Matern())
	model.optimize(x_train, y_train)
	ymu, ys2, fmu, fs2, lp = model.predict(x_test, ys = y_test)
	ymu_class = np.sign(ymu)
	Matern_ACC.append(pyGPs.Validation.valid.ACC(ymu_class, y_test))
	Matern_NLPD.append(NLPD(y_test, ymu, ys2))


with open('gaussianProcessResultsDemo.txt', 'w') as o:
	o.write('Cross Val Scores:' + '\n')
	o.write('RBF kernel:' + '\n')
	o.write('	Accuracy: ' + str(['%.4f' % round(n, 4) for n in RBF_ACC]) + '\n')
	o.write('	Accuracy Average: ' + str('%.4f' % round(np.mean(RBF_ACC), 4)) + '\n')
	o.write('	Accuracy Std: ' + str('%.4f' % round(np.std(RBF_ACC), 4)) + '\n')
	o.write('	NLPD: ' + str(['%.4f' % round(n, 4) for n in RBF_NLPD]) + '\n')
	o.write('	NLPD Mean: ' + str('%.4f' % round(np.mean(RBF_NLPD), 4)) + '\n')
	o.write('	NLPD Std: ' + str('%.4f' % round(np.std(RBF_NLPD), 4)) + '\n')
	o.write('Matern kernel:' + '\n')
	o.write('	Accuracy: ' + str(['%.4f' % round(n, 4) for n in Matern_ACC]) + '\n')
	o.write('	Accuracy Average: ' + str('%.4f' % round(np.mean(Matern_ACC), 4)) + '\n')
	o.write('	Accuracy Std: ' + str('%.4f' % round(np.std(Matern_ACC), 4)) + '\n')
	o.write('	NLPD: ' + str(['%.4f' % round(n, 4) for n in Matern_NLPD]) + '\n')
	o.write('	NLPD Mean: ' + str('%.4f' % round(np.mean(Matern_NLPD), 4)) + '\n')
	o.write('	NLPD Std: ' + str('%.4f' % round(np.std(Matern_NLPD), 4)) + '\n')
