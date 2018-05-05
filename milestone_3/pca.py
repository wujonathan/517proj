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

def NLPD(y, MU, S2):
	nlpd = 0.5*np.log(2*np.pi*S2) + 0.5*((y-MU)**2)/S2 
	nlpd = np.mean(nlpd)
	return nlpd


pca = PCA(n_components=2)
newXTrain = pca.fit_transform(XTrain)

K = 10
scoresR2 = []
scoresMSE = []
RBF_RMSE = []
RBF_NLPD = []

for x_train, x_test, y_train, y_test in pyGPs.Validation.valid.k_fold_validation(XTrain, yTrain, K):
	pca = PCA(n_components=2)
	new_x_train = pca.fit_transform(x_train)

	lr = lm.LinearRegression()
	lr.fit(new_x_train, y_train)
	
	new_x_test = pca.transform(x_test)
	scoresR2.append(lr.score(new_x_test, y_test))

	y_pred = lr.predict(new_x_test)
	scoresMSE.append(mse(y_test, y_pred))

	model = pyGPs.GPR()
	model.optimize(new_x_train, y_train)
	ymu, ys2, fmu, fs2, lp = model.predict(new_x_test, ys = y_test)
	RBF_RMSE.append(pyGPs.Validation.valid.RMSE(ymu, y_test))
	RBF_NLPD.append(NLPD(y_test, ymu, ys2))
	

with open('pcaLinearClassResults.txt', 'w') as o:
	o.write('Cross Val Scores:' + '\n')
	o.write('R2:' + '\n')
	o.write(str(['%.4f' % round(n, 4) for n in scoresR2]) + '\n')
	o.write(str('%.4f' % round(np.mean(scoresR2), 4)) + '\n');
	o.write('MSE:' + '\n')
	o.write(str(['%.4f' % round(n, 4) for n in scoresMSE]) + '\n')
	o.write(str('%.4f' % round(np.mean(scoresMSE), 4)) + '\n');


with open('pcaGaussianProcessResults.txt', 'w') as o:
	o.write('Cross Val Scores:' + '\n')
	o.write('RBF kernel:' + '\n')
	o.write('	RMSE: ' + str(['%.4f' % round(n, 4) for n in RBF_RMSE]) + '\n')
	o.write('	RMSE Average: ' + str('%.4f' % round(np.mean(RBF_RMSE), 4)) + '\n')
	o.write('	RMSE Std: ' + str('%.4f' % round(np.std(RBF_RMSE), 4)) + '\n')
	o.write('	NLPD: ' + str(['%.4f' % round(n, 4) for n in RBF_NLPD]) + '\n')
	o.write('	NLPD Mean: ' + str('%.4f' % round(np.mean(RBF_NLPD), 4)) + '\n')
	o.write('	NLPD Std: ' + str('%.4f' % round(np.std(RBF_NLPD), 4)) + '\n')


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter([i[0] for i in XTrain], [i[1] for i in XTrain], zs=yTrain)
# ax.set_xlabel('feature 1')
# ax.set_ylabel('feature 2')
# ax.set_zlabel('price')

# plt.show()