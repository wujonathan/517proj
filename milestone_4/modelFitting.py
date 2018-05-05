import numpy as np
from sklearn import linear_model as lm
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error as mse
import pyGPs
import json


xFilename = '../dataset/bostonX.csv'
yFilename = '../dataset/bostonY.csv'

XTrain = np.loadtxt(xFilename, delimiter=",")
yTrain = np.loadtxt(yFilename, delimiter=",")

yTrain = np.array([[i] for i in yTrain])


K = 10

scoresMSE = {"lr" : [], "clf": [], "gp": [], "lrPCA" : [], "gpPCA" : []}
finalScoresMSE = {"lr" : [], "clf": [], "gp": [], "lrPCA" : [], "gpPCA" : []}

for i in xrange(10):
	for x_train, x_test, y_train, y_test in pyGPs.Validation.valid.k_fold_validation(XTrain, yTrain, K, randomise = True):
		

		lr = lm.LinearRegression()
		lr.fit(x_train, y_train)
		y_pred = lr.predict(x_test)
		scoresMSE["lr"].append(mse(y_test, y_pred))


		clf = svm.SVR()
		clf.fit(x_train, y_train)
		y_pred = clf.predict(x_test)
		scoresMSE["clf"].append(mse(y_test, y_pred))


		model = pyGPs.GPR()
		model.optimize(x_train, y_train)
		ymu, ys2, fmu, fs2, lp = model.predict(x_test, ys = y_test)
		scoresMSE["gp"].append(mse(y_test, ymu))

		pca = PCA(n_components=2)
		new_x_train = pca.fit_transform(x_train)
		new_x_test = pca.transform(x_test)

		lr = lm.LinearRegression()
		lr.fit(new_x_train, y_train)
		y_pred = lr.predict(new_x_test)
		scoresMSE["lrPCA"].append(mse(y_test, y_pred))

		model = pyGPs.GPR()
		model.optimize(new_x_train, y_train)
		ymu, ys2, fmu, fs2, lp = model.predict(new_x_test, ys = y_test)
		scoresMSE["gpPCA"].append(mse(y_test, ymu))
	
	for key in scoresMSE:
		finalScoresMSE[key].append(np.mean(scoresMSE[key]))
	scoresMSE = {"lr" : [], "clf": [], "gp": [], "lrPCA" : [], "gpPCA" : []}

	print i



with open('ttest.txt', 'w') as o:
	json.dump(finalScoresMSE, o)
