import numpy as np
from sklearn.metrics.classification import accuracy_score, log_loss
from sklearn.gaussian_process import GaussianProcessClassifier as gpc
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, DotProduct
from sklearn.model_selection import cross_val_score as cvs
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer

# dataTestname = '../dataset/debug.csv'

dataFilename = '../dataset/dota2SmallSubset.csv'

numpyArr = np.loadtxt(dataFilename, delimiter=",")
yTrain, XTrain = np.hsplit(numpyArr, [1])

kf = KFold(10)
RBF_NLPD = []

for train, test in kf.split(Xtrain):
	gp_rbf = gpc(kernel = 1.0 * RBF([1.0]))
	print ("fitting at time: " + str(datetime.datetime.now()))
	gp_rbf.fit(XTrain[train], yTrain[train])

	predict_probs = gp_rbf.predict_probs(XTrain[test])


with open('gaussianProcessResults.txt', 'w') as o:
	o.write('Cross Val Scores:' + '\n')
	o.write('	RBF kernel:' + '\n')
	o.write('		accuracy: ' + str(accuracy_scores_rbf) + '\n')
	# o.write('		NLPD: ' + str(NLPD_scores_rbf) + '\n')
	o.write('	White kernel:' + '\n')
	o.write('		accuracy: ' + str(accuracy_scores_white) + '\n')
	# o.write('		NLPD: ' + str(NLPD_scores_white) + '\n')
	o.write('	Dot Product kernel:' + '\n')
	o.write('		accuracy: ' + str(accuracy_scores_dot) + '\n')
	# o.write('		NLPD: ' + str(NLPD_scores_dot) + '\n')

