Milestone 4
===========

README
------

For this final milestone, I continued to use the boston data and trained an SVM model using an RBF kernel. Next I performed 10 rounds of 10-fold cross validation using each of the machine learning methods I used before and performed t-tests on each combination of tests. We basically rejected the null hypothesis that the distributions of the MSE were the same for every test except for several including the PCA version of gaussian process. This is because there was some wild variation in the MSEs using that machine learning method. The order of best to worst machine learning methods are gaussian process, linear regression, svm, linear regression with PCA, gaussian process with PCA. 