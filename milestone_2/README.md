Milestone 2
===========

README
------

For this milestone, I trained a gaussian process classifier. First, I combined the training and testing dataset to create a combined data set. I used the pyGPs library to create and optimize gaussian process classifiers using RBF and Matern kernels. I evaluated the models by using ten-fold cross validation with both accuracy and NLPB as metrics. The model is generated and evaluated by running "python gaussianProcessModel_pyGPs.py", which produces the file "gaussianProcessResults.txt", which contains the results. 

There were major hurdles with this milestone in regards to having something computationally reasonable. Since the original dataset had over 100000 datapoint with a high featurespace dimentionality, running the classification on the entire dataset was infeasable. As an experiment, it took a little under and hour to run a single fold of our ten-fold cross validation on just 1% of the dataset. Therefore, this tentative analyisis is first performed on about .5% of the entire dataset (which took about an hour and a half to run). 
