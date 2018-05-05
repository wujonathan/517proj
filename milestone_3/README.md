Milestone 3
===========

README
------

This milestone brought some difficulties since all the features in the original data I used were categorical. I decided to switch to the boston housing data for this milestone and ran the code used in milestones 1 and 2 on that data as well. 

I performed PCA on the data using the hyperparameter of 2 components (to better visualize the data). First I fitted a linear regression model the same way I did for milestone 1. The results were worse in terms of both R2 values and MSE values. This might be expected since dimentionality reduction with a medium sized feature space will lose a good amount of information. Next I fitted a gaussian model with an RBF kernel the same way I did for milestone 2. Surprisingly, the model with reduced dimentionality did better than the original. This might be because 