
run.ipynb

Description:
The software pre-processes the dataset and use the optimal weights found with 'optimal_weights.ipynb' in order to make optimal predictions, 

To run the code:
The file 'optimal_weights.ipynb' must be in the same folder as 'helpers.py' and 'implementations.py'.
The same folder must contain a folder named 'data', inside this folder there must be a folder named 'dataset' which contains the files 'x_test.csv', 'x_train.csv' and 'y_train.csv'

The output:
The software generates a file named 'submission.csv' in the folder 'Data'



optimal_hyperparameters.ipynb

Description:
The software pre-processes the dataset and use it to derive optimal weights through a regularized logistic regression, and predict the output through a logistic function.
It outputs a graph showing F1 SCORE and RMSE for different combinations of (\lambda, \gamma) trough which it is possible to choose the best couple of parameters and use them to make final predictions.
The software takes about 6 hours to run, so you should run it only if you want to see the distributions of optimal values.

To run the code:
The file 'optimal_weights.ipynb' must be in the same folder as 'helpers.py' and 'implementations.py'.
The same folder must contain a folder named 'data', inside this folder there must be a folder named 'dataset' which contains the files 'x_test.csv', 'x_train.csv' and 'y_train.csv'



Implementations.py

Description:
Contains functions for regression and losses used in 'optimal_hyperparameters.ipynb' and 'run.ipynb' 




Helpers.py


Description:
Contains functions to load the data and to create the output file 