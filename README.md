
Description:
The software pre-processes the dataset and use it to derive optimal weights through a regularized logistic regression, and predict the output through a logistic function.
It outputs a graph showing F1 SCORE and RMSE for different combinations of (\lambda, \gamma) trough which it is possible to choose the best couple of parameters and use them to make final predictions.

To run the code:
The file 'run.ipynb' must be in the same folder as 'helpers.py' and 'implementations.py'.
The same folder must contain a folder named 'data', inside this folder there must be a folder named 'dataset' which contains the files 'x_test.csv', 'x_train.csv' and 'y_train.csv'

The output:
The output file 'submission.csv' is created inside the folder 'data'

