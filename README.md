# Cardiovascular Diseases (CVDs) prediction with machine learning
This project is the Project 1 of **EPFL CS-433 Machine Learning**.

**Team name**: The Overfitters   
**Team members**: Hadrien Sevel, Pietro Pecchini, Matthijs Scheerder

## Files

### run.ipynb
This notebook pre-processes the dataset and use the optimal weights found with 'hyperparameters_tuning.ipynb' in order to make optimal predictions.

**To run the code:**   
The notebook must be in the same folder as 'helpers.py' and 'implementations.py'.
The same folder must contain a folder named 'data', inside this folder there must be a folder named 'dataset' which contains the files 'x_test.csv', 'x_train.csv' and 'y_train.csv'.

**Output:**   
The software generates a file named 'submission.csv' in the folder 'data'.



### hyperparameters_tuning.ipynb
This notebook pre-processes the dataset and run a 4-fold cross-validation for multiple values of gamma (step size) and lambda (regularization parameter), our two hyperparameters.  
It outputs a graph showing the RMSE and F1 score for the different combinations of lambda and gamma trough which it is possible to choose the best couple of parameters and use them to make final predictions. It also save the different data obtained in the tuning folder.   
*Disclaimer: this takes a few hours to run.*

**To run the code:**   
The notebook must be in the same folder as 'helpers.py' and 'implementations.py'.
The same folder must contain a folder named 'data', inside this folder there must be a folder named 'dataset' which contains the files 'x_test.csv', 'x_train.csv' and 'y_train.csv'. It should also contain a 'tuning' folder to save the data.



### implementations.py
Contains the functions for regressions and losses used in 'run.ipynb' and 'hyperparameters_tuning.ipynb'.

### helpers.py
Contains the functions to load the data and create the submission file.