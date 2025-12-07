preprocessing.py
Handles all data loading, cleaning, and feature preparation.
Performs median imputation for numeric values and most-frequent imputation for categorical values.
Applies ordinal encoding and standardization.
Creates engineered features (TotalSF and TotalPorch).
Returns cleaned training data for all models.

plot.py
Generates all plots used in the report.
Creates prediction-vs-actual scatter plots, residual plots, and comparison charts for MSE and MAE.
Outputs images into the plots/ folder.

Linear_regression.py
Runs a basic linear regression model on the preprocessed dataset.
Prints MSE and MAE.

polynomial_regression.py
Runs polynomial regression (degree 2).
Prints MSE and MAE.

mlp_regression.py
Runs a basic multilayer perceptron regressor.
Prints MSE and MAE.

random_forest.py
Implements a Random Forest Regressor, based on methodology from literature model 1.
Prints evaluation metrics.

xgboost_alg.py
Implements XGBoost Regressor, based on methodology from literature model 2.
Prints evaluation metrics.

plots/
Contains all generated figures referenced in the report.
Includes scatter plots, residual plots, MSE comparison, and MAE comparison.

train.csv
Dataset file required by preprocessing.py

Requirements
scikit-learn
xgboost
pandas
numpy
matplotlib
python 3.10+
