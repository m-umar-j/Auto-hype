![image](https://github.com/m-umar-j/Auto-hype/assets/132677327/fc6da6f5-69d1-4b4d-b808-2d7f190786ff)# Automatic Hyperparameter Tuning using TPE with Hyperopt

This repository demonstrates how to perform automatic hyperparameter tuning for machine learning models using Tree-structured Parzen Estimator (TPE) algorithm with Hyperopt. Specifically, it focuses on tuning hyperparameters for Support Vector Machines (SVM), Random Forest, and Logistic Regression models using TPE.

## Dataset

The example uses the breast cancer dataset (`The_Cancer_data_1500_V2.csv`) for binary classification. The dataset is preprocessed to split features (`X`) and target (`Y`), followed by a train-test split.

## Models and Hyperparameter Spaces

Three models are considered with their respective hyperparameter spaces defined:

1. **Support Vector Machine (SVM)**:
   - Parameters: C, kernel, gamma, degree

2. **Random Forest**:
   - Parameters: n_estimators, max_depth, min_samples_split, min_samples_leaf

3. **Logistic Regression**:
   - Parameters: C, penalty, max_iter

## Objective Function

The objective function (`objective_function`) evaluates each set of hyperparameters by performing cross-validation and computing the ROC AUC score. This score is used as the metric to optimize during hyperparameter tuning.

## TPE Hyperparameter Optimization

Using Hyperopt, TPE algorithm optimizes the hyperparameters for each model type over a predefined number of trials (`max_evals`). The best set of hyperparameters for each model is printed after optimization.
## Best Hyperparameters Found
The results from TPE optimization yield the best hyperparameters for each model. These hyperparameters are compared with those obtained from the initial random search to demonstrate the effectiveness of TPE in improving model performance.
Random forest was the best performing model in both cases.
For HyperOpt, the best set of hyperparameters with ROC AUC score 0.9462645 are:

Best parameters for random_forest: {'criterion': 0,
 'max_depth': 990.0,
 'min_samples_leaf': 2.0,
 'min_samples_split': 3.0,
 'n_estimators': 450.0}

For this model, the best set of hyperparameters with ROC AUC score 0.94467389 are:
{'n_estimators': 64,
 'max_depth': 5,
 'min_samples_split': 2,
 'min_samples_leaf': 6}


## Learning Curve Plotting

After hyperparameter tuning, learning curves are plotted for three instances of Random Forest classifiers:
- **Optimized**: Hyperparameters optimized manually
- **HyperOpt**: Hyperparameters optimized using Hyperopt with TPE
- **Random**: Randomly chosen hyperparameters

### Learning Curves of this model, HyperOpt and random selection of hyperparameters
![image](https://github.com/m-umar-j/Auto-hype/assets/132677327/dc6de72c-b48d-48e2-8425-9c00c28cd6c2)


Learning curves illustrate the model's performance on training and cross-validation sets across varying training sizes. 

### Plot of ROC AUC score vs trial number for this model
![image](https://github.com/m-umar-j/Auto-hype/assets/132677327/23216823-76f3-4a44-a955-b586d045bd0b)





