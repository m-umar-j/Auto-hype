![image](https://github.com/m-umar-j/Auto-hype/assets/132677327/b349be90-40ac-489b-832c-cd281a784764)
# Automatic Hyperparameter Tuning using TPE with Hyperopt

This repository demonstrates how to perform automatic hyperparameter tuning for machine learning models using the Tree-structured Parzen Estimator (TPE) algorithm and compares it with Hyperopt. Specifically, it uses TPE to tune hyperparameters for Support Vector Machines (SVM), Random Forest, and Logistic Regression models.

## Dataset

The example uses the breast cancer dataset (`The_Cancer_data_1500_V2.csv`) for binary classification. The dataset is preprocessed to split features (`X`) and target (`Y`), followed by a train-test split.
dataset: https://www.kaggle.com/datasets/rabieelkharoua/cancer-prediction-dataset
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

TPE algorithm optimizes the hyperparameters for each model type over a predefined number of trials (`max_evals`). The best set of hyperparameters for each model is printed after optimization.
## Best Hyperparameters Found
The results from TPE optimization yield the best hyperparameters for each model. These hyperparameters are compared with those obtained from the initial random search to demonstrate the effectiveness of TPE in improving model performance.
Random forest was the best performing model in both cases.
For HyperOpt, the best set of hyperparameters with ROC AUC score 0.94510 are:

Best parameters for random_forest: {'criterion': 0,
 'max_depth': 350,
 'min_samples_leaf': 5,
 'min_samples_split': 8,
 'n_estimators': 600}

For this model, the best set of hyperparameters with ROC AUC score 0.944430 are:
{'n_estimators': 68,
 'max_depth': 11,
 'min_samples_split': 4,
 'min_samples_leaf': 8}


## Learning Curve Plotting

After hyperparameter tuning, learning curves are plotted for three instances of Random Forest classifiers:
- **Optimized**: Hyperparameters optimized manually
- **HyperOpt**: Hyperparameters optimized using Hyperopt with TPE
- **Random**: Randomly chosen hyperparameters

### Learning Curves of this model, HyperOpt, and random selection of hyperparameters
![image](https://github.com/m-umar-j/Auto-hype/assets/132677327/880e853d-1fa3-4157-a4db-c9b9502b25d4)



Learning curves illustrate the model's performance on training and cross-validation sets across varying training sizes. 

### Plot of ROC AUC score vs trial number for this model
![image](https://github.com/m-umar-j/Auto-hype/assets/132677327/31ced5bf-0f2f-4013-b10f-90dccbba36e5)






