# C Transpiler (Regression)

author: Quentin Le Helloco

## Usage
> python3 create_model.py

Create both models for later usage.

> python3 transpile_regression.py

Let you choose between "logistic" or "linear" model. The process will then create and compilate a c file with predefined features and print the output prediction of both the c file and python original model. They should be the same.

## Requirement
Scikit-learn
Joblib
numpy
subprocess
