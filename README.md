# Transpileur C (Regression)

auteur: Quentin Le Helloco

## Utilisation
> python3 create_model.py

Pour creer les deux models utilises.

> python3 transpile_regression.py

Permet de choisir "logistic" ou "linear". Le process va ensuite creer puis compiler un .c avec des features predefiniees et renvoyer l'output du .c puis celui du python (groundtruth). Les deux devraient etre similaires.

## Requirement
Scikit-learn
Joblib
numpy
subprocess