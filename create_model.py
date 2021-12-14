import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_moons


# Create random linear dataset
x_train = np.random.randn(30, 2) * 1.5
y_train = np.random.randn(30) + 2 * x_train[:, 0] - 0.5 * x_train[:, 1]

# Create and fit linear model
model = LinearRegression().fit(x_train, y_train)

# Save model
joblib.dump(model, "linear_regression.joblib")


# Create classification dataset for regression
X, y = make_moons(200, noise=0.3, random_state=42)
model = LogisticRegression().fit(X, y)

joblib.dump(model, "logistic_regression.joblib")