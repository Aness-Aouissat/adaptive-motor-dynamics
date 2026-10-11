from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np

polynomial = (PolynomialFeatures(degree = 3, include_bias = False))

def to_polynomial(features: np.ndarray) -> np.ndarray:
    return polynomial.fit_transform(features)

model = LinearRegression()