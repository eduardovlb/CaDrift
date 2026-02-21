import numpy as np
import math

class TargetFunction:
    def compute_function(self, X, a, b):
        raise NotImplementedError

class LinearFunction(TargetFunction):
    def compute_function(self, X: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        # scale_factor = 0.3 / X.shape[0]  # The more parents, the smaller the weights
        # noise = np.random.normal(0, 0.1, size=a.shape)
        # scaled_weights = (a + b) / np.sqrt(X.shape[0])
        value = np.dot(a, X) + b
        return value
    
    def __str__(self) -> str:
        return "Linear Target Function"
    
class PolynomialFunction(TargetFunction):
    def compute_function(self, X: np.ndarray, a: np.ndarray, b: np.ndarray, degree: int = 2) -> float:
        scale_factor = 1.0 / (X.shape[0] ** degree)
        return (np.sum((a * X) ** degree, axis=0)+b) * scale_factor
    
    def __str__(self) -> str:
        return "Polynomial Target Function"
    
class SineFunction(TargetFunction):
    def compute_function(self, X: np.ndarray, a=None, b=None) -> float:
        return np.sum(np.sin(X))# + np.random.normal(0, 0.1)
    
    def __str__(self):
        return "Sine Target Function"
    
class ThresholdFunction(TargetFunction):
    def compute_function(self, X: np.ndarray, a=None, b=None) -> float:
        return (np.sum(X) > 0).astype(float)# + np.random.normal(0, 0.1)
    
    def __str__(self) -> str:
        return "Threshold Target Function"
    
class RadialBasisFunction(TargetFunction):
    def compute_function(self, X: np.ndarray, a=None, b=None) -> float:
        norm_sq = np.sum(X ** 2)
        sigma=1.0
        result = np.exp(-norm_sq / (2 * sigma ** 2))
        return result
    
    def __str__(self) -> str:
        return "Radial Basis Target Function"
    
class CheckerboardFunction(TargetFunction):
    def compute_function(self, X: np.ndarray, a=None, b=None) -> float:
       return np.sum(np.floor(X)) % 2
    
    def __str__(self):
        return "Checkerboard Target Function"
    
# class ExponentialFunction(TargetFunction):
#     def compute_function(self, X: np.ndarray, a=None, b=None) -> float:
#         return np.exp(np.sum(X))# + np.random.normal(0, 0.1)
    
#     def __str__(self):
#         return "Exponential Target Function"
    
class LogarithmicFunction(TargetFunction):
    def compute_function(self, X: np.ndarray, a=None, b=None) -> float:
        safe_X = np.where(X <= 0, 1e-10, X)
        return np.sum(np.log(safe_X))# + np.random.normal(0, 0.1)
    
    def __str__(self):
        return "Logarithmic Target Function"
    
# class GaussianFunction(TargetFunction):
#     def compute_function(self, X: np.ndarray, a=None, b=None) -> float:
#         mean = np.mean(X)
#         variance = np.var(X) + 1e-6  # Prevent division by zero
#         coeff = 1.0 / math.sqrt(2 * math.pi * variance)
#         exponent = -((X - mean) ** 2) / (2 * variance)
#         return np.sum(coeff * np.exp(exponent))
    
#     def __str__(self):
#         return "Gaussian Target Function"