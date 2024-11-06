import logging
import numpy as np
from typing import Tuple

# Configure logging
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class PolynomialRegressionQuadratic:

    def __init__(self) -> None:
        """Initialize the Coefficients"""
        self.beta_0: float = 0.0
        self.beta_1: float = 0.0
        self.beta_2: float = 0.0

    def Coefficients(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate the Coefficients for the Quadratic Polynomial Regression.
        
        Parameters:
            x (np.ndarray): 1D array of predictor variable values.
            y (np.ndarray): 1D array of response variable values.
        
        Returns:
            Tuple[float, float, float]: The computed beta_0, beta_1, and beta_2 Coefficients.
        """
        try:
            # Precompute Sums required for Matrix Solution
            n = len(x)
            S_x = np.sum(x)
            S_y = np.sum(y)
            S_xx = np.sum(x**2)
            S_xy = np.sum(x * y)
            S_xxx = np.sum(x**3)
            S_xxy = np.sum(x**2 * y)
            S_xxxx = np.sum(x**4)
            
            # Setup the System of Equations as Matrices A and B
            A = np.array([
                [n, S_x, S_xx],
                [S_x, S_xx, S_xxx],
                [S_xx, S_xxx, S_xxxx]
            ])
            B = np.array([S_y, S_xy, S_xxy])
            
            # Solve for beta Coefficients
            self.beta_0, self.beta_1, self.beta_2 = np.linalg.solve(A, B)
            return self.beta_0, self.beta_1, self.beta_2
        except np.linalg.LinAlgError as e:
            logging.error("Matrix Inversion Error during Coefficient Calculation", exc_info=e)
            raise ValueError("Error in Calculating Coefficients. The Input Data may be Unsuitable for a Quadratic Fit.")
        except Exception as e:
            logging.error("Unexpected Error during Coefficient Calculation", exc_info=e)
            raise

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the Quadratic Polynomial Regression Model to the provided data.
        
        Parameters:
            x (np.ndarray): 1D array of predictor variable values.
            y (np.ndarray): 1D array of response variable values.
        """
        try:
            self.beta_0, self.beta_1, self.beta_2 = self.Coefficients(x, y)
        except Exception as e:
            logging.error("Model Fitting Failed", exc_info=e)
            raise

    def predict(self, x: float) -> float:
        """
        Predict the Response Variable using the Fitted Quadratic Polynomial Model.
        
        Parameters:
            x (float): Predictor Variable Value.
        
        Returns:
            float: Predicted Response Variable Value.
        """
        try:
            return self.beta_0 + (self.beta_1 * x) + (self.beta_2 * x**2)
        except Exception as e:
            logging.error("Prediction Failed", exc_info=e)
            raise ValueError("Failed to make a Prediction. Model may not be Fitted Correctly.")