import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.ERROR,
                       format='%(asctime)s - %(levelname)s - %(message)s')

class Multiple_Linear_Regression:
    
    def __init__(self) -> None:
        """Initialize the Coefficients."""
        self.coefficients: np.ndarray = None  # Coefficients & Intercept

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the Multiple Linear Regression Model using Ordinary Least Squares (OLS).
        
        Args:
            X (np.ndarray): Matrix of Input Features.
            y (np.ndarray): Target Variable.
        """
        try:
            # Add Column of Ones to X to Account for the Intercept (beta_0)
            X = np.column_stack([np.ones(X.shape[0], dtype=int), X])
            
            # OLS Formula: beta = (X^T X)^-1 X^T y
            XtX = X.T.dot(X)  # X^T X
            XtX_inv = np.linalg.inv(XtX)  # (X^T X)^-1
            Xty = X.T.dot(y)  # X^T y
            self.coefficients = XtX_inv.dot(Xty)  # beta = (X^T X)^-1 X^T y
        except np.linalg.LinAlgError as e:
            logging.error("Linear Algebra Error Occurred while Matrix Inversion.",
                                                exc_info=True)
            raise e
        except Exception as e:
            logging.error("An Error Occurred while Fitting the Model.",
                                    exc_info=True)
            raise e

    def predict(self, X_new: np.ndarray) -> np.ndarray:
        """Predict the Target Variable using the Fitted Model.
        
        Args:
            X_new (np.ndarray): Matrix of New Input Data for Prediction.
            
        Returns:
            np.ndarray: Predicted Target Values.
        """
        try:
            # Add a column of ones to X_new to account for the Intercept
            X_new = np.column_stack([np.ones(X_new.shape[0]), X_new])
            return X_new.dot(self.coefficients)
        except Exception as e:
            logging.error("An Error Occurred during prediction.",
                                     exc_info=True)
            raise e