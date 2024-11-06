import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.ERROR,
                       format='%(asctime)s - %(levelname)s - %(message)s')

class SimpleLinearRegression:
    
    def __init__(self) -> None:
        """Initialize beta Coefficients."""
        self.beta_0: float = 0  # Intercept
        self.beta_1: float = 0  # Slope
    
    def mean(self, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        """Calculate the Mean of x and y.
        
        Args:
            x (np.ndarray): Array of Input Features.
            y (np.ndarray): Array of Target Values.
            
        Returns:
            tuple[float, float]: Mean of x and y.
        """
        try:
            return np.mean(x), np.mean(y)
        except Exception as e:
            logging.error("An Error Occurred while Calculating Mean.",
                                    exc_info=True)
            raise e
        
    def slope(self, x: np.ndarray, y: np.ndarray, X: float, Y: float) -> float:
        """Calculate the slope (beta_1) for the Linear Regression.
        
        Args:
            x (np.ndarray): Array of Input Features.
            y (np.ndarray): Array of Target Values.
            X (float): Mean of x.
            Y (float): Mean of y.
            
        Returns:
            float: Slope of the Regression Line.
        """
        try:
            numerator = sum((i - X) * (j - Y) for i, j in zip(x, y))
            denominator = sum((i - X) ** 2 for i in x)
            return numerator / denominator
        except Exception as e:
            logging.error("An Error Occurred while Calculating the Slope.",
                                    exc_info=True)
            raise e
        
    def intercept(self, X: float, Y: float, beta_1: float) -> float:
        """Calculate the intercept (beta_0) for the Linear Regression.
        
        Args:
            X (float): Mean of x.
            Y (float): Mean of y.
            beta_1 (float): Calculated slope (beta_1).
            
        Returns:
            float: Intercept of the Regression Line.
        """
        try:
            return Y - (beta_1 * X)
        except Exception as e:
            logging.error("An Error Occurred while Calculating the Intercept.",
                                    exc_info=True)
            raise e
        
    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Fit the Linear Regression Model to the provided data.
        
        Args:
            x (np.ndarray): Array of Input Features.
            y (np.ndarray): Array of Target Values.
        """
        try:
            X, Y = self.mean(x, y)
            self.beta_1 = self.slope(x, y, X, Y)
            self.beta_0 = self.intercept(X, Y, self.beta_1)
        except Exception as e:
            logging.error("An Error Occurred during Model Fitting.",
                                    exc_info=True)
            raise e
        
    def predict(self, x: float) -> float:
        """Predict the Target Value for a given Input using the Fitted Model.
        
        Args:
            x (float): Input value for prediction.
            
        Returns:
            float: Predicted Target Value.
        """
        try:
            return self.beta_0 + (self.beta_1 * x)
        except Exception as e:
            logging.error("An Error Occurred during Prediction.",
                                    exc_info=True)
            raise e