import logging
import numpy as np

# Configure Logging
logging.basicConfig(level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class LogisticRegressionCostFunction:
    
    def __init__(self) -> None:
        """Initialize the Cost Function Class."""
        pass
    
    def binary_cross_entropy(self, y_true, y_pred, epsilon=1e-9):
        """
        Compute the Binary Cross-Entropy Loss.
    
        Parameters:
            y_true (np.ndarray): Array of True Binary Labels (0 or 1).
            y_pred (np.ndarray): Array of Predicted Probabilities (between 0 and 1).
            epsilon (float): Small Value to Prevent Log(0).
    
        Returns:
            float: The Computed Binary Cross-Entropy Loss.
        """
        try:
            y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
            loss = -np.mean(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
            return loss
        
        except Exception as e:
            logging.error("An Error Occurred: ", exc_info=e)
            raise e