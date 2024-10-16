import logging

# Configure Logging
logging.basicConfig(level=logging.ERROR,
                          format='%(asctime)s - %(levelname)s - %(message)s')

class LinearRegressionCostFunction:
    
    def __init__(self) -> None:
        """Initialize the Cost Function Class."""
        pass
    
    def mean_squared_error(self, y, Y) -> float:
        """
        Calculate the Mean Squared Error (MSE) between Actual
        and Predicted Values.
        
        Args:
            y (list or np.ndarray): List or Array of Actual Values.
            Y (list or np.ndarray): List or Array of Predicted Values.
            
        Returns:
            float: The Computed Mean Squared Error.
        """
        try:
            if len(y) != len(Y):
                raise ValueError("The lengths of Actual and Predicted Values must be Same.")
            
            mse = (1 / (2 * len(y))) * sum((i - j) ** 2 for i, j in zip(y, Y))
            return mse

        except Exception as e:
            logging.error("An Error Occurred while Calculating Mean Squared Error.",
                                     exc_info=True)
            raise e