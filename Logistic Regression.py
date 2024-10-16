import logging
import numpy as np

# Configure Logging
logging.basicConfig(level=logging.INFO)

class LogisticRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        """
        Initialize the Logistic Regression Model with Learning Rate and Iterations.
        """
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        """
        Compute the Sigmoid Function for the Input Z.
        """
        try:
            return 1 / (1 + np.exp(-z))
        except Exception as e:
            logging.error("Error in Sigmoid Function: %s", e, exc_info=True)
            raise

    def log_likelihood(self, y, y_pred):
        """
        Compute the Log-Likelihood for the Predicted and Actual Values.
        """
        try:
            return np.sum(y * np.log(y_pred + 1e-9) + (1 - y) * np.log(1 - y_pred + 1e-9))
        except Exception as e:
            logging.error("Error in Log_Likelihood Function: %s", e, exc_info=True)
            raise

    def gradient_descent(self, X, y, y_pred):
        """
        Perform Optimization using Gradient Descent and Update Weights and Bias.
        """
        try:
            m = X.shape[0]
            dw = np.dot(X.T, (y - y_pred)) / m
            db = np.sum(y - y_pred) / m

            self.weights += self.learning_rate * dw
            self.bias += self.learning_rate * db
        except Exception as e:
            logging.error("Error in Gradient Descent: %s", e, exc_info=True)
            raise

    def fit(self, X, y):
        """
        Train the Logistic Regression Model using the Input data X and Labels y.
        """
        try:
            m, n = X.shape
            self.weights = np.zeros(n)
            self.bias = 0

            for i in range(0, self.iterations):
                z = np.dot(X, self.weights) + self.bias
                
                y_pred = self.sigmoid(z)
                if i % 100 == 0:
                    log_likelihood_value = self.log_likelihood(y, y_pred)
                    logging.info(f"Iteration {i}, Log-Likelihood: {log_likelihood_value}")

                self.gradient_descent(X, y, y_pred)
        except Exception as e:
            logging.error("Error during Model Training: %s", e, exc_info=True)
            raise

    def predict(self, X):
        """
        Predict Binary Class Labels for the Input Data X.
        """
        try:
            z = np.dot(X, self.weights) + self.bias
            y_pred_prob = self.sigmoid(z)

            return np.array([1 if prob > 0.5 else 0 for prob in y_pred_prob])
        except Exception as e:
            logging.error("Error during Prediction: %s", e, exc_info=True)
            raise