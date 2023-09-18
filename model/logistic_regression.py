"""
Logistic regression model
Classifies 2 classes

Components
- score computation
- classification probability
- loss function
- parameter updating rule

References
- https://developer.ibm.com/articles/implementing-logistic-regression-from-scratch-in-python/
- https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

"""
import logging

import numpy as np


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the accuracy for the predictions.

    :param y_true: The target values
    :param y_pred: The predicted values
    :return: The accuracy
    """
    number_of_values = y_true.shape[0]
    accuracy = (y_true == y_pred).sum() / number_of_values

    return accuracy


class LogisticRegressionCustomModel:
    def __init__(self, log_level: int = logging.INFO):
        """
        Initialise the model's logger

        :param log_level: The model's log level
            e.g. logging.INFO, logging.DEBUG
        """

        self.weights: np.ndarray = None
        self.bias: float = None
        self.train_accuracies: list = None
        self.train_losses: list = None
        self.train_probabilities: list = None
        self._train_iterations: int = 0

        self.logger = logging.getLogger()
        self.logger.setLevel(log_level)

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        max_iterations: int = 1000,
        learning_rate: float = 0.1,
        tolerance: float = 1e-6,
    ) -> None:
        """
        Optimise the model's weights and bias for the given training data.

        :param x_train: The training data inputs
        :param y_train: The training data target values
        :param max_iterations: Number of times to pass the entire training dataset
        :param learning_rate: The hyperparameter that controls the step size when updating the parameters in gradient descent
        :param tolerance: The convergence criterion.
            Stop the optimisation process when the change in loss is less than the tolerance
        :return: N/A
        """

        self.weights = np.zeros(x_train.shape[1])
        self.bias = 0
        self.train_accuracies = []
        self.train_losses = []
        self._train_iterations = 0

        self.logger.info(
            f"""Fitting Model to training data.
            Using Hyperparameters:
            {max_iterations=}
            {learning_rate=}
            {tolerance=}"""
        )

        for i in range(max_iterations):
            x_dot_weights = np.matmul(self.weights, x_train.transpose())

            pred = self._apply_logistic_function(x_dot_weights)

            gradient_w, gradient_b = self._compute_gradients(x_train, y_train, pred)
            self._update_model_parameters(gradient_w, gradient_b, learning_rate)

            pred_to_class = np.array([1 if p > 0.5 else 0 for p in pred])
            accuracy = accuracy_score(y_train, pred_to_class)
            loss = self._compute_loss(y_train, pred_to_class)
            loss_change = abs(loss - self.train_losses[-1]) if len(self.train_losses) > 0 else None
            gradient_magnitude = self.compute_gradient_magnitude(gradient_w, gradient_b)

            self.logger.debug(f"{i}: {accuracy=} {loss=} {loss_change=} {gradient_magnitude=}")

            self.train_accuracies.append(accuracy)
            self.train_losses.append(loss)

            self._train_iterations = i + 1
            if gradient_magnitude < tolerance:
                break
            # if loss_change is not None and loss_change < tolerance:
            #     break

        final_accuracy = self.train_accuracies[-1] if len(self.train_accuracies) > 0 else None
        self.logger.info(f"Model fitting complete. {final_accuracy=} number of iterations={self._train_iterations}")

    def predict(self, x: np.ndarray) -> list:
        """
        Perform binary classification for the given input

        :param x: The input to make predictions off
        :return: The predicted classes - either 0 or 1
        """
        self.logger.info("Starting predictions...")

        if self.weights is None or self.bias is None:
            raise Exception(
                "Please fit the model to some training data first, before making predictions."
            )

        x_dot_weights = np.matmul(x, self.weights.transpose()) + self.bias
        probabilities = self._apply_logistic_function(x_dot_weights)

        self.logger.info("Finished predictions.")

        return [1 if p > 0.5 else 0 for p in probabilities]

    def _apply_logistic_function(self, x: np.ndarray) -> np.ndarray:
        """
        Apples the sigmoid function to the given set of inputs.
        The outputs represent the probability of belonging to class 1.

        :param x: Inputs to the sigmoid function
        :return The corresponding sigmoid function y-values
        """
        return np.array([self._sigmoid(value) for value in x])

    def _sigmoid(self, x: float) -> float:
        """
        Apples the sigmoid function to the given input.

        The use of if-else and the two expressions of the sigmoid function
        ensure that there is no numerical buffer overflow.

        :param x: Input to the sigmoid function
        :return The corresponding sigmoid function value
        """
        if x >= 0:
            z = np.exp(-x)
            return 1 / (1 + z)
        else:
            z = np.exp(x)
            return z / (1 + z)

    def _compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Computes the overall loss from the predicted and target values.
        Here, the average binary cross entropy loss is used.

        :param y_true: The list of target values
        :param y_pred: The list of predicted values
        :return: The calculated average binary cross entropy loss
        """

        y_zero_loss = y_true * np.log(y_pred + 1e-9)
        y_one_loss = (1 - y_true) * np.log(1 - y_pred + 1e-9)

        return -np.mean(y_zero_loss + y_one_loss)

    def _compute_gradients(
        self, x: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """
        Calculate the partial derivatives of the loss function.
        Here, the gradients for the weights and bias represent the partial derivatives of the
        binary cross entropy loss function.

        :param x: The training input
        :param y_true: The list of target values
        :param y_pred: The predicted values
        :return: The gradient values for the weights and the bias.
            For the weights, this is a set of gradients, one for each weight.
        """

        difference = y_pred - y_true
        gradient_b = np.mean(difference)
        gradients_w = np.matmul(x.transpose(), difference)
        gradients_w = np.array([np.mean(grad) for grad in gradients_w])

        return gradients_w, gradient_b

    def compute_gradient_magnitude(self, gradients_w: np.ndarray, gradient_b: float) -> float:
        """

        :param gradients_w:
        :param gradient_b:
        :return:
        """
        partial_derivatives = np.append(gradients_w, gradient_b)

        # Compute the magnitude (L2 norm) of the gradient vector
        gradient_magnitude = np.linalg.norm(partial_derivatives)

        return gradient_magnitude

    def _update_model_parameters(
        self, gradient_w, gradient_b, learning_rate: float
    ) -> None:
        """
        Calculate the model's new weights and bias as a result of the given gradients for
        the parameters.

        :param gradient_w: The gradients for the weights
        :param gradient_b: The gradient for the bias
        :return: N/A
        """
        # self.weights = self.weights - gradient_w
        # self.bias = self.bias - gradient_b
        self.weights = self.weights - learning_rate * gradient_w
        self.bias = self.bias - learning_rate * gradient_b
