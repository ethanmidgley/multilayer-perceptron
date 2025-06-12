import numpy as np


class MSE:

    @staticmethod
    def calculate(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2, axis=-1)

    @staticmethod
    def deriv(y_true, yhat):

        samples = len(yhat)
        outputs = len(yhat[0])

        grad = -2 * (y_true - yhat) / outputs

        return grad / samples


class BinaryCrossEntropy:

    # Forward pass
    @staticmethod
    def calculate(y_true, y_pred):

        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Calculate sample-wise loss
        sample_losses = -(
            y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped)
        )
        sample_losses = np.mean(sample_losses, axis=-1)

        # Return losses
        return sample_losses

    # Backward pass
    @staticmethod
    def deriv(y_true, dvalues):

        # Number of samples
        samples = len(dvalues)
        # Number of outputs in every sample
        # We'll use the first sample to count them
        outputs = len(dvalues[0])

        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)

        # Calculate gradient
        grad = (
            -(y_true / clipped_dvalues - (1 - y_true) / (1 - clipped_dvalues)) / outputs
        )
        # Normalize gradient
        return grad / samples


class CategoricalCrossEntropy:

    @staticmethod
    def calculate(y_true, y_pred):

        samples = len(y_pred)

        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log = -np.log(correct_confidences)
        return negative_log

    @staticmethod
    def deriv(y_true, y_pred):

        samples = len(y_pred)

        labels = len(y_pred[0])

        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            y_true = np.eye(labels, dtype=int)[y_true]

        grad = -y_true / y_pred_clipped
        return grad / samples
