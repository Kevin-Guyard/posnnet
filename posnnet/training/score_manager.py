import numpy as np
from typing import Self, Union, Tuple


class ScoreManager:

    def __init__(
        self: Self,
        patience: Union[int, None]
    ) -> None:

        """
        Initialize an instance of ScoreManager that handle the storage of the training and validation metrics, the verbosity 
        and the early stopping detection.

        Args:
            - patience (Union[int, None]): The number of epochs without validation results improvement before early stopping training.
                                           If None, training is never early stopped.
        """

        # Memorize parameters
        self.patience = patience

        # Initialize training metrics history
        self.history_loss_velocity_masked_train = []
        self.history_loss_velocity_unmasked_train = []
        self.history_average_trajectory_error_train = []

        # Initialize validation metrics history
        self.history_loss_velocity_masked_val = []
        self.history_loss_velocity_unmasked_val = []
        self.history_average_trajectory_error_val = []

    def push_training_metrics(
        self: Self,
        loss_velocity_masked_train: float,
        loss_velocity_unmasked_train: float,
        average_trajectory_error_train: float
    ) -> None:

        """
        Append the last training epoch metrics.

        Args:
            - loss_velocity_masked_train (float): The velocity loss on the masked part of the sequence during training.
            - loss_velocity_unmasked_train (float): The velocity loss on the unmasked part of the sequence during training.
            - average_trajectory_error_train (float): The average trajectory error on the masked part of the sequence during training.
        """

        # Transform undefined value into infinite (to keep mathematical properties such as inferior/superior to).
        if np.isnan(loss_velocity_masked_train):
            loss_velocity_masked_train = np.inf
        if np.isnan(loss_velocity_unmasked_train):
            loss_velocity_unmasked_train = np.inf
        if np.isnan(average_trajectory_error_train):
            average_trajectory_error_train = np.inf

        # Store metrics
        self.history_loss_velocity_masked_train.append(loss_velocity_masked_train)
        self.history_loss_velocity_unmasked_train.append(loss_velocity_unmasked_train)
        self.history_average_trajectory_error_train.append(average_trajectory_error_train)

    def push_validation_metrics(
        self: Self,
        loss_velocity_masked_val: float,
        loss_velocity_unmasked_val: float,
        average_trajectory_error_val: float
    ) -> None:

        """
        Append the last validation epoch metrics.

        Args:
            - loss_velocity_masked_val (float): The velocity loss on the masked part of the sequence during validation.
            - loss_velocity_unmasked_val (float): The velocity loss on the unmasked part of the sequence during validation.
            - average_trajectory_error_val (float): The average trajectory error on the masked part of the sequence during validation.
        """

        # Transform undefined value into infinite (to keep mathematical properties such as inferior/superior to).
        if np.isnan(loss_velocity_masked_val):
            loss_velocity_masked_val = np.inf
        if np.isnan(loss_velocity_unmasked_val):
            loss_velocity_unmasked_val = np.inf
        if np.isnan(average_trajectory_error_val):
            average_trajectory_error_val = np.inf

        # Store metrics
        self.history_loss_velocity_masked_val.append(loss_velocity_masked_val)
        self.history_loss_velocity_unmasked_val.append(loss_velocity_unmasked_val)
        self.history_average_trajectory_error_val.append(average_trajectory_error_val)

    def print_verbose(
        self: Self
    ) -> None:

        """
        Print the last epoch metrics of training and validation.
        """

        i_epoch = len(self.history_loss_velocity_masked_train) - 1

        verbose_message = f"Epoch {i_epoch + 1:03d} \n"
        verbose_message += f"Training  : masked = {self.history_loss_velocity_masked_train[-1]:.6f} / unmasked = {self.history_loss_velocity_unmasked_train[-1]:.6f} / ATE = {self.history_average_trajectory_error_train[-1]:.6f} \n"
        verbose_message += f"Validation: masked = {self.history_loss_velocity_masked_val[-1]:.6f} / unmasked = {self.history_loss_velocity_unmasked_val[-1]:.6f} / ATE = {self.history_average_trajectory_error_val[-1]:.6f} \n"
        verbose_message += "-" * 68
        
        print(verbose_message)

    def evaluate_early_stopping_criterion(
        self: Self
    ) -> bool:

        """
        Evaluate if the training should be early stopped. Training is early stopped if their is not any improvement in the
        velocity loss on the masked sequence during validation for x epochs, where x is the patience.

        Returns:
            - flag_early_stopping (bool): Indicate if the training should be early stopped
        """

        i_epoch = len(self.history_loss_velocity_masked_train) - 1

        # If patience is None, no early stopping
        if self.patience is None:
            flag_early_stopping = False
            
        # The number of performed epochs is inferior to the patience so the ealry stopping cannot be triggered yet.
        elif i_epoch < self.patience:
            flag_early_stopping = False
            
        # If their is an improvement in the velocity loss on masked part during validation (lower loss on the last epochs), do not trigger early stopping.
        elif self.history_loss_velocity_masked_val[- self.patience - 1] > np.min(self.history_loss_velocity_masked_val[- self.patience:]):
            flag_early_stopping = False
            
        # Otherwise, trigger early stopping.
        else:
            flag_early_stopping = True

        return flag_early_stopping

    def has_model_improved(
        self: Self
    ) -> bool:

        """
        Evaluate if the model has improved during last epoch (last velocity loss on masked part of the sequence during validation
        if lower than the previous lowest score).

        Returns:
            - flag_has_model_improved (bool): Indicate if the model has improved during the last epoch.
        """

        # If this is the first epoch, model is considered improved.
        if len(self.history_loss_velocity_masked_train) == 1:
            flag_has_model_improved = True
        # If the last score is inferior to the minimum score of all the previous epoch, model has improved.
        elif self.history_loss_velocity_masked_train[-1] < np.min(self.history_loss_velocity_masked_train[:-1]):
            flag_has_model_improved = True
        # Otherwise, the model has not improved.
        else:
            flag_has_model_improved = False

        return flag_has_model_improved

    def get_best_score(
        self: Self
    ) -> Tuple[float, float]:

        """
        Return the best score (best velocity loss on the masked part of the sequence during validation).

        Returns:
            - best_loss_achieved (float): The best (lowest) velocity loss achieved during validation.
            - associated_ate (float): The average trajectory error of the epoch when the model achieved the lowest velocity loss
        """

        best_loss_achieved = np.min(self.history_loss_velocity_masked_val) # Get the minimum loss
        associated_ate = self.history_average_trajectory_error_val[np.argmin(self.history_loss_velocity_masked_val)] # Get the ATE for the epoch with the lowest loss.

        return best_loss_achieved, associated_ate