import numpy as np
import torch
from typing import Self, Tuple

import posnnet.data.scalers


class TrajectoryErrorComputer:

    def __init__(
        self: Self,
        scalers: posnnet.data.scalers.Scalers,
        coeff_frequency_division: int,
        frequency: int
    ) -> None:

        """
        Initialize a TrajectoryErrorComputer instance that manages the computation of the trajectory error.

        Args:
            - scalers (posnnet.data.scalers.Scalers): The scalers used to scale / unscale the data.
            - coeff_frequency_division (int): The coefficient of frequency reduction used by the framework.
            - frequency (int): The original frequency of the sensor.
        """

        self.scalers = scalers
        self.coeff_frequency_division = coeff_frequency_division
        self.frequency = frequency

    def __generate_weight_for_beginning_mask(
        self: Self,
        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:

        """
        Generate the weight for the forward and backward pass merge for the beginning case.

        Args:
            - mask (np.ndarray): The array that contains the mask (GPS outage simulation).

        Returns:
            - forward_weight (np.ndarray): The weight for the forward pass.
            - backward_weight (np.ndarray): The weight for the backward pass.
        """

        len_window, n_velocity_axis = mask.shape
        size_mask = np.sum(mask[:, 0])

        forward_weight = np.zeros(shape=(len_window, n_velocity_axis))
        backward_weight = np.concatenate([
            np.ones(shape=(size_mask, n_velocity_axis)),
            np.zeros(shape=(len_window - size_mask, n_velocity_axis))
        ], axis=0)

        return forward_weight, backward_weight

    def __generate_weight_for_end_mask(
        self: Self,
        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:

        """
        Generate the weight for the forward and backward pass merge for the end case.

        Args:
            - mask (np.ndarray): The array that contains the mask (GPS outage simulation).

        Returns:
            - forward_weight (np.ndarray): The weight for the forward pass.
            - backward_weight (np.ndarray): The weight for the backward pass.
        """

        len_window, n_velocity_axis = mask.shape
        size_mask = np.sum(mask[:, 0])

        forward_weight = np.concatenate([
            np.zeros(shape=(len_window - size_mask, n_velocity_axis)),
            np.ones(shape=(size_mask, n_velocity_axis))
        ], axis=0)
        backward_weight = np.zeros(shape=(len_window, n_velocity_axis))

        return forward_weight, backward_weight

    def __generate_weight_for_within_session_mask(
        self: Self,
        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:

        """
        Generate the weight for the forward and backward pass merge for the within session case.

        Args:
            - mask (np.ndarray): The array that contains the mask (GPS outage simulation).

        Returns:
            - forward_weight (np.ndarray): The weight for the forward pass.
            - backward_weight (np.ndarray): The weight for the backward pass.
        """

        len_window, n_velocity_axis = mask.shape
        size_mask = sum(mask[:, 0])
        size_left = np.argmax(mask[:, 0])
        size_right = len_window - size_mask - size_left

        forward_weight = np.concatenate([
            np.zeros(shape=(size_left, n_velocity_axis)),
            np.linspace(
                start=[1] * n_velocity_axis,
                stop=[0] * n_velocity_axis,
                num=size_mask
            ),
            np.zeros(shape=(size_right, n_velocity_axis)),
        ])
        backward_weight = np.concatenate([
            np.zeros(shape=(size_left, n_velocity_axis)),
            np.linspace(
                start=[0] * n_velocity_axis,
                stop=[1] * n_velocity_axis,
                num=size_mask
            ),
            np.zeros(shape=(size_right, n_velocity_axis)),
        ])

        return forward_weight, backward_weight

    def __generate_weight_for_mask(
        self: Self,
        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:

        """
        Generate the weight for the forward and backward pass merge.

        Args:
            - mask (np.ndarray): The array that contains the mask (GPS outage simulation).

        Returns:
            - forward_weight (np.ndarray): The weight for the forward pass.
            - backward_weight (np.ndarray): The weight for the backward pass.
        """

        if mask[0, 0] == True:
            forward_weight, backward_weight = self.__generate_weight_for_beginning_mask(mask=mask)
        elif mask[-1, 0] == True:
            forward_weight, backward_weight = self.__generate_weight_for_end_mask(mask=mask)
        else:
            forward_weight, backward_weight = self.__generate_weight_for_within_session_mask(mask=mask)

        return forward_weight, backward_weight

    def compute(
        self: Self,
        y_velocity_fusion_unscaled_undivided: torch.Tensor,
        y_velocity_pred: torch.Tensor,
        masks: torch.Tensor
    ) -> float:

        """
        Compute the trajectory error.

        Args:
            - y_velocity_fusion_unscaled_undivided (torch.Tensor): The tensor that contains the ground truth unscaled velocity with the original frequency.
            - y_velocity_pred (torch.Tensor): The tensor that contains the predicted velocity.
            - masks (torch.Tensor): The tensor that contains the masks (GPS outages simulation).

        Returns:
            - trajectory_error (float): The sum of the trajectory error over all the sample of the batch.
        """

        # Convert torch tensors to numpy arrays
        y_velocity_fusion_unscaled_undivided = y_velocity_fusion_unscaled_undivided.numpy()
        y_velocity_pred = y_velocity_pred.detach().cpu().numpy()
        masks = masks.cpu().numpy().astype(dtype=np.bool)

        # Unscale predictions (predictions are scale using a MinMaxScaler between -1 and 1).
        # Sklearn scalers handle 2 dimensions arrays, so the array is unscaled batch element by batch element.
        y_velocity_pred = np.stack([
            self.scalers.unscale(data=y_velocity_pred[i_batch], scaling_type="normalization", source_name="velocity_fusion")
            for i_batch in range(len(y_velocity_pred))
        ], axis=0)

        # Restore original frequency in predicted velocity and masks (frequency is divided and the network predicts the average over the longer period).
        y_velocity_pred = np.repeat(a=y_velocity_pred, repeats=self.coeff_frequency_division, axis=1)
        masks = np.repeat(a=masks, repeats=self.coeff_frequency_division, axis=1)

        # Set unmasked velocity to 0 so they do not impact the trajectory error computation
        y_velocity_fusion_unscaled_undivided[~ masks] = 0
        y_velocity_pred[~ masks] = 0

        # Add a null velocity at the beginning of the arrays
        batch_size, _, n_velocity_axis = y_velocity_pred.shape
        y_velocity_fusion_unscaled_undivided = np.concatenate([np.zeros(shape=(batch_size, 1, n_velocity_axis)), y_velocity_fusion_unscaled_undivided], axis=1)
        y_velocity_pred = np.concatenate([np.zeros(shape=(batch_size, 1, n_velocity_axis)), y_velocity_pred], axis=1)

        # Compute the forward pass for ground truth and prediction.
        # Forward pass = cumulative sum of the velocity.
        # NB : The timedelta is applied after for computation efficiency.
        y_forward = np.cumsum(y_velocity_fusion_unscaled_undivided, axis=1)[:, :-1, :] # Do not take the last one
        y_forward_pred = np.cumsum(y_velocity_pred, axis=1)[:, :-1, :] # Do not take the last one

        # Compute the backward pass for ground truth and prediction.
        # Backward pass = cumulative sum of the reversed negative array.
        # NB : The timedelta is applied after for computation efficiency.
        y_backward = np.flip(np.cumsum(np.flip(- y_velocity_fusion_unscaled_undivided, axis=1), axis=1), axis=1)[:, 1:, :] # Do not take the first one
        y_backward_pred = np.flip(np.cumsum(np.flip(- y_velocity_pred, axis=1), axis=1), axis=1)[:, 1:, :] # Do not take the first one

        # Construct the weight for forward and backward.
        forward_weights, backward_weights = zip( * (self.__generate_weight_for_mask(mask=mask) for mask in masks))

        # Construct ground truth and predicted positions
        y_position = y_forward * forward_weights + y_backward * backward_weights
        y_position_pred = y_forward_pred * forward_weights + y_backward_pred * backward_weights

        # Compute the euclidean distance
        position_error = y_position - y_position_pred
        euclidean_distance = np.sqrt(np.sum(position_error ** 2, axis=2))

        # Sum over the sequences and scale with the timedelta
        trajectory_error = np.sum(euclidean_distance)
        trajectory_error = (1 / self.frequency) * trajectory_error
        trajectory_error = float(trajectory_error)

        return trajectory_error