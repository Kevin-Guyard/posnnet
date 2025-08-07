import itertools
import numpy as np
import torch
from typing import TypeVar, Type, Dict, Tuple, List, Union

import posnnet.data.evaluation_dataset
import posnnet.data.scalers
import posnnet.models


class Evaluator:

    @classmethod
    def __generate_weight_for_beginning_mask(
        cls: Type[TypeVar("Evaluator")],
        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:

        """
        Generate the weight for the forward and backward pass merge for the beginning case.

        Args:
            - mask (np.ndarray): The array that contains the mask (GPS outage simulation). Shape = (len_window, n_velocity_axis).

        Returns:
            - forward_weight (np.ndarray): The weight for the forward pass. Shape = (len_window, n_velocity_axis).
            - backward_weight (np.ndarray): The weight for the backward pass. Shape = (len_window, n_velocity_axis).
        """

        len_window, n_velocity_axis = mask.shape
        size_mask = np.sum(mask[:, 0])

        forward_weight = np.zeros(shape=(len_window, n_velocity_axis))
        backward_weight = np.concatenate([
            np.ones(shape=(size_mask, n_velocity_axis)),
            np.zeros(shape=(len_window - size_mask, n_velocity_axis))
        ], axis=0)

        return forward_weight, backward_weight

    @classmethod
    def __generate_weight_for_end_mask(
        cls: Type[TypeVar("Evaluator")],
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:

        """
        Generate the weight for the forward and backward pass merge for the end case.

        Args:
            - mask (np.ndarray): The array that contains the mask (GPS outage simulation). Shape = (len_window, n_velocity_axis).

        Returns:
            - forward_weight (np.ndarray): The weight for the forward pass. Shape = (len_window, n_velocity_axis).
            - backward_weight (np.ndarray): The weight for the backward pass. Shape = (len_window, n_velocity_axis).
        """

        len_window, n_velocity_axis = mask.shape
        size_mask = np.sum(mask[:, 0])

        forward_weight = np.concatenate([
            np.zeros(shape=(len_window - size_mask, n_velocity_axis)),
            np.ones(shape=(size_mask, n_velocity_axis))
        ], axis=0)
        backward_weight = np.zeros(shape=(len_window, n_velocity_axis))

        return forward_weight, backward_weight

    @classmethod
    def __generate_weight_for_within_session_mask(
        cls: Type[TypeVar("Evaluator")],
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:

        """
        Generate the weight for the forward and backward pass merge for the within session case.

        Args:
            - mask (np.ndarray): The array that contains the mask (GPS outage simulation). Shape = (len_window, n_velocity_axis).

        Returns:
            - forward_weight (np.ndarray): The weight for the forward pass. Shape = (len_window, n_velocity_axis).
            - backward_weight (np.ndarray): The weight for the backward pass. Shape = (len_window, n_velocity_axis).
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

    @classmethod
    def __generate_weight_for_mask(
        cls: Type[TypeVar("Evaluator")],
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:

        """
        Generate the weight for the forward and backward pass merge.

        Args:
            - mask (np.ndarray): The array that contains the mask (GPS outage simulation). Shape = (len_window, n_velocity_axis).

        Returns:
            - forward_weight (np.ndarray): The weight for the forward pass. Shape = (len_window, n_velocity_axis).
            - backward_weight (np.ndarray): The weight for the backward pass. Shape = (len_window, n_velocity_axis).
        """

        if mask[0, 0] == True:
            forward_weight, backward_weight = cls.__generate_weight_for_beginning_mask(mask=mask)
        elif mask[-1, 0] == True:
            forward_weight, backward_weight = cls.__generate_weight_for_end_mask(mask=mask)
        else:
            forward_weight, backward_weight = cls.__generate_weight_for_within_session_mask(mask=mask)

        return forward_weight, backward_weight

    @classmethod
    def __unscale_and_restore_frequency(
        cls: Type[TypeVar("Evaluator")],
        scalers: posnnet.data.scalers.Scalers,
        coeff_frequency_division: int,
        y_velocity_pred: np.ndarray,
        mask: Union[np.ndarray, None]
    ) -> Tuple[np.ndarray, Union[np.ndarray, None]]:

        """
        Unscale the predicted velocities and restore the original frequency for the predicted velocities and masks.

        Args:
            - scalers (posnnet.data.scalers.Scalers): The project Scalers instance.
            - coeff_frequency_division (int): The factor of frequency division.
            - y_velocity_pred (np.ndarray): The array that contains the predicted velocities. Shape = (len_seq, n_velocity_axis).
            - mask (Union[np.ndarray, None]): The array that contains the mask. Shape = (len_seq, n_velocity_axis).

        Returns:
            - y_velocity_pred (np.ndarray): The predicted velocities unscaled and with original frequency. Shape = (len_window, n_velocity_axis).
            - mask (Union[np.ndarray, None]): The mask with original frequency. Shape = (len_window, n_velocity_axis).
        """

        # Unscale predictions (predictions are scale using a MinMaxScaler between -1 and 1).
        # Sklearn scalers handle 2 dimensions arrays, so the array is unscaled batch element by batch element.
        y_velocity_pred = scalers.unscale(data=y_velocity_pred, scaling_type="normalization", source_name="velocity_fusion")

        # Restore original frequency in predicted velocity and masks (frequency is divided and the network predicts the average over the longer period).
        y_velocity_pred = np.repeat(a=y_velocity_pred, repeats=coeff_frequency_division, axis=0)
        if mask is not None:
            mask = np.repeat(a=mask, repeats=coeff_frequency_division, axis=0)

        return y_velocity_pred, mask

    @classmethod
    def __compute_predicted_positions(
        cls: Type[TypeVar("Evaluator")],
        frequency: int,
        y_position_target: np.ndarray,
        y_velocity_pred: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:

        """
        Compute the predicted positions.

        Args:
            - frequency (int): The original frequency of the data.
            - y_position_target (np.ndarray): The target (ground truth) position. Shape = (len_window, n_velocity_axis).
            - y_velocity_pred (np.ndarray): The predicted velocities unscaled and with the original frequency. Shape = (len_window, n_velocity_axis).
            - mask (np.ndarray): The mask (GPS outages simulation) with the orginal frequency. Shape = (len_window, n_velocity_axis).

        Returns:
            - position_pred (np.ndarray): The predicted positions. Shape = (len_window, n_velocity_axis).
        """

        # Collect the first and last index of the mask.
        first_idx_mask = np.flatnonzero(mask[:, 0])[0]
        last_idx_mask = np.flatnonzero(mask[:, 0])[-1]

        # Collect the length of the window, the number of velocity axis and the size of the masked part.
        len_window, n_velocity_axis = mask.shape
        size_mask = sum(mask[:, 0])

        # Concatenate velocities from last instant before GPS outage to last instant before the end of the GPS outage.
        if first_idx_mask > 0:
            y_velocity_pred_forward = np.concatenate([
                np.zeros(shape=(first_idx_mask, n_velocity_axis)),
                y_velocity_pred[first_idx_mask - 1 : last_idx_mask, :],
                np.zeros(shape=(len_window - last_idx_mask - 1, n_velocity_axis))
            ], axis=0)
        else:
            y_velocity_pred_forward = np.zeros(shape=(len_window, n_velocity_axis))

        # Collect the last position known before GPS outage.
        if first_idx_mask > 0:
            initial_position = y_position_target[first_idx_mask - 1, :]
        else:
            initial_position = np.zeros(shape=(1, n_velocity_axis))

        # Compute forward position.
        y_position_pred_forward = initial_position + (1 / frequency) * np.cumsum(y_velocity_pred_forward, axis=0)

        # Concatenate velocities during GPS outage.
        if last_idx_mask < len_window - 1:
            y_velocity_pred_backward = np.concatenate([
                np.zeros(shape=(first_idx_mask, n_velocity_axis)),
                y_velocity_pred[first_idx_mask : last_idx_mask + 1, :],
                np.zeros(shape=(len_window - last_idx_mask - 1, n_velocity_axis))
            ], axis=0)
        else:
            y_velocity_pred_backward = np.zeros(shape=(len_window, n_velocity_axis))

        # Collect the first position known after GPS outage.
        if last_idx_mask < len_window - 1:
            final_position = y_position_target[last_idx_mask + 1, :]
        else:
            final_position = np.zeros(shape=(1, n_velocity_axis))
            
        # Compute backward position.
        y_position_pred_backward = final_position - (1 / frequency) * np.flip(m=np.cumsum(a=np.flip(m=y_velocity_pred_backward, axis=0), axis=0), axis=0)

        # Construct the weight for forward and backward.
        forward_weight, backward_weight = cls.__generate_weight_for_mask(mask=mask)

        # Compute predicted position.
        y_position_pred = y_position_pred_forward * forward_weight + y_position_pred_backward * backward_weight

        # Fill predicted positions with known positions (target) on unmasked part.
        y_position_pred[~ mask] = y_position_target[~ mask]
        
        return y_position_pred

    @classmethod
    def __compute_velocity_metrics(
        cls: Type[TypeVar("Evaluator")],
        y_velocity_pred: np.ndarray,
        y_velocity_target: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[float, float, float]:

        """
        Compute the velocity related metrics.

        Args:
            - y_velocity_pred (np.ndarray): The predicted velocities (unscaled and with original frequency). Shape = (len_seq, n_velocity_axis).
            - y_velocity_target (np.ndarray): The target (ground truth) velocities. Shape = (len_seq, n_velocity_axis).
            - mask (np.ndarray): The mask (with original frequency). Shape = (len_seq, n_velocity_axis).

        Returns:
            - average_velocity_error (float): The average velocity error.
            - relative_maximum_velocity_error (float): The relative maximum velocity error.
            - std_velocity_error (float): The standard deviation of the velocity error.
        """

        # Compute per-axis residuals.
        velocity_residuals = y_velocity_target - y_velocity_pred
        # Square each component.
        squared_velocity_residuals = velocity_residuals ** 2
        # Sum over the coordinate axis to get the squared euclidean errors.
        squared_velocity_errors = squared_velocity_residuals.sum(axis=-1)
        # Square root to get the euclidean (l2) error.
        velocity_errors = np.sqrt(squared_velocity_errors)

        # Compute the average velocity error on masked part.
        average_velocity_error = float(velocity_errors[mask[:, 0]].mean())

        # Compute the maximum velocity error on masked part.
        maximum_velocity_error = float(velocity_errors[mask[:, 0]].max())
        # Compute the relative maximum velocity error (relative to the average velocity error).
        relative_maximum_velocity_error = maximum_velocity_error / average_velocity_error

        # Compute the standard deviation of the velocity errors on masked part.
        std_velocity_error = float(velocity_errors[mask[:, 0]].std())

        return average_velocity_error, relative_maximum_velocity_error, std_velocity_error

    @classmethod
    def __compute_position_metrics(
        cls: Type[TypeVar("Evaluator")],
        y_position_pred: np.ndarray,
        y_position_target: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[float, float, float]:

        """
        Compute the position related metrics.

        Args:
            - y_position_pred (np.ndarray): The predicted velocities (unscaled and with original frequency). Shape = (len_seq, n_velocity_axis).
            - y_position_target (np.ndarray): The target (ground truth) velocities. Shape = (len_seq, n_velocity_axis).
            - mask (np.ndarray): The mask (with original frequency). Shape = (len_seq, n_velocity_axis).

        Returns:
            - average_trajectory_error (float): The average trajectory error.
            - relative_maximum_trajectory_error (float): The relative maximum trajectory error.
            - std_trajectory_error (float): The standard deviation of the trajectory error.
        """

        # Compute per-axis residuals.
        trajectory_residuals = y_position_target - y_position_pred
        # Square each component.
        squared_trajectory_residuals = trajectory_residuals ** 2
        # Sum over the coordinate axis to get the squared euclidean errors.
        squared_trajectory_errors = squared_trajectory_residuals.sum(axis=-1)
        # Square root to get the euclidean (l2) error.
        trajectory_errors = np.sqrt(squared_trajectory_errors)

        # Compute the average position error on masked part.
        average_trajectory_error = float(trajectory_errors[mask[:, 0]].mean())

        # Compute the maximum position error on masked part.
        maximum_trajectory_error = float(trajectory_errors[mask[:, 0]].max())
        # Compute the relative maximum position error (relative to the average position error).
        relative_maximum_trajectory_error = maximum_trajectory_error / average_trajectory_error

        # Compute the standard deviation of the position errors on masked part.
        std_trajectory_error = float(trajectory_errors[mask[:, 0]].std())

        return average_trajectory_error, relative_maximum_trajectory_error, std_trajectory_error

    @classmethod
    def __compute_distance_metrics(
        cls: Type[TypeVar("Evaluator")],
        y_position_pred: np.ndarray,
        y_position_target: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[float]:

        """
        Compute the distance related metrics.

        Args:
            - y_position_pred (np.ndarray): The predicted velocities (unscaled and with original frequency). Shape = (len_seq, n_velocity_axis).
            - y_position_target (np.ndarray): The target (ground truth) velocities. Shape = (len_seq, n_velocity_axis).
            - mask (np.ndarray): The mask (with original frequency). Shape = (len_seq, n_velocity_axis).

        Returns:
            - relative_distance_error (float): The relative distance error.
        """

        # Compute per axis positions difference.
        position_diffs = np.diff(y_position_target, axis=0)
        position_pred_diffs = np.diff(y_position_pred, axis=0)
        # Square each component.
        squared_position_diff = position_diffs ** 2
        squared_position_pred_diff = position_pred_diffs ** 2
        # Sum over the coordinate axis to get the squared euclidean distance between every timestamp.
        squared_distances = squared_position_diff.sum(axis=-1)
        squared_distances_pred = squared_position_pred_diff.sum(axis=-1)
        # Square root to get the euclidean (L2) distances.
        distances = np.sqrt(squared_distances)
        distances_pred = np.sqrt(squared_distances_pred)

        # Compute relative distance error.      
        relative_distance_error = float(np.abs(np.sum(distances_pred[mask[1:, 0]]) / np.sum(distances[mask[1:, 0]]) - 1))

        return relative_distance_error

    @classmethod
    def __compute_transformed_position_metrics(
        cls: Type[TypeVar("Evaluator")],
        y_position_pred: np.ndarray,
        y_position_target: np.ndarray,
        mask: np.ndarray,
        average_trajectory_error: float
    ) -> Tuple[float, float, float, float]:

        """
        Compute position transformation related metrics.

        Args:
            - y_position_pred (np.ndarray): The predicted velocities (unscaled and with original frequency). Shape = (len_seq, n_velocity_axis).
            - y_position_target (np.ndarray): The target (ground truth) velocities. Shape = (len_seq, n_velocity_axis).
            - mask (np.ndarray): The mask (with original frequency). Shape = (len_seq, n_velocity_axis).
            - average_trajectory_error (float): The Average Trajectory Error (ATE) before transformation.

        Returnre
        
            - relative_form_transformed_trajectory_error (float): The ratio of ATE remaining after the transformations.
            - scale_error (float): The scaling error of the prediction.
            - translation_error (float): The translation error of the prediction.
            - rotation_error (float): The rotation error of the prediction.
        """

        # Keep only the part during GPS outage.
        prediction = y_position_pred[mask[:, 0], :]
        target = y_position_target[mask[:, 0], :]

        # Get the length of the GPS outage.
        len_gps_outage = len(prediction)

        # Collect the centroid for both prediction and target, shape = (n_velocity_axis, )
        centroid_prediction = prediction.mean(axis=0)
        centroid_target = target.mean(axis=0)

        # Center both prediction and target.
        prediction_centered = prediction - centroid_prediction
        target_centered = target - centroid_target

        # Compute SVD.
        sigma = (target_centered.T @ prediction_centered) / len_gps_outage
        U_left, sing_vals, Vt_right = np.linalg.svd(sigma)

        # Correction in case of reflexion.
        S = np.eye(U_left.shape[0])
        if np.linalg.det(U_left) * np.linalg.det(Vt_right) < 0:
            S[-1, -1] = -1

        # Compute the rotation matrix, shape = (n_velocity_axis, n_velocity_axis).
        rotation = U_left @ S @ Vt_right

        # Compute the scaling coefficient.
        var_pred = np.sum(prediction_centered ** 2) / len_gps_outage
        scale = np.trace(np.diag(sing_vals) @ S) / var_pred

        # Compute translation vector.
        translation = centroid_target - scale * rotation @ centroid_prediction

        # Transform the prediction with the best transformation.
        prediction_transformed = (scale * (rotation @ prediction.T)).T + translation

        # Compute the Average Trajectory Error (ATE) of the prediction transformed with the best transformation.
        average_trajectory_error_transformed = np.mean(np.linalg.norm(prediction_transformed - target, axis=1))

        # Compute the ratio of ATE after transformation.
        relative_form_transformed_trajectory_error = average_trajectory_error_transformed / average_trajectory_error

        # Compute the error of translation (the euclidean distance between the prediction centroid and the target centroid).
        translation_error = np.linalg.norm(centroid_prediction - centroid_target)

        # Compute the rotation error.
        cosθ = 0.5 * (np.trace(rotation) - 1)
        cosθ = np.clip(cosθ, a_min=-1.0, a_max=1.0)
        rotation_error = np.degrees(np.arccos(cosθ))

        # Compute the scale error.
        scale_error = abs(scale - 1)

        return relative_form_transformed_trajectory_error, scale_error, translation_error, rotation_error

    @classmethod
    def __compute_metrics(
        cls: Type[TypeVar("Evaluator")],
        y_position_pred: np.ndarray,
        y_velocity_pred: np.ndarray,
        y_position_target: np.ndarray,
        y_velocity_target: np.ndarray,
        mask: np.ndarray
    ) -> Dict[str, np.ndarray]:

        """
        Compute the metrics on the batch elements.

        Args:
            - y_position_pred (np.ndarray): The predicted velocities (unscaled and with original frequency). Shape = (len_seq, n_velocity_axis).
            - y_velocity_pred (np.ndarray): The predicted velocities (unscaled and with original frequency). Shape = (len_seq, n_velocity_axis).
            - y_position_target (np.ndarray): The target (ground truth) velocities. Shape = (len_seq, n_velocity_axis).
            - y_velocity_target (np.ndarray): The target (ground truth) velocities. Shape = (len_seq, n_velocity_axis).
            - mask (np.ndarray): The mask (with original frequency). Shape = (len_seq, n_velocity_axis).

        Returns:
            - metrics (Dict[str, float]): A dictionnary with as key, the name of the metrics and as value, an array with the metric value.
        """

        average_velocity_error, relative_maximum_velocity_error, std_velocity_error = cls.__compute_velocity_metrics(
            y_velocity_pred=y_velocity_pred,
            y_velocity_target=y_velocity_target,
            mask=mask
        )

        average_trajectory_error, relative_maximum_trajectory_error, std_trajectory_error = cls.__compute_position_metrics(
            y_position_pred=y_position_pred,
            y_position_target=y_position_target,
            mask=mask
        )

        relative_distance_error = cls.__compute_distance_metrics(
            y_position_pred=y_position_pred,
            y_position_target=y_position_target,
            mask=mask
        )

        relative_form_transformed_trajectory_error, scale_error, translation_error, rotation_error = cls.__compute_transformed_position_metrics(
            y_position_pred=y_position_pred,
            y_position_target=y_position_target,
            mask=mask,
            average_trajectory_error=average_trajectory_error
        )

        metrics = {
            "average_velocity_error": average_velocity_error,
            "relative_maximum_velocity_error": relative_maximum_velocity_error,
            "std_velocity_error": std_velocity_error,
            "average_trajectory_error": average_trajectory_error,
            "relative_maximum_trajectory_error": relative_maximum_trajectory_error,
            "std_trajectory_error": std_trajectory_error,
            "relative_distance_error": relative_distance_error,
            "relative_form_transformed_trajectory_error": relative_form_transformed_trajectory_error,
            "scale_error": scale_error,
            "translation_error": translation_error,
            "rotation_error": rotation_error
        }

        return metrics

    @classmethod
    def __apply_mask(
        cls: Type[TypeVar("Evaluator")],
        mask: torch.Tensor,
        x_velocity_gps: torch.Tensor,
        x_velocity_fusion: torch.Tensor,
        x_orientation_fusion: torch.Tensor,
        scaling_type: str,
        relax_points: Dict[str, Dict[str, np.ndarray]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        """
        Mask the GPS and fusion data.

        Args:
            - mask (torch.Tensor): A boolean tensor that indicate the GPS outages of size = (1, len_seq, 1).
            - x_velocity_gps (torch.Tensor): A tensor that contains the GPS velocity data.
            - x_velocity_fusion (torch.Tensor): A tensor that contains the fusion velocity data.
            - x_orientation_fusion (torch.Tensor): A tensor that contains the fusion orientation data.
            - scaling_type (int): Either 'normalization' for a min-max scaling between -1 and 1 or 'standardization' for a standard scaling with zero mean and unit std.
            - relax_points (Dict[str, Dict[str, np.ndarray]]): The relax points (scaled) of the dataset (velocity GPS = 0 / velocity fusion = 0 / orientation fusion = 0).

        Returns:
            - x_velocity_gps (torch.Tensor): The GPS velocity data masked during the GPS outages.
            - x_velocity_fusion (torch.Tensor): The fusion velocity data masked during the GPS outages.
            - x_orientation_fusion (torch.Tensor): The fusion orientation data masked during the GPS outages.
        """

        # Mask GPS velocity data if the sensor provide at least GPS velocity on one axe.
        if x_velocity_gps.size(dim=-1) > 0:
            x_velocity_gps = torch.cat([
                x_velocity_gps[:, :, idx : idx + 1].masked_fill(mask=mask, value=relax_points[scaling_type]["velocity_gps"][idx])
                for idx in range(x_velocity_gps.size(dim=-1))
            ], dim=-1)

        # Mask fusion velocity data
        x_velocity_fusion = torch.cat([
            x_velocity_fusion[:, :, idx : idx + 1].masked_fill(mask=mask, value=relax_points[scaling_type]["velocity_fusion"][idx])
            for idx in range(x_velocity_fusion.size(dim=-1))
        ], dim=-1)

        # Mask fusion orientation data if the sensor provide at least fusion orientation on one axe.
        if x_orientation_fusion.size(dim=-1) > 0:
            x_orientation_fusion = torch.cat([
                x_orientation_fusion[:, :, idx : idx + 1].masked_fill(mask=mask, value=relax_points[scaling_type]["orientation_fusion"][idx])
                for idx in range(x_orientation_fusion.size(dim=-1))
            ], dim=-1)

        return x_velocity_gps, x_velocity_fusion, x_orientation_fusion

    @classmethod
    def evaluate_on_subcase(
        cls: Type[TypeVar("Evaluator")],
        model: posnnet.models.GeneralModel,
        dataset_eval: posnnet.data.evaluation_dataset.EvaluationDataset,
        scalers: posnnet.data.scalers.Scalers,
        coeff_frequency_division: int,
        frequency: int,
        scaling_type: str,
        relax_points: Dict[str, Dict[str, np.ndarray]],
        num_workers: int,
        device: torch.device,
        dtype: torch.dtype,
        verbosity: bool,
        subcase_name: str
    ) -> Dict[str, float]:

        """
        Perform the evaluation of the model on the subcase.

        Args:
            - model (posnnet.models.GeneralModel): The model for which perform the evaluation.
            - dataset_eval (posnnet.data.dataset.Dataset): The dataset that contains evaluation data.
            - scalers (posnnet.data.scalers.Scalers): The project Scalers instance.
            - coeff_frequency_division (int): The factor of frequency division.
            - frequency (int): The original frequency of the sensor.
            - scaling_type (int): Either 'normalization' for a min-max scaling between -1 and 1 or 'standardization' for a standard scaling with zero mean and unit std.
            - relax_points (Dict[str, Dict[str, np.ndarray]]): The relax points (scaled) of the dataset (velocity GPS = 0 / velocity fusion = 0 / orientation fusion = 0).
            - num_workers (int): The number of process that will instanciate a dataloader instance to load the data in parallel. 0 means that*
                                 the dataloader will be loaded in the main process. A rule of thumb is to use 4 processes by GPU.
            - device (torch.Tensor): The torch device used for training and inference.
            - dtype (torch.dtype): The dtype of the network and the data.
            - verbosity (bool): Indicate if the framework should print verbose to inform of the progress of the evaluation.
            - subcase_name (str): The name of the subcase (used only for verbose).

        Returns:
            - metrics (Dict[str, float]): A dictionnary that contains the metrics of the model on the evaluation dataset. Available metrics:
                                          average_velocity_error, relative_maximum_velocity_error, std_velocity_error, average_trajectory_error, 
                                          relative_maximum_trajectory_error, std_trajectory_error, relative_distance_error,
                                          relative_form_transformed_trajectory_error, scale_error, translation_error, rotation_error
        """

        # Transfer the model to the device, cast into the good dtype and set it in evaluation mode.
        model = model.to(device=device, dtype=dtype)
        model = model.eval()

        # Wrap the dataset into a dataloader.
        dataloader_eval = torch.utils.data.DataLoader(
            dataset=dataset_eval,
            batch_size=1,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=False,
            drop_last=False
        )

        metrics = {}

        # Iterate over the evaluation dataset.
        for i_batch, (x_accelerometer, x_gyroscope, x_magnetometer, x_velocity_gps, x_velocity_fusion, x_orientation_fusion, mask, y_position_target, y_velocity_target) in enumerate(dataloader_eval):

            # Apply mask during GPS outage.
            x_velocity_gps, x_velocity_fusion, x_orientation_fusion = cls.__apply_mask(
                mask=mask,
                x_velocity_gps=x_velocity_gps,
                x_velocity_fusion=x_velocity_fusion,
                x_orientation_fusion=x_orientation_fusion,
                scaling_type=scaling_type,
                relax_points=relax_points
            )

            # Create x mask from mask.
            x_mask = mask.clone().detach()

            # Transfer the data to the device.
            x_accelerometer = x_accelerometer.to(device=device)
            x_gyroscope = x_gyroscope.to(device=device)
            x_magnetometer = x_magnetometer.to(device=device)
            x_velocity_gps = x_velocity_gps.to(device=device)
            x_velocity_fusion = x_velocity_fusion.to(device=device)
            x_orientation_fusion = x_orientation_fusion.to(device=device)
            x_mask = x_mask.to(device=device)

            # Deactivate gradient computation (not needed) for performance purpose during forward pass.
            with torch.no_grad():

                y_velocity_pred = model(
                    x_accelerometer=x_accelerometer,
                    x_gyroscope=x_gyroscope,
                    x_magnetometer=x_magnetometer,
                    x_velocity_gps=x_velocity_gps,
                    x_velocity_fusion=x_velocity_fusion,
                    x_orientation_fusion=x_orientation_fusion,
                    x_mask=x_mask
                )

            # Convert torch tensor to numpy array on cpu.
            y_velocity_pred = y_velocity_pred.squeeze(dim=0).cpu().numpy()
            y_velocity_target = y_velocity_target.squeeze(dim=0).numpy()
            y_position_target = y_position_target.squeeze(dim=0).numpy()
            mask = mask.squeeze(dim=0).numpy()

            # Extend mask last dimension.
            mask = np.repeat(a=mask, repeats=y_velocity_target.shape[-1], axis=-1)

            # Unscaled predicted velocities (the network is trained to predict a scaled velocity between -1 anc 1)
            # and restore the original frequency (the network is trained to predicte 1/N velocity).
            y_velocity_pred, mask = cls.__unscale_and_restore_frequency(
                scalers=scalers,
                coeff_frequency_division=coeff_frequency_division,
                y_velocity_pred=y_velocity_pred,
                mask=mask
            ) # Shape (batch_size, len_seq, n_velocity_axis) ==> (batch_size, len_window, n_velocity_axis)

            # Replace known velocities inside predicted velocities.
            y_velocity_pred[~ mask] = y_velocity_target[~ mask]

            # Compute the predicted positions using known positions (before and after GPS outages) and predicted velocities during GPS outages.
            y_position_pred = cls.__compute_predicted_positions(
                frequency=frequency,
                y_position_target=y_position_target,
                y_velocity_pred=y_velocity_pred,
                mask=mask
            )

            batch_metrics = cls.__compute_metrics(
                y_position_pred=y_position_pred,
                y_velocity_pred=y_velocity_pred,
                y_position_target=y_position_target,
                y_velocity_target=y_velocity_target,
                mask=mask
            )

            for metric_name, metric_value in batch_metrics.items():
                metrics[metric_name] = metrics.get(metric_name, []) + [metric_value]

            if verbosity == True:
                print(
                    f"{subcase_name:s} progress: {100 * (i_batch + 1) / len(dataloader_eval):.2f}%", 
                    end="\r" if i_batch + 1 < len(dataloader_eval) else "\n"
                )

        for metric_name, metric_values in metrics.items():
            metrics[metric_name] = float(np.mean(metric_values))

        return metrics

    @classmethod
    def evaluate_averaging_on_subcase(
        cls: Type[TypeVar("Evaluator")],
        model_selection_levels: List[float],
        models: Dict[str, posnnet.models.GeneralModel],
        tuning_ates: Dict[str, float],
        dataset_eval_normalized: posnnet.data.evaluation_dataset.EvaluationDataset,
        dataset_eval_standardized: posnnet.data.evaluation_dataset.EvaluationDataset,
        scalers: posnnet.data.scalers.Scalers,
        coeffs_frequency_division: Dict[str, int],
        frequency: int,
        scaling_types: Dict[str, str],
        relax_points: Dict[str, Dict[str, np.ndarray]],
        device: torch.device,
        dtype: torch.dtype,
        verbosity: bool,
        subcase_name: str
    ) -> Dict[float, Dict[str, float]]:

        """
        Perform the evaluation of the averaging on the subcase.

        Args:
            - model_selection_levels (List[float]): The list of the level for the model selection (e.g. [0.33, 0.66, 1] for 33%, 66% and 100%).
            - models (Dict[str, posnnet.models.GeneralModel]): A dictionnary which contains the available models for this case (key = id_config, value = model).
            - tuning_ates (Dict[str, float]): A dictionnary which contains the ATE achieved by the models on the validation dataset during tuning (key = id_config, value = ATE).
            - dataset_eval_normalized (posnnet.data.dataset.Dataset): The dataset that contains evaluation data normalized (between -1 and 1).
            - dataset_eval_standardized (posnnet.data.dataset.Dataset): The dataset that contains evaluation data standardized (mean = 0, std = 1).
            - scalers (posnnet.data.scalers.Scalers): The project Scalers instance.
            - coeffs_frequency_division (Dict[str, int]): A dictionnary that contains the factor of frequency division associated to each model (key = id_config, value = coeff).
            - frequency (int): The original frequency of the sensor.
            - scaling_types (Dict[str, str]): A dictionnary that contains the scaling type for each model (key = id_config, value = scaling type).
            - relax_points (Dict[str, Dict[str, np.ndarray]]): The relax points (scaled) of the dataset (velocity GPS = 0 / velocity fusion = 0 / orientation fusion = 0).
            - device (torch.Tensor): The torch device used for training and inference.
            - dtype (torch.dtype): The dtype of the network and the data.
            - verbosity (bool): Indicate if the framework should print verbose to inform of the progress of the evaluation.
            - subcase_name (str): The name of the subcase (used only for verbose).

        Returns:
        - metrics (Dict[float, Dict[str, float]]): Dictionnaries that contains the metrics of the models on the evaluation dataset (one dictionnary for each model selection level).
                                                   Available metrics: average_velocity_error, relative_maximum_velocity_error, std_velocity_error, average_trajectory_error, 
                                                   relative_maximum_trajectory_error, std_trajectory_error, relative_distance_error,
                                                   relative_form_transformed_trajectory_error, scale_error, translation_error, rotation_error
        """

        # Prepare the list of selected models for each level.
        ates = sorted(list(tuning_ates.values()))
        ate_limits = [
            ates[max(int(model_selection_level * len(ates)) - 1, 0)] 
            for model_selection_level in model_selection_levels
        ]
        models_selections = [
            [
                id_config 
                for id_config, tuning_ate in tuning_ates.items() 
                if tuning_ate <= ate_limit
            ]
            for ate_limit in ate_limits
        ]

        # Create a unique list to skip models that are never selected.
        models_to_evaluate = list(set(itertools.chain.from_iterable(models_selections)))

        # Prepare a dict for metrics storage.
        metrics = {
            model_selection_level: {}
            for model_selection_level in model_selection_levels
        }

        # Transfer the model to the device, cast into the good dtype and set it in evaluation mode.
        models = {
            id_config: model.to(device=device, dtype=dtype).eval()
            for id_config, model in models.items()
        }

        # Prepare the list of coefficient of frequency division for each scaling type.
        coeffs_frequency_division_by_scaling_type = {
            "normalization": list(set([coeffs_frequency_division[id_config] for id_config in scaling_types.keys() if scaling_types[id_config] == "normalization"])), # Ensure uniqueness.
            "standardization": list(set([coeffs_frequency_division[id_config] for id_config in scaling_types.keys() if scaling_types[id_config] == "standardization"])), # Ensure uniqueness.
        }
        
        # Iterate over the evaluation dataset.
        for i_batch in range(dataset_eval_normalized.__len__()):

            # Preparate dict to store data (one tensor for each scaling type and coeff frequency division).
            accelerometer_data = {scaling: {} for scaling in coeffs_frequency_division_by_scaling_type.keys()}
            gyroscope_data = {scaling: {} for scaling in coeffs_frequency_division_by_scaling_type.keys()}
            magnetometer_data = {scaling: {} for scaling in coeffs_frequency_division_by_scaling_type.keys()}
            velocity_gps_data = {scaling: {} for scaling in coeffs_frequency_division_by_scaling_type.keys()}
            velocity_fusion_data = {scaling: {} for scaling in coeffs_frequency_division_by_scaling_type.keys()}
            orientation_fusion_data = {scaling: {} for scaling in coeffs_frequency_division_by_scaling_type.keys()}
            mask_data = {scaling: {} for scaling in coeffs_frequency_division_by_scaling_type.keys()}

            # Iterate over the scaling types used.
            for scaling_type in coeffs_frequency_division_by_scaling_type.keys():

                if scaling_type == "normalization":
                    dataset = dataset_eval_normalized
                else:
                    dataset = dataset_eval_standardized

                # Iterate over the coeff frequency division used for this scaling type
                for coeff_frequency_division in coeffs_frequency_division_by_scaling_type[scaling_type]:

                    # Set the dataset frequency division and collect data.
                    dataset.set_coeff_frequency_division(coeff_frequency_division=coeff_frequency_division)
                    batch_data = dataset.__getitem__(idx=i_batch)

                    accelerometer_data[scaling_type][coeff_frequency_division] = torch.from_numpy(batch_data[0]).unsqueeze(0) # (1, len_seq, 3)
                    gyroscope_data[scaling_type][coeff_frequency_division] = torch.from_numpy(batch_data[1]).unsqueeze(0) # (1, len_seq, 3)
                    magnetometer_data[scaling_type][coeff_frequency_division] = torch.from_numpy(batch_data[2]).unsqueeze(0) # (1, len_seq, 3)
                    velocity_gps_data[scaling_type][coeff_frequency_division] = torch.from_numpy(batch_data[3]).unsqueeze(0) # (1, len_seq, 3)
                    velocity_fusion_data[scaling_type][coeff_frequency_division] = torch.from_numpy(batch_data[4]).unsqueeze(0) # (1, len_seq, 3)
                    orientation_fusion_data[scaling_type][coeff_frequency_division] = torch.from_numpy(batch_data[5]).unsqueeze(0) # (1, len_seq, 3)
                    mask_data[scaling_type][coeff_frequency_division] = torch.from_numpy(batch_data[6]).unsqueeze(0) # (1, len_seq, 1)

                    # Apply mask during GPS outage.
                    (velocity_gps_data[scaling_type][coeff_frequency_division], 
                    velocity_fusion_data[scaling_type][coeff_frequency_division], 
                    orientation_fusion_data[scaling_type][coeff_frequency_division]) = cls.__apply_mask(
                        mask=mask_data[scaling_type][coeff_frequency_division],
                        x_velocity_gps=velocity_gps_data[scaling_type][coeff_frequency_division],
                        x_velocity_fusion=velocity_fusion_data[scaling_type][coeff_frequency_division],
                        x_orientation_fusion=orientation_fusion_data[scaling_type][coeff_frequency_division],
                        scaling_type=scaling_type,
                        relax_points=relax_points
                    )

            mask = np.copy(batch_data[6])
            y_position_target = batch_data[7]
            y_velocity_target = batch_data[8]
            
            # Extend mask last dimension.
            mask = np.repeat(a=mask, repeats=y_velocity_target.shape[-1], axis=-1)

            # Restore original frequency in mask.
            mask = np.repeat(a=mask, repeats=coeff_frequency_division, axis=0)

            # Transfer the data to the device.
            accelerometer_data = {
                scaling_type: {
                    coeff_frequency_division: tensor_data.to(device=device)
                    for coeff_frequency_division, tensor_data in accelerometer_data[scaling_type].items()
                }
                for scaling_type in accelerometer_data.keys()
            }
            gyroscope_data = {
                scaling_type: {
                    coeff_frequency_division: tensor_data.to(device=device)
                    for coeff_frequency_division, tensor_data in gyroscope_data[scaling_type].items()
                }
                for scaling_type in gyroscope_data.keys()
            }
            magnetometer_data = {
                scaling_type: {
                    coeff_frequency_division: tensor_data.to(device=device)
                    for coeff_frequency_division, tensor_data in magnetometer_data[scaling_type].items()
                }
                for scaling_type in magnetometer_data.keys()
            }
            velocity_gps_data = {
                scaling_type: {
                    coeff_frequency_division: tensor_data.to(device=device)
                    for coeff_frequency_division, tensor_data in velocity_gps_data[scaling_type].items()
                }
                for scaling_type in velocity_gps_data.keys()
            }
            velocity_fusion_data = {
                scaling_type: {
                    coeff_frequency_division: tensor_data.to(device=device)
                    for coeff_frequency_division, tensor_data in velocity_fusion_data[scaling_type].items()
                }
                for scaling_type in velocity_fusion_data.keys()
            }
            orientation_fusion_data = {
                scaling_type: {
                    coeff_frequency_division: tensor_data.to(device=device)
                    for coeff_frequency_division, tensor_data in orientation_fusion_data[scaling_type].items()
                }
                for scaling_type in orientation_fusion_data.keys()
            }
            mask_data = {
                scaling_type: {
                    coeff_frequency_division: tensor_data.to(device=device)
                    for coeff_frequency_division, tensor_data in mask_data[scaling_type].items()
                }
                for scaling_type in mask_data.keys()
            }

            # Initialize empty dict for predicted positions.
            ys_position_pred = {}

            # Iterate over available models.
            for id_config, model in models.items():

                if id_config not in models_to_evaluate:
                    continue

                # Select the data based on the scaling type and coefficient frequency division of the model.
                x_accelerometer = accelerometer_data[scaling_types[id_config]][coeffs_frequency_division[id_config]]
                x_gyroscope = gyroscope_data[scaling_types[id_config]][coeffs_frequency_division[id_config]]
                x_magnetometer = magnetometer_data[scaling_types[id_config]][coeffs_frequency_division[id_config]]
                x_velocity_gps = velocity_gps_data[scaling_types[id_config]][coeffs_frequency_division[id_config]]
                x_velocity_fusion = velocity_fusion_data[scaling_types[id_config]][coeffs_frequency_division[id_config]]
                x_orientation_fusion = orientation_fusion_data[scaling_types[id_config]][coeffs_frequency_division[id_config]]
                x_mask = mask_data[scaling_types[id_config]][coeffs_frequency_division[id_config]]

                # Deactivate gradient computation (not needed) for performance purpose during forward pass.
                with torch.no_grad():
                    
                    y_velocity_pred = model(
                        x_accelerometer=x_accelerometer,
                        x_gyroscope=x_gyroscope,
                        x_magnetometer=x_magnetometer,
                        x_velocity_gps=x_velocity_gps,
                        x_velocity_fusion=x_velocity_fusion,
                        x_orientation_fusion=x_orientation_fusion,
                        x_mask=x_mask
                    )

                # Convert torch tensor to numpy array on cpu.
                y_velocity_pred = y_velocity_pred.squeeze(dim=0).cpu().numpy()

                # Unscaled predicted velocities (the network is trained to predict a scaled velocity between -1 anc 1)
                # and restore the original frequency (the network is trained to predicte 1/N velocity).
                y_velocity_pred, _ = cls.__unscale_and_restore_frequency(
                    scalers=scalers,
                    coeff_frequency_division=coeffs_frequency_division[id_config],
                    y_velocity_pred=y_velocity_pred,
                    mask=None
                ) # Shape (batch_size, len_seq, n_velocity_axis) ==> (batch_size, len_window, n_velocity_axis)

                # Replace known velocities inside predicted velocities.
                y_velocity_pred[~ mask] = y_velocity_target[~ mask]

                # Compute the predicted positions using known positions (before and after GPS outages) and predicted velocities during GPS outages.
                ys_position_pred[id_config] = cls.__compute_predicted_positions(
                    frequency=frequency,
                    y_position_target=y_position_target,
                    y_velocity_pred=y_velocity_pred,
                    mask=mask
                )

            for model_selection_level, models_selection in zip(model_selection_levels, models_selections):

                # Stack the predicted position of the selected models.
                y_position_pred_stacked = np.stack([
                    y_position_pred
                    for id_config, y_position_pred in ys_position_pred.items()
                    if id_config in models_selection
                ], axis=-1)
                # Average over the selected models.
                y_position_pred = np.nanmean(y_position_pred_stacked, axis=-1)

                # Compute the final velocity (derivate of the position).
                y_velocity_pred = np.concatenate([
                    np.diff(y_position_pred, axis=0),
                    np.zeros(shape=(1, 3))
                ], axis=0) * frequency
                # Replace known velocities inside predicted velocities.
                y_velocity_pred[~ mask] = y_velocity_target[~ mask]

                # Compute metrics.
                batch_metrics = cls.__compute_metrics(
                    y_position_pred=y_position_pred,
                    y_velocity_pred=y_velocity_pred,
                    y_position_target=y_position_target,
                    y_velocity_target=y_velocity_target,
                    mask=mask
                )

                # Add the batch metrics to the dataset metrics.
                for metric_name, metric_value in batch_metrics.items():
                    metrics[model_selection_level][metric_name] = metrics[model_selection_level].get(metric_name, []) + [metric_value]

            if verbosity == True:
                print(
                    f"{subcase_name:s} progress: {100 * (i_batch + 1) / dataset_eval_normalized.__len__():.2f}%", 
                    end="\r" if i_batch + 1 < dataset_eval_normalized.__len__() else "\n"
                )

        # Average the metrics over the dataset
        for model_selection_level in model_selection_levels:
            for metric_name, metric_values in metrics[model_selection_level].items():
                metrics[model_selection_level][metric_name] = float(np.mean(metric_values))

        # Transfer the model to the cpu.
        models = {
            id_config: model.to(device=torch.device("cpu")).eval()
            for id_config, model in models.items()
        }

        return metrics