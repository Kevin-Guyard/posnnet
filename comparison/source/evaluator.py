import numpy as np
import pandas as pd
import pathlib
from typing import TypeVar, Type, Callable, Tuple, Dict


class Evaluator:

    @classmethod
    def evaluate_one_subcase(
        cls: Type[TypeVar("Evaluator")],
        model: Callable,
        subcase: str
    ) -> Dict[str, float]:

        """
        Evaluate the model on the subcase.

        Args:
            - model (Callable): The reconstructor with a reconstruct method.
            - subcase (str): The subcase on which score the reconstructor (either "00-30 to 02-00" or "02-00 to 05-00").

        Returns:
            - metrics (Dict[str, float]): A dictionnary that contains the metrics of the model on the evaluation dataset. Available metrics:
                                          average_velocity_error, relative_maximum_velocity_error, std_velocity_error, average_trajectory_error, 
                                          relative_maximum_trajectory_error, std_trajectory_error, relative_distance_error,
                                          relative_form_transformed_trajectory_error, scale_error, translation_error, rotation_error
        """

        metrics = {}

        for session_file in pathlib.Path(f"./project_framework_validation/data/preprocessed/evaluation/within/{subcase:s}").iterdir():

            if not "session" in session_file.name:
                continue

            df_eval = pd.read_pickle(filepath_or_buffer=session_file)
            n_gnss_outages = len([column for column in df_eval.columns if "gps_outage" in column])

            for i_gnss_outage in range(1, n_gnss_outages + 1):

                metrics_on_gnss_outage = cls.evaluate_one_outage(
                    model=model,
                    df_eval=df_eval,
                    i_gnss_outage=i_gnss_outage
                )

                for metric_name, metric_value in metrics_on_gnss_outage.items():
                    metrics[metric_name] = metrics.get(metric_name, []) + [metric_value]

        for metric_name, metric_values in metrics.items():
            metrics[metric_name] = float(np.mean(metric_values))

        return metrics

    @classmethod
    def evaluate_one_outage(
        cls: Type[TypeVar("Evaluator")],
        model: Callable,
        df_eval: pd.DataFrame,
        i_gnss_outage: int
    ) -> Dict[str, float]:

        """
        Evaluate the model on one GNSS outage.

        Args:
            - model (Callable): The reconstructor with a reconstruct method.
            - df_eval (pd.DataFrame): A dataframe that contains the data of the session on which the GNSS outage is simulated.
            - i_gnss_outage (int): The index of the simulated GNSS outage.

        Returns:
            - metrics (Dict[str, float]): A dictionnary that contains the metrics of the model on the evaluation dataset. Available metrics:
                                          average_velocity_error, relative_maximum_velocity_error, std_velocity_error, average_trajectory_error, 
                                          relative_maximum_trajectory_error, std_trajectory_error, relative_distance_error,
                                          relative_form_transformed_trajectory_error, scale_error, translation_error, rotation_error
        """

        first_idx_gps_outage = df_eval[df_eval[f"gps_outage_{i_gnss_outage}"] == 1].index[0]
        last_idx_gps_outage = df_eval[df_eval[f"gps_outage_{i_gnss_outage}"] == 1].index[-1]
        
        acc = df_eval.loc[first_idx_gps_outage : last_idx_gps_outage, ["accelerometer_x", "accelerometer_y", "accelerometer_z"]].to_numpy()
        gyro = df_eval.loc[first_idx_gps_outage : last_idx_gps_outage, ["gyroscope_x", "gyroscope_y", "gyroscope_z"]].to_numpy()
        velocity_0 = df_eval.loc[first_idx_gps_outage - 1, ["velocity_fusion_x", "velocity_fusion_y", "velocity_fusion_z"]].to_numpy()
        position_0 = df_eval.loc[first_idx_gps_outage - 1, ["position_fusion_x", "position_fusion_y", "position_fusion_z"]].to_numpy()
        roll_0 = df_eval.loc[first_idx_gps_outage - 1, "orientation_fusion_x"]
        pitch_0 = df_eval.loc[first_idx_gps_outage - 1, "orientation_fusion_y"]
        yaw_0 = df_eval.loc[first_idx_gps_outage - 1, "orientation_fusion_z"]

        velocity_pred, position_pred = model.reconstruct(
            acc=acc,
            gyro=gyro,
            velocity_0=velocity_0,
            position_0=position_0,
            roll_0=roll_0,
            pitch_0=pitch_0,
            yaw_0=yaw_0,
        )

        velocity_target = df_eval.loc[first_idx_gps_outage : last_idx_gps_outage, ["velocity_fusion_x", "velocity_fusion_y", "velocity_fusion_z"]].to_numpy()
        position_target = df_eval.loc[first_idx_gps_outage : last_idx_gps_outage, ["position_fusion_x", "position_fusion_y", "position_fusion_z"]].to_numpy()

        average_velocity_error, relative_maximum_velocity_error, std_velocity_error = cls.__compute_velocity_metrics(
            y_velocity_pred=velocity_pred,
            y_velocity_target=velocity_target
        )

        average_trajectory_error, relative_maximum_trajectory_error, std_trajectory_error = cls.__compute_position_metrics(
            y_position_pred=position_pred,
            y_position_target=position_target
        )

        relative_distance_error = cls.__compute_distance_metrics(
            y_position_pred=position_pred,
            y_position_target=position_target
        )

        relative_form_transformed_trajectory_error, scale_error, translation_error, rotation_error = cls.__compute_transformed_position_metrics(
            y_position_pred=position_pred,
            y_position_target=position_target,
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
    def __compute_velocity_metrics(
        cls: Type[TypeVar("Evaluator")],
        y_velocity_pred: np.ndarray,
        y_velocity_target: np.ndarray
    ) -> Tuple[float, float, float]:

        """
        Compute the velocity related metrics.

        Args:
            - y_velocity_pred (np.ndarray): The predicted velocities (unscaled and with original frequency). Shape = (len_seq, n_velocity_axis).
            - y_velocity_target (np.ndarray): The target (ground truth) velocities. Shape = (len_seq, n_velocity_axis).

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

        # Compute the average velocity error.
        average_velocity_error = float(velocity_errors.mean())

        # Compute the maximum velocity error.
        maximum_velocity_error = float(velocity_errors.max())
        # Compute the relative maximum velocity error (relative to the average velocity error).
        relative_maximum_velocity_error = maximum_velocity_error / average_velocity_error

        # Compute the standard deviation of the velocity errors.
        std_velocity_error = float(velocity_errors.std())

        return average_velocity_error, relative_maximum_velocity_error, std_velocity_error

    @classmethod
    def __compute_position_metrics(
        cls: Type[TypeVar("Evaluator")],
        y_position_pred: np.ndarray,
        y_position_target: np.ndarray,
    ) -> Tuple[float, float, float]:

        """
        Compute the position related metrics.

        Args:
            - y_position_pred (np.ndarray): The predicted velocities (unscaled and with original frequency). Shape = (len_seq, n_velocity_axis).
            - y_position_target (np.ndarray): The target (ground truth) velocities. Shape = (len_seq, n_velocity_axis).

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

        # Compute the average position error.
        average_trajectory_error = float(trajectory_errors.mean())

        # Compute the maximum position error.
        maximum_trajectory_error = float(trajectory_errors.max())
        # Compute the relative maximum position error (relative to the average position error).
        relative_maximum_trajectory_error = maximum_trajectory_error / average_trajectory_error

        # Compute the standard deviation of the position errors.
        std_trajectory_error = float(trajectory_errors.std())

        return average_trajectory_error, relative_maximum_trajectory_error, std_trajectory_error

    @classmethod
    def __compute_distance_metrics(
        cls: Type[TypeVar("Evaluator")],
        y_position_pred: np.ndarray,
        y_position_target: np.ndarray,
    ) -> Tuple[float]:

        """
        Compute the distance related metrics.

        Args:
            - y_position_pred (np.ndarray): The predicted velocities (unscaled and with original frequency). Shape = (len_seq, n_velocity_axis).
            - y_position_target (np.ndarray): The target (ground truth) velocities. Shape = (len_seq, n_velocity_axis).

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
        relative_distance_error = float(np.abs(np.sum(distances_pred) / np.sum(distances) - 1))

        return relative_distance_error

    @classmethod
    def __compute_transformed_position_metrics(
        cls: Type[TypeVar("Evaluator")],
        y_position_pred: np.ndarray,
        y_position_target: np.ndarray,
        average_trajectory_error: float
    ) -> Tuple[float, float, float, float]:

        """
        Compute position transformation related metrics.

        Args:
            - y_position_pred (np.ndarray): The predicted velocities (unscaled and with original frequency). Shape = (len_seq, n_velocity_axis).
            - y_position_target (np.ndarray): The target (ground truth) velocities. Shape = (len_seq, n_velocity_axis).
            - average_trajectory_error (float): The Average Trajectory Error (ATE) before transformation.

        Returnre
        
            - relative_form_transformed_trajectory_error (float): The ratio of ATE remaining after the transformations.
            - scale_error (float): The scaling error of the prediction.
            - translation_error (float): The translation error of the prediction.
            - rotation_error (float): The rotation error of the prediction.
        """

        # Keep only the part during GPS outage.
        prediction = y_position_pred
        target = y_position_target

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