import numpy as np
import pandas as pd
import pathlib
import torch
from typing import Self, List, Tuple, Dict

import posnnet.data.scalers


class EvaluationDataset(torch.utils.data.Dataset):

    def __init__(
        self: Self,
        len_window: int,
        coeff_frequency_division: int,
        scaling_type: str,
        scalers: posnnet.data.scalers.Scalers,
        sessions_id: List[int],
        features_name_accelerometer: List[str],
        features_name_gyroscope: List[str],
        features_name_magnetometer: List[str],
        features_name_velocity_gps: List[str],
        features_name_position_fusion: List[str],
        features_name_velocity_fusion: List[str],
        features_name_orientation_fusion: List[str],
        dtype: np.typing.DTypeLike,
        path_data: pathlib.Path
    ) -> None:

        """
        Initiate an EvaluationDataset instance.

        Args:
            - len_window (int): The number of sample by window.
            - coeff_frequency_division (int): The ratio of frequency division (e.g. a window of 2400 samples with a frequency division of 10 leads to a sequence of 240 samples).
            - scaling_type (int): Either 'normalization' for a min-max scaling between -1 and 1 or 'standardization' for a standard scaling with zero mean and unit std.
            - scalers (posnnet.data.scalers.Scalers): The project Scalers instance.
            - sessions_id (List[int]): The list of sessions id to use for this dataset.
            - features_name_accelerometer (List[str]): The list of features name for the accelerometer (0 <= len(features_name_accelerometer) <= 3).
            - features_name_gyroscope (List[str]): The list of features name for the gyroscope (0 <= len(features_name_gyroscope) <= 3).
            - features_name_magnetometer (List[str]): The list of features name for the magnetometer (0 <= len(features_name_magnetometer) <= 3).
            - features_name_velocity_gps (List[str]): The list of features name for the GPS velocity (0 <= len(features_name_velocity_gps) <= 3).
            - features_name_position_fusion (List[str]): The list of features name for the fusion position (1 <= len(features_name_position_fusion) <= 3).
            - features_name_velocity_fusion (List[str]): The list of features name for the fusion velocity (1 <= len(features_name_velocity_fusion) <= 3).
            - features_name_orientation_fusion (List[str]): The list of features name for the fusion orientation (0 <= len(features_name_orientation_fusion) <= 3).
            - dtype (np.typing.DTypeLike): The dtype to use for the dataset (can be either np.float16, np.float32 or np.float64).
            - path_data (pathlib.Path): The path of the directory where are stored the sessions data.
        """

        # Memorize parameters.
        self.len_window = len_window
        self.coeff_frequency_division = coeff_frequency_division

        # Load all the dataframe of the dataset.
        # Use a if statement because some sessions could not be present (for e.g. if a session duration is 1 minute and GPS 
        # outages are simulated from 2 to 3 minutes).
        dfs_sessions = [
            pd.read_pickle(
                filepath_or_buffer=pathlib.Path(path_data, f"session_{session_id:d}.pkl")
            )
            for session_id in sessions_id
            if pathlib.Path(path_data, f"session_{session_id:d}.pkl").exists()
        ]

        # Create individual list for each data source.
        # The list contains arrays of every sessions data.
        self.l_x_accelerometer = [
            df_session[features_name_accelerometer].copy(deep=True).to_numpy().astype(dtype=dtype)
            for df_session in dfs_sessions
        ]
        self.l_x_gyroscope = [
            df_session[features_name_gyroscope].copy(deep=True).to_numpy().astype(dtype=dtype)
            for df_session in dfs_sessions
        ]
        self.l_x_magnetometer = [
            df_session[features_name_magnetometer].copy(deep=True).to_numpy().astype(dtype=dtype)
            for df_session in dfs_sessions
        ]
        self.l_x_velocity_gps = [
            df_session[features_name_velocity_gps].copy(deep=True).to_numpy().astype(dtype=dtype)
            for df_session in dfs_sessions
        ]
        self.l_x_velocity_fusion = [
            df_session[features_name_velocity_fusion].copy(deep=True).to_numpy().astype(dtype=dtype)
            for df_session in dfs_sessions
        ]
        self.l_x_orientation_fusion = [
            df_session[features_name_orientation_fusion].copy(deep=True).to_numpy().astype(dtype=dtype)
            for df_session in dfs_sessions
        ]
        self.l_mask = [
            df_session[[mask_column for mask_column in df_session.columns if "gps_outage" in mask_column]].copy(deep=True).to_numpy().astype(dtype=bool)
            for df_session in dfs_sessions
        ]

        # Create individual arrays for the target (one position and one velocity unscaled).
        self.l_y_position_target = [
            df_session[features_name_position_fusion].copy(deep=True).to_numpy().astype(dtype=dtype)
            for df_session in dfs_sessions
        ]
        self.l_y_velocity_target = [
            df_session[features_name_velocity_fusion].copy(deep=True).to_numpy().astype(dtype=dtype)
            for df_session in dfs_sessions
        ]

        # Store the number of cumulated simulated GPS outage by sessions
        self.cumulative_n_gps_outages = np.cumsum([x_mask.shape[-1] for x_mask in self.l_mask])

        # If the sensor has an accelerometer, scale accelerometer data.
        if len(features_name_accelerometer) > 0:
            self.l_x_accelerometer = [
                scalers.scale(data=x_accelerometer, source_name="accelerometer", scaling_type=scaling_type)
                for x_accelerometer in self.l_x_accelerometer
            ]

        # If the sensor has a gyroscope, scale gyroscope data.
        if len(features_name_gyroscope) > 0:
            self.l_x_gyroscope = [
                scalers.scale(data=x_gyroscope, source_name="gyroscope", scaling_type=scaling_type)
                for x_gyroscope in self.l_x_gyroscope
            ]

        # If the sensor has a magnetometer, scale magnetometer data.
        if len(features_name_magnetometer) > 0:
            self.l_x_magnetometer = [
                scalers.scale(data=x_magnetometer, source_name="magnetometer", scaling_type=scaling_type)
                for x_magnetometer in self.l_x_magnetometer
            ]

        # If the sensor has a GPS velocity, scale GPS velocity data.
        if len(features_name_velocity_gps) > 0:
            self.l_x_velocity_gps = [
                scalers.scale(data=x_velocity_gps, source_name="velocity_gps", scaling_type=scaling_type)
                for x_velocity_gps in self.l_x_velocity_gps
            ]

        # Scale fusion velocity data.
        self.l_x_velocity_fusion = [
            scalers.scale(data=x_velocity_fusion, source_name="velocity_fusion", scaling_type=scaling_type)
            for x_velocity_fusion in self.l_x_velocity_fusion
        ]

        # If the sensor has a fusion orientation, scale fusion orientation data.
        if len(features_name_orientation_fusion) > 0:
            self.l_x_orientation_fusion = [
                scalers.scale(data=x_orientation_fusion, source_name="orientation_fusion", scaling_type=scaling_type)
                for x_orientation_fusion in self.l_x_orientation_fusion
            ]

    def __len__(
        self: Self
    ) -> int:

        """
        Returns the length of the dataset.

        Returns:
            - length (int): The number of items in the dataset.
        """

        length = self.cumulative_n_gps_outages[-1]

        return length

    def __getitem__(
        self: Self,
        idx: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        """
        Overwriting of the torch.utils.data.Dataset.__getitem__ method.

        Args:
            - idx (int): The index of the item of the dataset that have to be served.

        Returns:
            - x_accelerometer (np.ndarray): The accelerometer data scaled either with Minmax from -1 to 1 or Standard with mean = 0 / std = 1.
            - x_gyroscope (np.ndarray): The gyroscope data scaled either with Minmax from -1 to 1 or Standard with mean = 0 / std = 1.
            - x_magnetometer (np.ndarray): The magnetometer data scaled either with Minmax from -1 to 1 or Standard with mean = 0 / std = 1.
            - x_velocity_gps (np.ndarray): The GPS velocity data scaled either with Minmax from -1 to 1 or Standard with mean = 0 / std = 1.
            - x_velocity_fusion (np.ndarray): The fusion velocity data scaled either with Minmax from -1 to 1 or Standard with mean = 0 / std = 1.
            - x_orientation_fusion (np.ndarray): The fusion orientation data scaled either with Minmax from -1 to 1 or Standard with mean = 0 / std = 1.
            - mask (np.ndarray): The mask (GPS outage) data (0 or 1).
            - y_position_target (np.ndarray): The output position target unscaled.
            - y_velocity_target (np.ndarray): The output velocity target unscaled.
        """

        # Determine in which session the item is present. For this, use the cumulative sum of the number of GPS outages in sessions.
        # For example, if the number of GPS outages in sessions is [5, 12, 16, 3, ...] and the index of the asked item is 25,
        # the cumulative sum will be [5, 17, 33, 36, ...] and idx < self.cumulative_n_gps_outages will be [False, False, True, True, ...].
        # Finally, argmax will return the index of the first True, which is 2 (so the third session).
        idx_session = np.argmax(idx < self.cumulative_n_gps_outages)

        # Now let determine the index of the item inside the session.
        # If the item is in the first session, the index inside the session is the index of the dataset.
        # If the item is in another session, the index inside the session is the index of the dataset - the number of item inside every previous session.
        # Let take the same example that previously. Their is 5 GPS outages in the first session and 12 in the second session.
        # Thus, if we ask the 26-th (idx = 25) GPS outage, we have to remove 17, so the index inside the third session is 8 (9-th GPS outage of the third session).
        if idx_session > 0:
            idx_subsession = idx - self.cumulative_n_gps_outages[idx_session - 1]
        else:
            idx_subsession = idx

        # Collect the mask (mask is 1 when GPS outage and 0 otherwise) and compute the indices when mask == 1.
        mask_data = self.l_mask[idx_session][:, idx_subsession]
        indices_mask = np.flatnonzero(mask_data) # E.g. indices_mask = np.array([5003, 5004, ..., 6530, 6531]).

        # Compute the length of the session using the length of mask data.
        len_session = len(mask_data)

        # Get the first and last index when mask == 1.
        first_idx_mask = indices_mask[0]
        last_idx_mask = indices_mask[-1]

        # Compute the index of the center of the mask.
        idx_mask_center = (first_idx_mask + last_idx_mask) // 2

        # Compute the first index of the window. The window is centered around the mask.
        first_idx_window = idx_mask_center - self.len_window // 2

        # Ensure that the first index of the window is at least at len_window sample away from the end of the session.
        first_idx_window = min(first_idx_window, len_session - self.len_window)

        # Ensure that the first index of the window is superior or equal to 0.
        first_idx_window = max(first_idx_window, 0)

        # Compute the last index of the window.
        last_idx_window = first_idx_window + self.len_window - 1

        # Ensure that the last index of the window is at maximum the last index of the session.
        last_idx_window = min(last_idx_window, len_session - 1)

        # Ensure that the length of the window can be divided by coeff_frequency_division.
        last_idx_window -= (last_idx_window - first_idx_window + 1) % self.coeff_frequency_division

        # Compute the length of the window (can be inferior to the original window length if the session is not long enough).
        len_window = last_idx_window - first_idx_window + 1

        # Sample 1 / self.coeff_frequency_division samples from first_idx_window to first_idx_window + len_window of the accelerometer data.
        x_accelerometer = self.l_x_accelerometer[idx_session][[
            first_idx_window + offset
            for offset in range(0, len_window, self.coeff_frequency_division)
        ], :]

        # Sample 1 / self.coeff_frequency_division samples from first_idx_window to first_idx_window + len_window of the gyroscope data.
        x_gyroscope = self.l_x_gyroscope[idx_session][[
            first_idx_window + offset
            for offset in range(0, len_window, self.coeff_frequency_division)
        ], :]

        # Sample 1 / self.coeff_frequency_division samples from first_idx_window to first_idx_window + len_window of the magnetometer data.
        x_magnetometer = self.l_x_magnetometer[idx_session][[
            first_idx_window + offset
            for offset in range(0, len_window, self.coeff_frequency_division)
        ], :]

        # Sample 1 / self.coeff_frequency_division samples from first_idx_window to first_idx_window
        # of the self.coeff_frequency_division slidding average of the GPS velocity data.
        x_velocity_gps = np.array([
            np.mean([
                self.l_x_velocity_gps[idx_session][first_idx_window + offset + i_filter]
                for i_filter in range(self.coeff_frequency_division)
            ], axis=0)
            for offset in range(0, len_window, self.coeff_frequency_division)
        ])

        # Sample 1 / self.coeff_frequency_division samples from first_idx_window to first_idx_window
        # of the self.coeff_frequency_division slidding average of the fusion velocity data.
        x_velocity_fusion = np.array([
            np.mean([
                self.l_x_velocity_fusion[idx_session][first_idx_window + offset + i_filter]
                for i_filter in range(self.coeff_frequency_division)
            ], axis=0)
            for offset in range(0, len_window, self.coeff_frequency_division)
        ])

        # Sample 1 / self.coeff_frequency_division samples from first_idx_window to first_idx_window + len_window of the fusion orientation data.
        x_orientation_fusion = self.l_x_orientation_fusion[idx_session][[
            first_idx_window + offset
            for offset in range(0, len_window, self.coeff_frequency_division)
        ], :]

        # Sample 1 / self.coeff_frequency_division samples from first_idx_window to first_idx_window + len_window of the mask data.
        mask = self.l_mask[idx_session][[
            first_idx_window + offset
            for offset in range(0, len_window, self.coeff_frequency_division)
        ], idx_subsession : idx_subsession + 1]

        # Sample every samples from first_idx_window to first_idx_window + len_window of the target fusion position data.
        y_position_target = self.l_y_position_target[idx_session][first_idx_window : first_idx_window + len_window, :]

        # Sample every samples from first_idx_window to first_idx_window + len_window of the target fusion velocity data.
        y_velocity_target = self.l_y_velocity_target[idx_session][first_idx_window : first_idx_window + len_window, :]

        item = (
            x_accelerometer,
            x_gyroscope,
            x_magnetometer,
            x_velocity_gps,
            x_velocity_fusion,
            x_orientation_fusion,
            mask,
            y_position_target,
            y_velocity_target
        )

        return item

    def set_coeff_frequency_division(
        self: Self,
        coeff_frequency_division: int
    ) -> None:

        self.coeff_frequency_division = coeff_frequency_division

    def get_n_axes_by_data_sources(
        self: Self
    ) -> Dict[str, int]:

        """
        Returns a dictionnary with keys 'accelerometer', 'gyroscope', 'magnetometer', 'velocity_gps',
        'velocity_fusion', 'orientation_fusion' with values the number of axis by data source.

        Returns:
            - n_axes_by_data_sources(Dict[str, int]): the numbers of axes for each data source.
        """

        n_axes_by_data_sources = {
            "accelerometer": self.l_x_accelerometer[0].shape[-1],
            "gyroscope": self.l_x_gyroscope[0].shape[-1],
            "magnetometer": self.l_x_magnetometer[0].shape[-1],
            "velocity_gps": self.l_x_velocity_gps[0].shape[-1],
            "velocity_fusion": self.l_x_velocity_fusion[0].shape[-1],
            "orientation_fusion": self.l_x_orientation_fusion[0].shape[-1]
        }

        return n_axes_by_data_sources