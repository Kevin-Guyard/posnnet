import numpy as np
import pandas as pd
import pathlib
import torch
from typing import Self, List, Tuple, Dict

import posnnet.data.scalers


class Dataset(torch.utils.data.Dataset):

    def __init__(
        self: Self,
        len_window: int,
        coeff_frequency_division: int,
        coeff_sampling: int,
        n_epochs_sampling: int,
        scaling_type: str,
        scalers: posnnet.data.scalers.Scalers,
        sessions_id: List[int],
        features_name_accelerometer: List[str],
        features_name_gyroscope: List[str],
        features_name_magnetometer: List[str],
        features_name_velocity_gps: List[str],
        features_name_velocity_fusion: List[str],
        features_name_orientation_fusion: List[str],
        dtype: np.typing.DTypeLike,
        path_data: pathlib.Path
    ) -> None:

        """
        Initiate a Dataset instance.

        Args:
            - len_window (int): The number of sample by window.
            - coeff_frequency_division (int): The ratio of frequency division (e.g. a window of 2400 samples with a frequency division of 10 leads to a sequence of 240 samples).
            - coeff_sampling (int): The ratio of sample that will be used as window beginning every epoch (e.g. coeff_sampling = 1000 means 1/1000 of the dataset sample will be used to create a window).
            - n_epochs_sampling (int): The number of epochs for a complete sampling rotation of the dataset.
            - scaling_type (int): Either 'normalization' for a min-max scaling between -1 and 1 or 'standardization' for a standard scaling with zero mean and unit std.
            - scalers (posnnet.data.scalers.Scalers): The project Scalers instance.
            - sessions_id (List[int]): The list of sessions id to use for this dataset.
            - features_name_accelerometer (List[str]): The list of features name for the accelerometer (0 <= len(features_name_accelerometer) <= 3).
            - features_name_gyroscope (List[str]): The list of features name for the gyroscope (0 <= len(features_name_gyroscope) <= 3).
            - features_name_magnetometer (List[str]): The list of features name for the magnetometer (0 <= len(features_name_magnetometer) <= 3).
            - features_name_velocity_gps (List[str]): The list of features name for the GPS velocity (0 <= len(features_name_velocity_gps) <= 3).
            - features_name_velocity_fusion (List[str]): The list of features name for the fusion velocity (1 <= len(features_name_velocity_fusion) <= 3).
            - features_name_orientation_fusion (List[str]): The list of features name for the fusion orientation (0 <= len(features_name_orientation_fusion) <= 3).
            - dtype (np.typing.DTypeLike): The dtype to use for the dataset (can be either np.float16, np.float32 or np.float64).
            - path_data (pathlib.Path): The path of the directory where are stored the sessions data.
        """

        # Save parameters
        self.len_window = len_window # The number of sample in a window (before frequency division).
        self.coeff_frequency_division = coeff_frequency_division # The ratio of frequency division (e.g. a window of 2400 samples with a frequency division of 10 leads to a sequence of 240 samples).
        self.coeff_sampling = coeff_sampling # The ratio of sample that will be used as window beginning every epoch (e.g. coeff_sampling = 1000 means 1/1000 of the dataset sample will be used to create a window).
        self.n_epochs_sampling = n_epochs_sampling # The number of epochs for a complete sampling rotation of the dataset.

        # Load all the dataframes of the dataset
        dfs_sessions = [
            pd.read_pickle(
                filepath_or_buffer=pathlib.Path(path_data, f"session_{session_id:d}.pkl")
            )
            for session_id in sessions_id
        ]

        # Compute the exploitable shapes for each session dataframe. The exploitable shape is the range of indices that can be used as beginning of a window
        # E.g. if a session dataframe has a shape of 10 000 and the len of a window is 1 000, the 9 000 first indices can be used as beginning of a window
        exploitable_shapes_by_df = np.array([
            len(df_session) - len_window + 1
            for df_session in dfs_sessions
        ])

        # Compute the cumulative sum of the exploitable shapes [len(df_1), len(df_1) + len(df_2), len(df_1) + len(df_2) + len(df_3), ...]
        cumulative_sum_exploitable_shapes = np.cumsum(exploitable_shapes_by_df)

        # Compute exploitable indices (exploitable idx = idx in the session + len of all the previous session)
        self.exploitable_indices = [
            idx + (len_window - 1) * sum([
                idx >= cumulative_sum_exploitable_shapes[idx_df] 
                for idx_df in range(len(cumulative_sum_exploitable_shapes))
            ])
            for idx in range(cumulative_sum_exploitable_shapes[-1])
        ]

        # Initialize sampler
        self.set_epoch(i_epoch=0)

        # Concatenate the dataframes into a single one
        df_sessions = pd.concat(dfs_sessions).reset_index(drop=True)

        # Create individual arrays for each data source
        self.x_accelerometer = df_sessions[features_name_accelerometer].copy(deep=True).to_numpy().astype(dtype=dtype)
        self.x_gyroscope = df_sessions[features_name_gyroscope].copy(deep=True).to_numpy().astype(dtype=dtype)
        self.x_magnetometer = df_sessions[features_name_magnetometer].copy(deep=True).to_numpy().astype(dtype=dtype)
        self.x_velocity_gps = df_sessions[features_name_velocity_gps].copy(deep=True).to_numpy().astype(dtype=dtype)
        self.x_velocity_fusion = df_sessions[features_name_velocity_fusion].copy(deep=True).to_numpy().astype(dtype=dtype)
        self.x_orientation_fusion = df_sessions[features_name_orientation_fusion].copy(deep=True).to_numpy().astype(dtype=dtype)

        # Create individual arrays for the target (one which will scaled and the other not)
        self.y_velocity_fusion = df_sessions[features_name_velocity_fusion].copy(deep=True).to_numpy().astype(dtype=dtype)
        self.y_velocity_fusion_unscaled = df_sessions[features_name_velocity_fusion].copy(deep=True).to_numpy().astype(dtype=dtype)

        # If the sensor has an accelerometer, scale accelerometer data
        if len(features_name_accelerometer) > 0:
            self.x_accelerometer = scalers.scale(data=self.x_accelerometer, source_name="accelerometer", scaling_type=scaling_type)

        # If the sensor has a gyroscope, scale gyroscope data
        if len(features_name_gyroscope) > 0:
            self.x_gyroscope = scalers.scale(data=self.x_gyroscope, source_name="gyroscope", scaling_type=scaling_type)

        # If the sensor has a magnetometer, scale magnetometer data
        if len(features_name_magnetometer) > 0:
            self.x_magnetometer = scalers.scale(data=self.x_magnetometer, source_name="magnetometer", scaling_type=scaling_type)

        # If the sensor has a velocity gps, scale velocity gps data
        if len(features_name_velocity_gps) > 0:
            self.x_velocity_gps = scalers.scale(data=self.x_velocity_gps, source_name="velocity_gps", scaling_type=scaling_type)

        # Scale velocity fusion data
        self.x_velocity_fusion = scalers.scale(data=self.x_velocity_fusion, source_name="velocity_fusion", scaling_type=scaling_type)

        # If the sensor has a orientation fusion, scale orientation fusion data
        if len(features_name_orientation_fusion) > 0:
            self.x_orientation_fusion = scalers.scale(data=self.x_orientation_fusion, source_name="orientation_fusion", scaling_type=scaling_type)

        # Scale target velocity fusion data
        self.y_velocity_fusion = scalers.scale(data=self.y_velocity_fusion, source_name="velocity_fusion", scaling_type="normalization")

    def __len__(
        self: Self
    ) -> int:

        """
        Returns the length of the dataset.

        Returns:
            - length (int): The number of items in the dataset (only valid for the current epoch).
        """

        # The len of the dataset is the number of sampled item (because not all the sample are considered as a starting point for a window at every epoch, only a fraction are considered)
        length = len(self.sampled_indices)

        return length

    def __getitem__(
        self: Self,
        idx: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

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
            - y_velocity_fusion (np.ndarray): The fusion velocity data scaled with MinMax from -1 or 1.
            - y_velocity_fusion_unscaled_undivide (np.ndarray): The fusion velocity data unscaled and without frequency division.
        """

        # Get the first index of the window
        idx_sampled = self.sampled_indices[idx]

        # Sample one sample every coeff_frequency_division samples for the accelerometer
        x_accelerometer = self.x_accelerometer[[
            idx_sampled + offset
            for offset in range(0, self.len_window, self.coeff_frequency_division)
        ], :]

        # Sample one sample every coeff_frequency_division samples for the gyroscope
        x_gyroscope = self.x_gyroscope[[
            idx_sampled + offset
            for offset in range(0, self.len_window, self.coeff_frequency_division)
        ], :]

        # Sample one sample every coeff_frequency_division samples for the magnetometer
        x_magnetometer = self.x_magnetometer[[
            idx_sampled + offset
            for offset in range(0, self.len_window, self.coeff_frequency_division)
        ], :]

        # Average the value of coeff_frequency_division samples for the GPS velocity
        x_velocity_gps = np.array([
            np.mean([
                self.x_velocity_gps[idx_sampled + offset + i_filter]
                for i_filter in range(self.coeff_frequency_division)
            ], axis=0)
            for offset in range(0, self.len_window, self.coeff_frequency_division)
        ])

        # Average the value of coeff_frequency_division samples for the fusion velocity
        x_velocity_fusion = np.array([
            np.mean([
                self.x_velocity_fusion[idx_sampled + offset + i_filter]
                for i_filter in range(self.coeff_frequency_division)
            ], axis=0)
            for offset in range(0, self.len_window, self.coeff_frequency_division)
        ])

        # Sample one sample every coeff_frequency_division samples for the fusion angle
        x_orientation_fusion = self.x_orientation_fusion[[
            idx_sampled + offset
            for offset in range(0, self.len_window, self.coeff_frequency_division)
        ]]

        # Average the value of coeff_frequency_division samples for the target (fusion velocity)
        y_velocity_fusion = np.array([
            np.mean([
                self.y_velocity_fusion[idx_sampled + offset + i_filter]
                for i_filter in range(self.coeff_frequency_division)
            ], axis=0)
            for offset in range(0, self.len_window, self.coeff_frequency_division)
        ])

        # Get every samples for the target unscaled (fusion velocity)
        y_velocity_fusion_unscaled_undivide = self.y_velocity_fusion_unscaled[idx_sampled : idx_sampled + self.len_window]

        item = (
            x_accelerometer,
            x_gyroscope,
            x_magnetometer,
            x_velocity_gps,
            x_velocity_fusion,
            x_orientation_fusion,
            y_velocity_fusion,
            y_velocity_fusion_unscaled_undivide
        )

        return item

    def set_epoch(
        self: Self,
        i_epoch: int
    ) -> None:

        """
        Prepare the dataset to serve the data of the epoch asked. In particular, compute the sampler modulo and sample indices.

        Args:
            - i_epoch: (int): The number of the epoch for which to serve the data.
        """

        # Set the modulo of the sampler in function of the current epoch
        modulo_sampler = (i_epoch * (self.coeff_sampling // self.n_epochs_sampling)) % self.coeff_sampling

        # Sample indices from exploitable indices
        self.sampled_indices = [
            idx 
            for offset, idx in enumerate(self.exploitable_indices) 
            if offset % self.coeff_sampling == modulo_sampler
        ]     

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
            "accelerometer": self.x_accelerometer.shape[-1],
            "gyroscope": self.x_gyroscope.shape[-1],
            "magnetometer": self.x_magnetometer.shape[-1],
            "velocity_gps": self.x_velocity_gps.shape[-1],
            "velocity_fusion": self.x_velocity_fusion.shape[-1],
            "orientation_fusion": self.x_orientation_fusion.shape[-1]
        }

        return n_axes_by_data_sources