import numpy as np
import pandas as pd
import pathlib
import random
import torch
from typing import Self, Tuple

from comparison.source.coordinate_frame import CoordinateFrameManager


class Dataset:

    def __init__(
        self: Self,
        sensor_frequency: float,
        beta: float,
        path_data: pathlib.Path,
    ) -> None:

        self.coordinate_frame_manager = CoordinateFrameManager(
            sensor_frequency=sensor_frequency,
            beta=beta
        )

        window_size = 100
        window_step = 25

        self.x = []
        self.y_vel = []
        self.y_pos = []

        for session_path in path_data.iterdir():

            if not "session" in session_path.name:
                continue

            df_session = pd.read_pickle(filepath_or_buffer=session_path)

            idx = 1
            len_session = len(df_session)

            acceleration = df_session[["accelerometer_x", "accelerometer_y", "accelerometer_z"]].copy(deep=True).to_numpy().astype(np.float32)
            gyroscope = df_session[["gyroscope_x", "gyroscope_y", "gyroscope_z"]].copy(deep=True).to_numpy().astype(np.float32)
            roll = df_session["orientation_fusion_x"].copy(deep=True).to_numpy().astype(np.float32)
            pitch = df_session["orientation_fusion_y"].copy(deep=True).to_numpy().astype(np.float32)
            yaw = df_session["orientation_fusion_z"].copy(deep=True).to_numpy().astype(np.float32)
            velocity = df_session[["velocity_fusion_x", "velocity_fusion_y", "velocity_fusion_z"]].copy(deep=True).to_numpy().astype(np.float32)
            position = df_session[["position_fusion_x", "position_fusion_y", "position_fusion_z"]].copy(deep=True).to_numpy().astype(np.float32)

            del df_session

            while idx + window_size < len_session:

                acc = acceleration[idx : idx + window_size]
                gyro = gyroscope[idx : idx + window_size]
                roll_0 = roll[idx - 1]
                pitch_0 = pitch[idx - 1]
                yaw_0 = yaw[idx - 1]
                
                acc, gyro = self.coordinate_frame_manager.acceleration_from_device_to_global_coordinate_frame(
                    acc=acc,
                    gyro=gyro,
                    roll_0=roll_0,
                    pitch_0=pitch_0,
                    yaw_0=yaw_0
                )

                self.x.append(np.concatenate([acc, gyro], axis=1))
                self.y_vel.append(velocity[idx : idx + window_size])
                self.y_pos.append(position[idx : idx + window_size] - position[idx - 1])

                idx += window_step

        self.x = np.stack(self.x).astype(np.float32)
        self.y_vel = np.stack(self.y_vel).astype(np.float32)
        self.y_pos = np.stack(self.y_pos).astype(np.float32)

    def __len__(
        self: Self
    ) -> int:

        return len(self.x)

    def __getitem__(
        self: Self,
        idx: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        return self.x[idx], self.y_vel[idx], self.y_pos[idx]