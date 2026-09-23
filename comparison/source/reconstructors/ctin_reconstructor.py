import joblib
import numpy as np
import pathlib
import torch
from typing import Self, Tuple

from comparison.source.coordinate_frame import CoordinateFrameManager
from comparison.source.ctin.model import CTIN


class CTINReconstructor:

    def __init__(
        self: Self,
        sensor_frequency: float,
        beta: float
    ) -> None:

        """
        Reconstruction model based on CTIN.

        Args:
            - sensor_frequency (float): The frequency of the IMU in Hz.
            - beta (float): The beta coefficient of the Madgiwk algorithm.
        """

        self.coordinate_frame_manager = CoordinateFrameManager(
            sensor_frequency=sensor_frequency,
            beta=beta
        )

        self.dt = 1.0 / sensor_frequency
        self.WINDOW_SIZE = 100
        self.BATCH_SIZE = 512

        best_val_score = np.inf
        best_d_model = None
        best_batch_size = None

        for file_path in pathlib.Path("./comparison/models_data/").iterdir():

            if not "ctin" in file_path.name or not "checkpoint" in file_path.name:
                continue

            checkpoint = joblib.load(file_path)
            
            if np.min(checkpoint["losses_val"]) < best_val_score:

                best_val_score = np.min(checkpoint["losses_val"])
                best_d_model = file_path.name.split("_")[1]
                best_batch_size = file_path.name.split("_")[2]

        if best_d_model is None or best_batch_size is None:
            raise RuntimeError("No CTIN checkpoint found in ./comparison/models_data/")

        self.ctin_net = CTIN(
            input_dim=6,
            out_dim=3,
            d_model=int(best_d_model)
        )
        self.ctin_net.load_state_dict(torch.load(f"./comparison/models_data/ctin_{best_d_model}_{best_batch_size}_state_dict.pt"))
        self.ctin_net.eval().cuda()

    def reconstruct(
        self: Self,
        acc: np.ndarray,
        gyro: np.ndarray,
        velocity_0: np.ndarray,
        position_0: np.ndarray,
        roll_0: float,
        pitch_0: float,
        yaw_0: float
    ) -> Tuple[np.ndarray, np.ndarray]:

        """
        Reconstruct the velocity and the position.

        Args:
            - acc (np.ndarray): An array of shape (N, 3) containing the accelerations in the device CF in m/s^2.
            - gyro (np.ndarray): An array of shape (N, 3) containing the rotation rates in the device CF in deg/s.
            - velocity_0 (np.ndarray): An array of shape (3, ) containing the velocity at t = 0.
            - position_0 (np.ndarray): An array of shape (3, ) containing the position at t = 0.
            - roll_0 (float): The roll angle between the device coordinate frame and the global coordinate frame at t = 0 (in rad/s).
            - pitch_0 (float): The pitch angle between the device coordinate frame and the global coordinate frame at t = 0 (in rad/s).
            - yaw_0 (float): The yaw angle between the device coordinate frame and the global coordinate frame at t = 0 (in rad/s).
        """

        acc, gyro = self.coordinate_frame_manager.acceleration_from_device_to_global_coordinate_frame(
            acc=acc,
            gyro=gyro,
            roll_0=roll_0,
            pitch_0=pitch_0,
            yaw_0=yaw_0
        )

        velocity_pred = np.full(fill_value=np.nan, shape=(len(acc), 3))

        imu_data = torch.from_numpy(
            np.concatenate([acc, gyro], axis=1)
        ).float().cuda()

        x = imu_data[:self.WINDOW_SIZE].unsqueeze(0)

        with torch.no_grad():
            y_vel_pred, _ = self.ctin_net(x)

        y_vel_pred = y_vel_pred.squeeze(0).cpu().numpy()
        
        velocity_pred[:self.WINDOW_SIZE] = y_vel_pred

        for idx in range(1, len(acc) - self.WINDOW_SIZE + 1, self.BATCH_SIZE):

            x = torch.stack([
                imu_data[idx + offset : idx + offset + self.WINDOW_SIZE]
                for offset in range(min(self.BATCH_SIZE, len(acc) - self.WINDOW_SIZE + 1 - idx))
            ])

            with torch.no_grad():
                y_vel_pred, _ = self.ctin_net(x)

            y_vel_pred = y_vel_pred.cpu().numpy()

            velocity_pred[idx + self.WINDOW_SIZE - 1: idx + self.WINDOW_SIZE - 1 + min(self.BATCH_SIZE, len(acc) - self.WINDOW_SIZE + 1 - idx)] = np.stack([
                y_vel_pred[offset, -1, :]
                for offset in range(min(self.BATCH_SIZE, len(acc) - self.WINDOW_SIZE + 1 - idx))
            ])

        position_pred = self.dt * np.cumsum(velocity_pred, axis=0) + position_0

        return velocity_pred, position_pred