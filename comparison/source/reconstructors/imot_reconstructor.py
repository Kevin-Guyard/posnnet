import joblib
import numpy as np
import pathlib
import torch
from typing import Self, Tuple

from comparison.source.coordinate_frame import CoordinateFrameManager
from comparison.source.imot.model import IMOT


class IMOTReconstructor:

    def __init__(
        self: Self,
        sensor_frequency: float,
        beta: float
    ) -> None:

        """
        Reconstruction model based on IMOT.

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
        self.BATCH_SIZE = 64

        self.imot_net = IMOT()

        checkpoint_imot_NCF = joblib.load("./comparison/models_data/imot_NCF_checkpoint.pkl")
        checkpoint_imot_DCF = joblib.load("./comparison/models_data/imot_DCF_checkpoint.pkl")

        self.acc_in_global_frame = np.min(checkpoint_imot_NCF["losses_val"]) <= np.min(checkpoint_imot_DCF["losses_val"])

        self.imot_net.load_state_dict(torch.load(f"./comparison/models_data/imot_{'NCF' if self.acc_in_global_frame else 'DCF'}_state_dict.pt"))
        self.imot_net.eval().cuda()

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

        if self.acc_in_global_frame:
            acc, gyro = self.coordinate_frame_manager.acceleration_from_device_to_global_coordinate_frame(
                acc=acc,
                gyro=gyro,
                roll_0=roll_0,
                pitch_0=pitch_0,
                yaw_0=yaw_0
            )

        velocity_pred = np.full(fill_value=np.nan, shape=(len(acc), 3))

        acc = torch.from_numpy(acc).float().cuda()
        gyro = torch.from_numpy(gyro).float().cuda()

        for idx in range(0, len(acc) - self.WINDOW_SIZE + 1, self.BATCH_SIZE):

            x_a_a = torch.stack([
                acc[idx + offset : idx + offset + self.WINDOW_SIZE]
                for offset in range(min(self.BATCH_SIZE, len(acc) - self.WINDOW_SIZE + 1 - idx))
            ]).swapaxes(1, 2)
            x_a_g = torch.stack([
                gyro[idx + offset : idx + offset + self.WINDOW_SIZE]
                for offset in range(min(self.BATCH_SIZE, len(acc) - self.WINDOW_SIZE + 1 - idx))
            ]).swapaxes(1, 2)

            with torch.no_grad():
                y = self.imot_net(a_a=x_a_a, a_g=x_a_g)

            y = y.cpu().numpy()

            velocity_pred[idx + self.WINDOW_SIZE - 1 : idx + self.WINDOW_SIZE - 1 + x_a_a.size(dim=0)] = np.stack([
                y[offset, :]
                for offset in range(x_a_a.size(dim=0))
            ])

        velocity_pred[0 : self.WINDOW_SIZE - 1] = velocity_pred[self.WINDOW_SIZE - 1]

        position_pred = self.dt * np.cumsum(velocity_pred, axis=0) + position_0

        return velocity_pred, position_pred