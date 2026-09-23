import numpy as np
from typing import Self, Tuple

from comparison.source.coordinate_frame import CoordinateFrameManager


class NaiveIntegrationReconstructor:

    def __init__(
        self: Self,
        sensor_frequency: float,
        beta: float
    ) -> None:

        """
        Reconstruction model based on Naive Integration

        Args:
            - sensor_frequency (float): The frequency of the IMU in Hz.
            - beta (float): The beta coefficient of the Madgiwk algorithm.
        """

        self.coordinate_frame_manager = CoordinateFrameManager(
            sensor_frequency=sensor_frequency,
            beta=beta
        )

        self.dt = 1.0 / sensor_frequency

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

        acc, _ = self.coordinate_frame_manager.acceleration_from_device_to_global_coordinate_frame(
            acc=acc,
            gyro=gyro,
            roll_0=roll_0,
            pitch_0=pitch_0,
            yaw_0=yaw_0
        )

        velocity_pred = self.dt * np.cumsum(acc, axis=0) + velocity_0
        position_pred = self.dt * np.cumsum(velocity_pred, axis=0) + position_0

        return velocity_pred, position_pred