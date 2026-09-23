import numpy as np
from scipy.spatial.transform import Rotation
from typing import Self, Tuple

from comparison.source.coordinate_frame.madgwick import Madgwick


class CoordinateFrameManager:

    def __init__(
        self: Self,
        sensor_frequency: float,
        beta: float
    ) -> None:

        """
        Manage the coordinate frame of the acceleration.

        Args:
            - sensor_frequency (float): The frequency of the IMU in Hz.
            - beta (float): The beta coefficient of the Madgiwk algorithm.
        """

        self.madgwick = Madgwick(
            sensor_frequency=sensor_frequency,
            beta=beta
        )

    def acceleration_from_device_to_global_coordinate_frame(
        self: Self,
        acc: np.ndarray,
        gyro: np.ndarray,
        roll_0: float,
        pitch_0: float,
        yaw_0: float,
    ) -> Tuple[np.ndarray, np.ndarray]:

        """
        Transfer the acceleration from the device to the global coordinate frame.

        Args:
            - acc (np.ndarray): An array of shape (N, 3) containing the accelerations in the device CF in m/s^2.
            - gyro (np.ndarray): An array of shape (N, 3) containing the rotation rates in the device CF in deg/s.
            - roll_0 (float): The roll angle between the device coordinate frame and the global coordinate frame at t = 0 (in rad/s).
            - pitch_0 (float): The pitch angle between the device coordinate frame and the global coordinate frame at t = 0 (in rad/s).
            - yaw_0 (float): The yaw angle between the device coordinate frame and the global coordinate frame at t = 0 (in rad/s).

        Returns:
            - acc (np.ndarray): An array of shape (N, 3) containing the accelerations in the global CF in m/s^2.
            - gyro (np.ndarray): An array of shape (N, 3) containing the rotation rates in the global CF in deg/s.
        """

        gyro = np.deg2rad(gyro)
        R0=Rotation.from_euler('zyx', [yaw_0, pitch_0, roll_0], degrees=False)

        qs = self.madgwick.collect_quaternions(
            R0=R0,
            acc=acc,
            gyro=gyro
        )

        R = Rotation.from_quat(qs, scalar_first=True)
        Rx180 = Rotation.from_euler('x', 180, degrees=True)
        acc = (Rx180 * R).apply(acc)
        gyro = (Rx180 * R).apply(gyro) 

        return acc, gyro