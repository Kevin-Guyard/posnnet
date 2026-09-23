import numpy as np
import torch
from typing import Self, Tuple

from comparison.source.coordinate_frame import CoordinateFrameManager
from comparison.source.ronin.model_resnet1d import ResNet1D, BasicBlock1D, FCOutputModule


class RoninResNet1DReconstructor:

    def __init__(
        self: Self,
        sensor_frequency: float,
        beta: float
    ) -> None:

        """
        Reconstruction model based on Ronin ResNet1D.

        Args:
            - sensor_frequency (float): The frequency of the IMU in Hz.
            - beta (float): The beta coefficient of the Madgiwk algorithm.
        """

        self.coordinate_frame_manager = CoordinateFrameManager(
            sensor_frequency=sensor_frequency,
            beta=beta
        )

        self.dt = 1.0 / sensor_frequency
        self.window_size = 100
        self.window_step = 5

        self.resnet1d = ResNet1D(
            num_inputs=6,
            num_outputs=3,
            block_type=BasicBlock1D,
            group_sizes=[2, 2, 2, 2],
            base_plane=64,
            output_block=FCOutputModule,
            kernel_size=3,
            **{'fc_dim': 512, 'in_dim': 4, 'dropout': 0.5, 'trans_planes': 128}
        )
        self.resnet1d.load_state_dict(torch.load("./comparison/models_data/ronin_resnet1d_state_dict.pt"))
        self.resnet1d.eval().cuda()

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

        for i in range(0, len(acc) - self.window_size + 1, self.window_step):

            x = torch.from_numpy(
                np.concatenate([
                    acc[i : i + self.window_size],
                    gyro[i : i + self.window_size]
                ], axis=1)
            ).unsqueeze(0).swapaxes(1, 2).float().cuda()

            with torch.no_grad():
                y = self.resnet1d(x)

            if i == 0:
                
                velocity_pred[0 : self.window_size] = y.squeeze(0).cpu().numpy() / (self.window_size * self.dt)
                
            else:
                
                velocity_pred[i + self.window_size - 1] = y.squeeze(0).cpu().numpy() / (self.window_size * self.dt)
                velocity_pred[i + self.window_size - self.window_step : i + self.window_size - 1] = np.stack([
                    velocity_pred[i + self.window_size - self.window_step - 1] + (j + 1) * (velocity_pred[i + self.window_size - 1] - velocity_pred[i + self.window_size - self.window_step - 1])
                    for j in range(self.window_step - 1)
                ])

        velocity_pred[len(acc) - len(acc) % self.window_step : len(acc)] = velocity_pred[len(acc) - len(acc) % self.window_step - 1]
        position_pred = self.dt * np.cumsum(velocity_pred, axis=0) + position_0

        return velocity_pred, position_pred