from numba import njit
import numpy as np
import scipy
from typing import Self


@njit
def update_quaternion_compiled(
    q: np.ndarray,
    ax: float,
    ay: float,
    az: float,
    gx: float,
    gy: float,
    gz: float,
    beta: float,
    dt: float
) -> np.ndarray:

    """
    Update the quaternion q with the last IMU data.

    Args:
        - q (np.ndarray): The quaternion to update of shape (4, ).
        - ax (float): acceleration on the axe X of the coordinate frame of the device, in m/s^2.
        - ay (float): acceleration on the axe Y of the coordinate frame of the device, in m/s^2.
        - az (float): acceleration on the axe Z of the coordinate frame of the device, in m/s^2.
        - gx (float): rotation rate on the axe X of the coordinate frame of the device, in rad/s.
        - gy (float): rotation rate on the axe Y of the coordinate frame of the device, in rad/s.
        - gz (float): rotation rate on the axe Z of the coordinate frame of the device, in rad/s.
        - beta (float): The beta coefficient of the algorithm.
        - df (float): The sampling period in seconds.

    Returns:
        - q (np.ndarray): The updated quaternion.
    """

    q1, q2, q3, q4 = q

    # Normalize accelerometer
    norm = np.sqrt(ax*ax + ay*ay + az*az)

    if norm == 0.0:

        # Can't correct with invalid accel, just integrate gyro
        q_dot = 0.5 * np.array([
            -q2*gx - q3*gy - q4*gz,
             q1*gx + q3*gz - q4*gy,
             q1*gy - q2*gz + q4*gx,
             q1*gz + q2*gy - q3*gx,
        ])

        q += q_dot * dt
        q /= np.linalg.norm(q)

        return q

    ax, ay, az = ax / norm, ay / norm, az / norm

    # Auxiliary variables
    _2q1 = 2.0 * q1
    _2q2 = 2.0 * q2
    _2q3 = 2.0 * q3
    _2q4 = 2.0 * q4

    # Objective function (difference between estimated gravity and measured accel)
    # (From Madgwick 2010 report, IMU update)
    f1 = _2q2*q4 - _2q1*q3 - ax
    f2 = _2q1*q2 + _2q3*q4 - ay
    f3 = 1.0 - 2.0*(q2*q2 + q3*q3) - az

    # Jacobian matrix J of f wrt q
    J = np.array([
        [-_2q3,      _2q4,      -_2q1,      _2q2],
        [ _2q2,       _2q1,       _2q4,      _2q3],
        [ 0.0,       -4.0*q2,    -4.0*q3,   0.0 ]
    ])

    f = np.array([f1, f2, f3])

    # Gradient (matrix multiplication J^T * f)
    step = J.T @ f
    step_norm = np.linalg.norm(step)
    if step_norm > 0.0:
        step /= step_norm  # normalize step

    # Compute quaternion derivative from gyroscope
    # q_dot_gyro = 0.5 * q ⊗ [0, gx, gy, gz]
    q_dot_gyro = 0.5 * np.array([
        -q2*gx - q3*gy - q4*gz,
         q1*gx + q3*gz - q4*gy,
         q1*gy - q2*gz + q4*gx,
         q1*gz + q2*gy - q3*gx,
    ])

    # Combine gyroscope and gradient descent corrective step
    q_dot = q_dot_gyro - beta * step

    # Integrate to yield new quaternion
    q += q_dot * dt
    q /= np.linalg.norm(q)  # normalize

    return q

@njit
def collect_quaternions_compiled(
    q0: np.ndarray,
    acc: np.ndarray,
    gyro: np.ndarray,
    beta: float,
    dt: float
) -> np.ndarray:

    """
    Collect the quaternions for every timestamp.

    Args:
        - q0 (np.ndarray): The quaternion at t0 of shape (4, ).
        - acc (np.ndarray): An array of shape (N, 3) containing the accelerations in m/s^2.
        - gyro (np.ndarray): An array of shape (N, 3) containing the rotation rates in rad/s.
        - beta (float): The beta coefficient of the algorithm.
        - df (float): The sampling period in seconds.

    Returns:
        - qs (np.ndarray): The quaternions for every timestamp of shape (N, 4).
    """

    N_samples = acc.shape[0]
    qs = np.empty((N_samples, 4))
    q = q0.copy()

    for i_sample in range(N_samples):

        q = update_quaternion_compiled(
            q=q,
            ax=acc[i_sample, 0],
            ay=acc[i_sample, 1],
            az=acc[i_sample, 2],
            gx=gyro[i_sample, 0],
            gy=gyro[i_sample, 1],
            gz=gyro[i_sample, 2],
            beta=beta,
            dt=dt
        )
        qs[i_sample, :] = q

    return qs

class Madgwick:

    def __init__(
        self: Self,
        sensor_frequency: float,
        beta: float
    ) -> None:

        self.dt = 1.0 / sensor_frequency
        self.beta = beta

    def collect_quaternions(
        self: Self,
        R0: scipy.spatial.transform.Rotation,
        acc: np.ndarray,
        gyro: np.ndarray
    ) -> np.ndarray:

        """
        Collect the quaternions for every timestamp.
    
        Args:
            - R0 (scipy.spatial.transform.Rotation): The rotation between the device and the global coordinate frame at t0.
            - acc (np.ndarray): An array of shape (N, 3) containing the accelerations in m/s^2.
            - gyro (np.ndarray): An array of shape (N, 3) containing the rotation rates in rad/s.
    
        Returns:
            - qs (np.ndarray): The quaternions for every timestamp of shape (N, 4).
        """

        q0 = R0.as_quat(scalar_first=True)
        qs = collect_quaternions_compiled(
            q0=q0,
            acc=acc,
            gyro=gyro,
            beta=self.beta,
            dt=self.dt
        )

        return qs