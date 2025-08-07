import joblib
import numpy as np
import pandas as pd
import pathlib
import sklearn.preprocessing
from typing import Self, TypeVar, Type, List


class Scalers:

    def __init__(
        self: Self
    ) -> None:

        """
        Initialize the a MinMaxScaler (data range between -1 and 1) and a StandardScaler (data zero mean unit std) for every data source.
        """

        self.__scalers = {
            "accelerometer": {
                "normalization": sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1)),
                "standardization": sklearn.preprocessing.StandardScaler(),
            },
            "gyroscope": {
                "normalization": sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1)),
                "standardization": sklearn.preprocessing.StandardScaler(),
            },
            "magnetometer": {
                "normalization": sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1)),
                "standardization": sklearn.preprocessing.StandardScaler(),
            },
            "velocity_gps": {
                "normalization": sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1)),
                "standardization": sklearn.preprocessing.StandardScaler(),
            },
            "velocity_fusion": {
                "normalization": sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1)),
                "standardization": sklearn.preprocessing.StandardScaler(),
            },
            "orientation_fusion": {
                "normalization": sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1)),
                "standardization": sklearn.preprocessing.StandardScaler(),
            }
        }

    @classmethod
    def fit(
        cls: Type[TypeVar("Scalers")],
        columns_for_accelerometer: List[str],
        columns_for_gyroscope: List[str],
        columns_for_magnetometer: List[str],
        columns_for_velocity_gps: List[str],
        columns_for_velocity_fusion: List[str],
        columns_for_orientation_fusion: List[str],
        dtype: np.typing.DTypeLike,
        path_preprocessed_training_data: pathlib.Path,
        path_scalers: pathlib.Path,
    ) -> None:

        """
        Instantiate the scalers and fit them on the training dataset.

        Args:
            - columns_for_accelerometer (List[str]): The list of columns name for the accelerometer (0 <= len <= 3).
            - columns_for_gyroscope (List[str]): The list of columns name for the gyroscope (0 <= len <= 3).
            - columns_for_magnetometer (List[str]): The list of columns name for the magnetometer (0 <= len <= 3).
            - columns_for_velocity_gps (List[str]): The list of columns name for the GPS velocity (0 <= len <= 3).
            - columns_for_velocity_fusion (List[str]): The list of columns name for the Kalman filter output velocity (0 <= len <= 3).
            - columns_for_orientation_fusion (List[str]): The list of columns name for the Kalman filter output orientation (0 <= len <= 3).
            - dtype (np.typing.DTypeLike): The dtype of the data / neural networks (can be np.float16, np.float32 or np.float64).
            - path_preprocessed_training_data (pathlib.Path): The path where are stored preprocessed training data.
            - path_scalers (pathlib.Path): The path where will be stored the scalers and the relax points.
        """

        # Create an instance
        self = cls()

        # Iterate over all the sessions in the preprocessed training data directory
        # and concat them inside one unique pandas dataframe.
        df_train = pd.concat([
            pd.read_pickle(filepath_or_buffer=session_file)
            for session_file in path_preprocessed_training_data.iterdir()
            if "session" in session_file.name
        ])

        for source_name, columns_in_dataframe in [
            ("accelerometer", columns_for_accelerometer),
            ("gyroscope", columns_for_gyroscope),
            ("magnetometer", columns_for_magnetometer),
            ("velocity_gps", columns_for_velocity_gps),
            ("velocity_fusion", columns_for_velocity_fusion),
            ("orientation_fusion", columns_for_orientation_fusion),
        ]:

            # If the sensor hasn't any feature for this data source, skip it.
            if len(columns_in_dataframe) == 0:
                continue

            # Load data into a numpy array
            x = df_train[columns_in_dataframe].copy(deep=True).to_numpy().astype(dtype=dtype)
            # Fit both MinMax and Standard scalers.
            self.__scalers[source_name]["normalization"].fit(x)
            self.__scalers[source_name]["standardization"].fit(x)

        # Save the instance on disk
        joblib.dump(value=self, filename=pathlib.Path(path_scalers, "scalers.pkl"))

        # Initialize relax points dictionnary.
        relax_points = {}

        for scaling_type in ["normalization", "standardization"]:

            # Initialize sub dictionnary
            relax_points[scaling_type] = {}

            for source_name, columns_in_dataframe in [
                ("velocity_gps", columns_for_velocity_gps),
                ("velocity_fusion", columns_for_velocity_fusion),
                ("orientation_fusion", columns_for_orientation_fusion),
            ]:
    
                if len(columns_in_dataframe) > 0:
                    
                    relax_points[scaling_type][source_name] = self.scale(
                        data=np.zeros(shape=(1, len(columns_in_dataframe))),
                        source_name=source_name,
                        scaling_type=scaling_type
                    )[0]
    
                else:
    
                    relax_points[scaling_type][source_name] = np.array([])

        # Save the relax points
        joblib.dump(value=relax_points, filename=pathlib.Path(path_scalers, "relax_points.pkl"))
        
    @classmethod
    def load(
        cls: Type[TypeVar("Scalers")],
        path_scalers: pathlib.Path
    ) -> Self:

        """
        Load the Scalers instance present in the provided path.

        Args:
            - path_scalers (pathlib.Path): The path of the directory where is stored the Scalers instance.

        Returns:
            - scalers (Scalers): The instance of the scalers.
        """

        scalers = joblib.load(filename=pathlib.Path(path_scalers, "scalers.pkl"))

        return scalers

    def scale(
        self: Self,
        data: np.ndarray,
        source_name: str,
        scaling_type: str
    ) -> np.ndarray:

        """
        Scale the data.

        Args:
            - data (np.ndarray): The data to scale with a shape (L, D) where L is a positive integer and 1 <= D <= 3 depending of the number of axes of the sensor.
            - source_name (str): The name of the data source ('accelerometer', 'gyroscope', 'magnetometer', 'velocity_gps', 'velocity_fusion', 'orientation_fusion').
            - scaling_type (str): The type of scaling, either 'normalization' for a scaling with a min max from -1 to 1 or 'standardization' for a scaling 0 mean and unit std.

        Returns:
            - data (np.ndarray): The input array scaled.
        """

        data = self.__scalers[source_name][scaling_type].transform(data)

        return data

    def unscale(
        self: Self,
        data: np.ndarray,
        source_name: str,
        scaling_type: str
    ) -> np.ndarray:

        """
        Unscale the data.

        Args:
            - data (np.ndarray): The data to unscale with a shape (L, D) where L is a positive integer and 1 <= D <= 3 depending of the number of axes of the sensor.
            - source_name (str): The name of the data source ('accelerometer', 'gyroscope', 'magnetometer', 'velocity_gps', 'velocity_fusion', 'orientation_fusion').
            - scaling_type (str): The type of scaling, either 'normalization' for a scaling with a min max from -1 to 1 or 'standardization' for a scaling 0 mean and unit std.

        Returns:
            - data (np.ndarray): The input array scaled.
        """

        data = self.__scalers[source_name][scaling_type].inverse_transform(data)

        return data