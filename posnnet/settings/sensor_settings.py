from typing import Self, Union, List, Tuple, Any

from posnnet.setter_guards import (
    setter_typeguard, 
    setter_strictly_positive_integer_guard,
    setter_no_empty_string_or_none_guard
)
from posnnet.settings.base_settings import BaseSettings


class SensorSettings(BaseSettings):

    def __init__(
        self: Self
    ) -> None:

        """
        Initialize the sensor settings.

        Settings:
            - frequency (int): The frequency of the sensor in Hertz (number of timestamp by seconds). Have to be a strictly positive integer.
            - name_accelerometer_x (Union[str, None]): The name of the axe X of the accelerometer (name of the columns in the CSV files). Set to None if the axe does exist in the sensor.
            - name_accelerometer_y (Union[str, None]): The name of the axe Y of the accelerometer (name of the columns in the CSV files). Set to None if the axe does exist in the sensor.
            - name_accelerometer_z (Union[str, None]): The name of the axe Z of the accelerometer (name of the columns in the CSV files). Set to None if the axe does exist in the sensor.
            - name_gyroscope_x (Union[str, None]): The name of the axe X of the gyroscope (name of the columns in the CSV files). Set to None if the axe does exist in the sensor.
            - name_gyroscope_y (Union[str, None]): The name of the axe Y of the gyroscope (name of the columns in the CSV files). Set to None if the axe does exist in the sensor.
            - name_gyroscope_z (Union[str, None]): The name of the axe Z of the gyroscope (name of the columns in the CSV files). Set to None if the axe does exist in the sensor.
            - name_magnetometer_x (Union[str, None]): The name of the axe X of the magnetometer (name of the columns in the CSV files). Set to None if the axe does exist in the sensor.
            - name_magnetometer_y (Union[str, None]): The name of the axe Y of the magnetometer (name of the columns in the CSV files). Set to None if the axe does exist in the sensor.
            - name_magnetometer_z (Union[str, None]): The name of the axe Z of the magnetometer (name of the columns in the CSV files). Set to None if the axe does exist in the sensor.
            - name_velocity_gps_x (Union[str, None]): The name of the axe X of the GPS velocity (name of the columns in the CSV files). Set to None if the axe does exist in the sensor.
            - name_velocity_gps_y (Union[str, None]): The name of the axe Y of the GPS velocity (name of the columns in the CSV files). Set to None if the axe does exist in the sensor.
            - name_velocity_gps_z (Union[str, None]): The name of the axe Z of the GPS velocity (name of the columns in the CSV files). Set to None if the axe does exist in the sensor.
            - name_position_fusion_x (Union[str, None]): The name of the axe X of the fusion (output) position (name of the columns in the CSV files). Set to None if the axe does exist in the sensor.
            - name_position_fusion_y (Union[str, None]): The name of the axe Y of the fusion (output) position (name of the columns in the CSV files). Set to None if the axe does exist in the sensor.
            - name_position_fusion_z (Union[str, None]): The name of the axe Z of the fusion (output) position (name of the columns in the CSV files). Set to None if the axe does exist in the sensor.
            - name_velocity_fusion_x (Union[str, None]): The name of the axe X of the fusion (output) velocity (name of the columns in the CSV files). Set to None if the axe does exist in the sensor.
            - name_velocity_fusion_y (Union[str, None]): The name of the axe Y of the fusion (output) velocity (name of the columns in the CSV files). Set to None if the axe does exist in the sensor.
            - name_velocity_fusion_z (Union[str, None]): The name of the axe Z of the fusion (output) velocity (name of the columns in the CSV files). Set to None if the axe does exist in the sensor.
            - name_orientation_fusion_x (Union[str, None]): The name of the axe X of the fusion (output) orientation (name of the columns in the CSV files). Set to None if the axe does exist in the sensor.
            - name_orientation_fusion_y (Union[str, None]): The name of the axe Y of the fusion (output) orientation (name of the columns in the CSV files). Set to None if the axe does exist in the sensor.
            - name_orientation_fusion_z (Union[str, None]): The name of the axe Z of the fusion (output) orientation (name of the columns in the CSV files). Set to None if the axe does exist in the sensor.
        """

        self.frequency = 100
        self.name_accelerometer_x = None
        self.name_accelerometer_y = None
        self.name_accelerometer_z = None
        self.name_gyroscope_x = None
        self.name_gyroscope_y = None
        self.name_gyroscope_z = None
        self.name_magnetometer_x = None
        self.name_magnetometer_y = None
        self.name_magnetometer_z = None
        self.name_velocity_gps_x = None
        self.name_velocity_gps_y = None
        self.name_velocity_gps_z = None
        self.name_position_fusion_x = None
        self.name_position_fusion_y = None
        self.name_position_fusion_z = None
        self.name_velocity_fusion_x = None
        self.name_velocity_fusion_y = None
        self.name_velocity_fusion_z = None
        self.name_orientation_fusion_x = None
        self.name_orientation_fusion_y = None
        self.name_orientation_fusion_z = None

    @property
    def frequency(
        self: Self
    ) -> int:

        return self._frequency

    @frequency.setter
    @setter_typeguard
    @setter_strictly_positive_integer_guard
    def frequency(
        self: Self,
        frequency: int
    ) -> None:

        self._frequency = frequency

    @property
    def name_accelerometer_x(
        self: Self
    ) -> Union[str, None]:

        return self._name_accelerometer_x

    @name_accelerometer_x.setter
    @setter_typeguard
    @setter_no_empty_string_or_none_guard
    def name_accelerometer_x(
        self: Self,
        name_accelerometer_x: Union[str, None]
    ) -> None:

        self._name_accelerometer_x = name_accelerometer_x

    @property
    def name_accelerometer_y(
        self: Self
    ) -> Union[str, None]:

        return self._name_accelerometer_y

    @name_accelerometer_y.setter
    @setter_typeguard
    @setter_no_empty_string_or_none_guard
    def name_accelerometer_y(
        self: Self,
        name_accelerometer_y: Union[str, None]
    ) -> None:

        self._name_accelerometer_y = name_accelerometer_y

    @property
    def name_accelerometer_z(
        self: Self
    ) -> Union[str, None]:

        return self._name_accelerometer_z

    @name_accelerometer_z.setter
    @setter_typeguard
    @setter_no_empty_string_or_none_guard
    def name_accelerometer_z(
        self: Self,
        name_accelerometer_z: Union[str, None]
    ) -> None:

        self._name_accelerometer_z = name_accelerometer_z
        
    @property
    def name_gyroscope_x(
        self: Self
    ) -> Union[str, None]:

        return self._name_gyroscope_x

    @name_gyroscope_x.setter
    @setter_typeguard
    @setter_no_empty_string_or_none_guard
    def name_gyroscope_x(
        self: Self,
        name_gyroscope_x: Union[str, None]
    ) -> None:

        self._name_gyroscope_x = name_gyroscope_x

    @property
    def name_gyroscope_y(
        self: Self
    ) -> Union[str, None]:

        return self._name_gyroscope_y

    @name_gyroscope_y.setter
    @setter_typeguard
    @setter_no_empty_string_or_none_guard
    def name_gyroscope_y(
        self: Self,
        name_gyroscope_y: Union[str, None]
    ) -> None:

        self._name_gyroscope_y = name_gyroscope_y

    @property
    def name_gyroscope_z(
        self: Self
    ) -> Union[str, None]:

        return self._name_gyroscope_z

    @name_gyroscope_z.setter
    @setter_typeguard
    @setter_no_empty_string_or_none_guard
    def name_gyroscope_z(
        self: Self,
        name_gyroscope_z: Union[str, None]
    ) -> None:

        self._name_gyroscope_z = name_gyroscope_z

    @property
    def name_magnetometer_x(
        self: Self
    ) -> Union[str, None]:

        return self._name_magnetometer_x

    @name_magnetometer_x.setter
    @setter_typeguard
    @setter_no_empty_string_or_none_guard
    def name_magnetometer_x(
        self: Self,
        name_magnetometer_x: Union[str, None]
    ) -> None:

        self._name_magnetometer_x = name_magnetometer_x

    @property
    def name_magnetometer_y(
        self: Self
    ) -> Union[str, None]:

        return self._name_magnetometer_y

    @name_magnetometer_y.setter
    @setter_typeguard
    @setter_no_empty_string_or_none_guard
    def name_magnetometer_y(
        self: Self,
        name_magnetometer_y: Union[str, None]
    ) -> None:

        self._name_magnetometer_y = name_magnetometer_y

    @property
    def name_magnetometer_z(
        self: Self
    ) -> Union[str, None]:

        return self._name_magnetometer_z

    @name_magnetometer_z.setter
    @setter_typeguard
    @setter_no_empty_string_or_none_guard
    def name_magnetometer_z(
        self: Self,
        name_magnetometer_z: Union[str, None]
    ) -> None:

        self._name_magnetometer_z = name_magnetometer_z

    @property
    def name_velocity_gps_x(
        self: Self
    ) -> Union[str, None]:

        return self._name_velocity_gps_x

    @name_velocity_gps_x.setter
    @setter_typeguard
    @setter_no_empty_string_or_none_guard
    def name_velocity_gps_x(
        self: Self,
        name_velocity_gps_x: Union[str, None]
    ) -> None:

        self._name_velocity_gps_x = name_velocity_gps_x

    @property
    def name_velocity_gps_y(
        self: Self
    ) -> Union[str, None]:

        return self._name_velocity_gps_y

    @name_velocity_gps_y.setter
    @setter_typeguard
    @setter_no_empty_string_or_none_guard
    def name_velocity_gps_y(
        self: Self,
        name_velocity_gps_y: Union[str, None]
    ) -> None:

        self._name_velocity_gps_y = name_velocity_gps_y

    @property
    def name_velocity_gps_z(
        self: Self
    ) -> Union[str, None]:

        return self._name_velocity_gps_z

    @name_velocity_gps_z.setter
    @setter_typeguard
    @setter_no_empty_string_or_none_guard
    def name_velocity_gps_z(
        self: Self,
        name_velocity_gps_z: Union[str, None]
    ) -> None:

        self._name_velocity_gps_z = name_velocity_gps_z     
        
    @property
    def name_position_fusion_x(
        self: Self
    ) -> Union[str, None]:

        return self._name_position_fusion_x

    @name_position_fusion_x.setter
    @setter_typeguard
    @setter_no_empty_string_or_none_guard
    def name_position_fusion_x(
        self: Self,
        name_position_fusion_x: Union[str, None]
    ) -> None:

        self._name_position_fusion_x = name_position_fusion_x

    @property
    def name_position_fusion_y(
        self: Self
    ) -> Union[str, None]:

        return self._name_position_fusion_y

    @name_position_fusion_y.setter
    @setter_typeguard
    @setter_no_empty_string_or_none_guard
    def name_position_fusion_y(
        self: Self,
        name_position_fusion_y: Union[str, None]
    ) -> None:

        self._name_position_fusion_y = name_position_fusion_y

    @property
    def name_position_fusion_z(
        self: Self
    ) -> Union[str, None]:

        return self._name_position_fusion_z

    @name_position_fusion_z.setter
    @setter_typeguard
    @setter_no_empty_string_or_none_guard
    def name_position_fusion_z(
        self: Self,
        name_position_fusion_z: Union[str, None]
    ) -> None:

        self._name_position_fusion_z = name_position_fusion_z
        
    @property
    def name_velocity_fusion_x(
        self: Self
    ) -> Union[str, None]:

        return self._name_velocity_fusion_x

    @name_velocity_fusion_x.setter
    @setter_typeguard
    @setter_no_empty_string_or_none_guard
    def name_velocity_fusion_x(
        self: Self,
        name_velocity_fusion_x: Union[str, None]
    ) -> None:

        self._name_velocity_fusion_x = name_velocity_fusion_x

    @property
    def name_velocity_fusion_y(
        self: Self
    ) -> Union[str, None]:

        return self._name_velocity_fusion_y

    @name_velocity_fusion_y.setter
    @setter_typeguard
    @setter_no_empty_string_or_none_guard
    def name_velocity_fusion_y(
        self: Self,
        name_velocity_fusion_y: Union[str, None]
    ) -> None:

        self._name_velocity_fusion_y = name_velocity_fusion_y

    @property
    def name_velocity_fusion_z(
        self: Self
    ) -> Union[str, None]:

        return self._name_velocity_fusion_z

    @name_velocity_fusion_z.setter
    @setter_typeguard
    @setter_no_empty_string_or_none_guard
    def name_velocity_fusion_z(
        self: Self,
        name_velocity_fusion_z: Union[str, None]
    ) -> None:

        self._name_velocity_fusion_z = name_velocity_fusion_z

    @property
    def name_orientation_fusion_x(
        self: Self
    ) -> Union[str, None]:

        return self._name_orientation_fusion_x

    @name_orientation_fusion_x.setter
    @setter_typeguard
    @setter_no_empty_string_or_none_guard
    def name_orientation_fusion_x(
        self: Self,
        name_orientation_fusion_x: Union[str, None]
    ) -> None:

        self._name_orientation_fusion_x = name_orientation_fusion_x

    @property
    def name_orientation_fusion_y(
        self: Self
    ) -> Union[str, None]:

        return self._name_orientation_fusion_y

    @name_orientation_fusion_y.setter
    @setter_typeguard
    @setter_no_empty_string_or_none_guard
    def name_orientation_fusion_y(
        self: Self,
        name_orientation_fusion_y: Union[str, None]
    ) -> None:

        self._name_orientation_fusion_y = name_orientation_fusion_y

    @property
    def name_orientation_fusion_z(
        self: Self
    ) -> Union[str, None]:

        return self._name_orientation_fusion_z

    @name_orientation_fusion_z.setter
    @setter_typeguard
    @setter_no_empty_string_or_none_guard
    def name_orientation_fusion_z(
        self: Self,
        name_orientation_fusion_z: Union[str, None]
    ) -> None:

        self._name_orientation_fusion_z = name_orientation_fusion_z

    def get_features_name_accelerometer(
        self: Self
    ) -> List[str]:

        """
        Return the list of the features name for the accelerometer.

        Returns:
            - features_name_accelerometer (List[str]): The list of the features name for the accelerometer.
        """

        features_name_accelerometer = [
            feature_name
            for feature_name in [self._name_accelerometer_x, self._name_accelerometer_y, self._name_accelerometer_z]
            if feature_name is not None
        ]

        return features_name_accelerometer

    def get_features_name_gyroscope(
        self: Self
    ) -> List[str]:

        """
        Return the list of the features name for the gyroscope.

        Returns:
            - features_name_gyroscope (List[str]): The list of the features name for the gyroscope.
        """

        features_name_gyroscope = [
            feature_name
            for feature_name in [self._name_gyroscope_x, self._name_gyroscope_y, self._name_gyroscope_z]
            if feature_name is not None
        ]

        return features_name_gyroscope

    def get_features_name_magnetometer(
        self: Self
    ) -> List[str]:

        """
        Return the list of the features name for the magnetometer.

        Returns:
            - features_name_magnetometer (List[str]): The list of the features name for the magnetometer.
        """

        features_name_magnetometer = [
            feature_name
            for feature_name in [self._name_magnetometer_x, self._name_magnetometer_y, self._name_magnetometer_z]
            if feature_name is not None
        ]

        return features_name_magnetometer

    def get_features_name_velocity_gps(
        self: Self
    ) -> List[str]:

        """
        Return the list of the features name for the GPS velocity.

        Returns:
            - features_name_velocity_gps (List[str]): The list of the features name for the GPS velocity.
        """

        features_name_velocity_gps = [
            feature_name
            for feature_name in [self._name_velocity_gps_x, self._name_velocity_gps_y, self._name_velocity_gps_z]
            if feature_name is not None
        ]

        return features_name_velocity_gps

    def get_features_name_position_fusion(
        self: Self
    ) -> List[str]:

        """
        Return the list of the features name for the fusion (output) position.

        Returns:
            - features_name_position_fusion (List[str]): The list of the features name for the fusion (output) position.
        """

        features_name_position_fusion = [
            feature_name
            for feature_name in [self._name_position_fusion_x, self._name_position_fusion_y, self._name_position_fusion_z]
            if feature_name is not None
        ]

        return features_name_position_fusion

    def get_features_name_velocity_fusion(
        self: Self
    ) -> List[str]:

        """
        Return the list of the features name for the fusion (output) velocity.

        Returns:
            - features_name_velocity_fusion (List[str]): The list of the features name for the fusion (output) velocity.
        """

        features_name_velocity_fusion = [
            feature_name
            for feature_name in [self._name_velocity_fusion_x, self._name_velocity_fusion_y, self._name_velocity_fusion_z]
            if feature_name is not None
        ]

        return features_name_velocity_fusion

    def get_features_name_orientation_fusion(
        self: Self
    ) -> List[str]:

        """
        Return the list of the features name for the fusion (output) position.

        Returns:
            - features_name_orientation_fusion (List[str]): The list of the features name for the fusion (output) position.
        """

        features_name_orientation_fusion = [
            feature_name
            for feature_name in [self._name_orientation_fusion_x, self._name_orientation_fusion_y, self._name_orientation_fusion_z]
            if feature_name is not None
        ]

        return features_name_orientation_fusion

    def get_features_name(
        self: Self
    ) -> List[str]:

        """
        Return the list of the features name (accelerometer, gyroscope, magnetometer, GPS velocity, fusion position, fusion velocity and fusion orientation).

        Returns:
            - features_name (List[str]): The list of the features name.
        """

        features_name = self.get_features_name_accelerometer()
        features_name += self.get_features_name_gyroscope()
        features_name += self.get_features_name_magnetometer()
        features_name += self.get_features_name_velocity_gps()
        features_name += self.get_features_name_position_fusion()
        features_name += self.get_features_name_velocity_fusion()
        features_name += self.get_features_name_orientation_fusion()

        return features_name

    def validate_settings(
        self: Self,
        **kwargs: Any
    ) -> Tuple[List[str], List[str]]:

        """
        Validate the value of the different setting parameters.

        Returns:
            - warnings (List[str]): A list of warnings on the different setting parameter values.
            - errors (List[str]): A list of errors on the different setting parameter values.
        """

        warnings, errors = [], []

        # If no name is provided for the axe X of the accelerometer, generate a warning.
        if self.name_accelerometer_x is None:
            warnings.append("You do not have provided a name for the axe X of the accelerometer. Ignore this message if your sensor does not have it.")

        # If no name is provided for the axe Y of the accelerometer, generate a warning.
        if self.name_accelerometer_y is None:
            warnings.append("You do not have provided a name for the axe Y of the accelerometer. Ignore this message if your sensor does not have it.")

        # If no name is provided for the axe Z of the accelerometer, generate a warning.
        if self.name_accelerometer_z is None:
            warnings.append("You do not have provided a name for the axe Z of the accelerometer. Ignore this message if your sensor does not have it.")

        # If no name is provided for the axe X of the gyroscope, generate a warning.
        if self.name_gyroscope_x is None:
            warnings.append("You do not have provided a name for the axe X of the gyroscope. Ignore this message if your sensor does not have it.")

        # If no name is provided for the axe Y of the gyroscope, generate a warning.
        if self.name_gyroscope_y is None:
            warnings.append("You do not have provided a name for the axe Y of the gyroscope. Ignore this message if your sensor does not have it.")

        # If no name is provided for the axe Z of the gyroscope, generate a warning.
        if self.name_gyroscope_z is None:
            warnings.append("You do not have provided a name for the axe Z of the gyroscope. Ignore this message if your sensor does not have it.")

        # If no name is provided for the axe X of the magnetometer, generate a warning.
        if self.name_magnetometer_x is None:
            warnings.append("You do not have provided a name for the axe X of the magnetometer. Ignore this message if your sensor does not have it.")

        # If no name is provided for the axe Y of the magnetometer, generate a warning.
        if self.name_magnetometer_y is None:
            warnings.append("You do not have provided a name for the axe Y of the magnetometer. Ignore this message if your sensor does not have it.")

        # If no name is provided for the axe Z of the magnetometer, generate a warning.
        if self.name_magnetometer_z is None:
            warnings.append("You do not have provided a name for the axe Z of the magnetometer. Ignore this message if your sensor does not have it.")

        # If no name is provided for the axe X of the GPS velocity, generate a warning.
        if self.name_velocity_gps_x is None:
            warnings.append("You do not have provided a name for the axe X of the GPS velocity. Ignore this message if your sensor does not have it.")

        # If no name is provided for the axe Y of the GPS velocity, generate a warning.
        if self.name_velocity_gps_y is None:
            warnings.append("You do not have provided a name for the axe Y of the GPS velocity. Ignore this message if your sensor does not have it.")

        # If no name is provided for the axe Z of the GPS velocity, generate a warning.
        if self.name_velocity_gps_z is None:
            warnings.append("You do not have provided a name for the axe Z of the GPS velocity. Ignore this message if your sensor does not have it.")

        # If no name is provided for the axe X of the fusion position, generate a warning.
        if self.name_position_fusion_x is None:
            warnings.append("You do not have provided a name for the axe X of the fusion position. Ignore this message if your sensor does not have it.")

        # If no name is provided for the axe Y of the fusion position, generate a warning.
        if self.name_position_fusion_y is None:
            warnings.append("You do not have provided a name for the axe Y of the fusion position. Ignore this message if your sensor does not have it.")

        # If no name is provided for the axe Z of the fusion position, generate a warning.
        if self.name_position_fusion_z is None:
            warnings.append("You do not have provided a name for the axe Z of the fusion position. Ignore this message if your sensor does not have it.")

        # If no name is provided for the axe X of the fusion velocity, generate a warning.
        if self.name_velocity_fusion_x is None:
            warnings.append("You do not have provided a name for the axe X of the fusion velocity. Ignore this message if your sensor does not have it.")

        # If no name is provided for the axe Y of the fusion velocity, generate a warning.
        if self.name_velocity_fusion_y is None:
            warnings.append("You do not have provided a name for the axe Y of the fusion velocity. Ignore this message if your sensor does not have it.")

        # If no name is provided for the axe Z of the fusion velocity, generate a warning.
        if self.name_velocity_fusion_z is None:
            warnings.append("You do not have provided a name for the axe Z of the fusion velocity. Ignore this message if your sensor does not have it.")

        # If no name is provided for the axe X of the fusion orientation, generate a warning.
        if self.name_orientation_fusion_x is None:
            warnings.append("You do not have provided a name for the axe X of the fusion orientation. Ignore this message if your sensor does not have it.")

        # If no name is provided for the axe Y of the fusion orientation, generate a warning.
        if self.name_orientation_fusion_y is None:
            warnings.append("You do not have provided a name for the axe Y of the fusion orientation. Ignore this message if your sensor does not have it.")

        # If no name is provided for the axe Z of the fusion orientation, generate a warning.
        if self.name_orientation_fusion_z is None:
            warnings.append("You do not have provided a name for the axe Z of the fusion orientation. Ignore this message if your sensor does not have it.")

        # If no IMU information is provided, generate an error.
        if (
            self.name_accelerometer_x is None and 
            self.name_accelerometer_y is None and 
            self.name_accelerometer_z is None and
            self.name_gyroscope_x is None and 
            self.name_gyroscope_y is None and 
            self.name_gyroscope_z is None and
            self.name_accelerometer_x is None and 
            self.name_accelerometer_y is None and 
            self.name_accelerometer_z is None
        ):
            errors.append("You do not have provided any name for IMU (accelerometer, gyroscope, magnetometer). You need at least one axe of one of them.")

        if (
            self.name_position_fusion_x is None and
            self.name_position_fusion_y is None and
            self.name_position_fusion_z is None
        ):
            errors.append("You do not have provided any name for the fusion position. You need at least one axe.")

        if (
            self.name_velocity_fusion_x is None and
            self.name_velocity_fusion_y is None and
            self.name_velocity_fusion_z is None
        ):
            errors.append("You do not have provided any name for the fusion velocity. You need at least one axe.")

        if (
            (self.name_position_fusion_x is None and self.name_velocity_fusion_x is not None) or
            (self.name_position_fusion_x is not None and self.name_velocity_fusion_x is None)
        ):
            errors.append("If you want to use the axe X of the fusion, you have to provide both the position and the velocity.")

        if (
            (self.name_position_fusion_y is None and self.name_velocity_fusion_y is not None) or
            (self.name_position_fusion_y is not None and self.name_velocity_fusion_y is None)
        ):
            errors.append("If you want to use the axe Y of the fusion, you have to provide both the position and the velocity.")

        if (
            (self.name_position_fusion_z is None and self.name_velocity_fusion_z is not None) or
            (self.name_position_fusion_z is not None and self.name_velocity_fusion_z is None)
        ):
            errors.append("If you want to use the axe Z of the fusion, you have to provide both the position and the velocity.")

        return warnings, errors