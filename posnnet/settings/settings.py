from itertools import chain
import pathlib
from typing import Self, Any, Tuple, List

from posnnet.settings.csv_files_settings import CsvFilesSettings
from posnnet.settings.dataset_settings import DatasetSettings
from posnnet.settings.general_settings import GeneralSettings
from posnnet.settings.objective_settings import ObjectiveSettings
from posnnet.settings.sensor_settings import SensorSettings
from posnnet.settings.training_settings import TrainingSettings
from posnnet.settings.tuning_settings import TuningSettings


class Settings:

    def __init__(
        self: Self
    ) -> None:

        self.general = GeneralSettings()
        self.csv_files = CsvFilesSettings()
        self.dataset = DatasetSettings()
        self.training = TrainingSettings()
        self.tuning = TuningSettings()
        self.objective = ObjectiveSettings()
        self.sensor = SensorSettings()

    def print_general_settings(
        self: Self
    ) -> None:

        self.general.print_settings()

    def print_csv_files_settings(
        self: Self
    ) -> None:

        self.csv_files.print_settings()

    def print_dataset_settings(
        self: Self
    ) -> None:

        self.dataset.print_settings()

    def print_training_settings(
        self: Self
    ) -> None:

        self.training.print_settings()

    def print_tuning_settings(
        self: Self
    ) -> None:

        self.tuning.print_settings()

    def print_objective_settings(
        self: Self
    ) -> None:

        self.objective.print_settings()

    def print_sensor_settings(
        self: Self
    ) -> None:

        self.sensor.print_settings()

    def set_general_settings(
        self: Self,
        **kwargs: Any
    ) -> None:

        self.general.set_settings(**kwargs)

    def set_csv_files_settings(
        self: Self,
        **kwargs: Any
    ) -> None:

        self.csv_files.set_settings(**kwargs)

    def set_dataset_settings(
        self: Self,
        **kwargs: Any
    ) -> None:

        self.dataset.set_settings(**kwargs)

    def set_training_settings(
        self: Self,
        **kwargs: Any
    ) -> None:

        self.training.set_settings(**kwargs)

    def set_tuning_settings(
        self: Self,
        **kwargs: Any
    ) -> None:

        self.tuning.set_settings(**kwargs)

    def set_objective_settings(
        self: Self,
        **kwargs: Any
    ) -> None:

        self.objective.set_settings(**kwargs)

    def set_sensor_settings(
        self: Self,
        **kwargs: Any
    ) -> None:

        self.sensor.set_settings(**kwargs)

    def validate_settings(
        self: Self,
        path_raw_data: pathlib.Path
    ) -> Tuple[List[str], List[str]]:

        """
        Validate the value of the different setting parameters.

        Args:
            - path_raw_data (pathlib.Path): The path where raw data are stored.

        Returns:
            - warnings (List[str]): A list of warnings on the different setting parameter values.
            - errors (List[str]): A list of errors on the different setting parameter values.
        """

        warnings_errors_generator = (
            validate_setting_submethod(path_raw_data=path_raw_data) 
            for validate_setting_submethod in [
                self.general.validate_settings, self.csv_files.validate_settings, self.dataset.validate_settings,
                self.training.validate_settings, self.tuning.validate_settings, self.objective.validate_settings,
                self.sensor.validate_settings
            ]
        )

        warnings, errors = map(
            lambda x: list(chain.from_iterable(x)),
            zip( * warnings_errors_generator)
        )

        return warnings, errors