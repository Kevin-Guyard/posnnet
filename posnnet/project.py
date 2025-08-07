import joblib
import pandas as pd
import pathlib
import subprocess
from typing import Self, Any, Union, Tuple, List
import warnings

from posnnet.data.data_preprocessor import DataPreprocessor
from posnnet.data.scalers import Scalers
from posnnet.evaluation.evaluation_manager import EvaluationManager
from posnnet.input_guards import (
    change_project_name_input_guard, 
    evaluate_configuration_input_guard,
    init_input_guard, 
    move_to_previous_stage_input_guard,
    preprocess_data_input_guard,
    run_tuning_input_guard,
    validate_models_tuning_input_guard,
    validate_settings_input_guard,
    averaging_configuration_input_guard
)
from posnnet.project_life_cycle_manager import ProjectLifeCycleManager
from posnnet.settings import Settings
from posnnet.tuning_manager import TuningManager


class Project:

    @init_input_guard
    def __init__(
        self: Self,
        project_name: str
    ) -> None:

        """
        Load an existing project or create a new one.

        Args:
            - project_name (str): The name of the project.
        """

        # Memorize project name.
        self.__project_name = project_name

        # If the project already exists, load it, else create a new project.
        if pathlib.Path(f"./project_{project_name:s}/").exists():
            self.__load_project(project_name=project_name)
            print(f"Project {project_name} successfully loaded.")
        else:
            self.__create_project(project_name=project_name)
            print(f"Project {project_name} successfully created.")

        # Disable useless warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

    def __load_project(
        self: Self,
        project_name: str
    ) -> None:

        """
        Load an existing project.

        Args:
            - project_name (str): The name of the project.
        """

        self.__settings = joblib.load(filename=pathlib.Path(f"./project_{project_name:s}/settings.pkl"))
        self._life_cycle_manager = joblib.load(filename=pathlib.Path(f"./project_{project_name:s}/life_cycle_manager.pkl"))

    def __create_project(
        self: Self,
        project_name: str
    ) -> None:

        """
        Create a new project.

        Args:
            - project_name (str): The name of the project.
        """

        # Initialize default settings.
        self.__settings = Settings()
        self._life_cycle_manager = ProjectLifeCycleManager()

        # Initialize raw data directory.
        pathlib.Path(f"./project_{project_name:s}/data/raw/").mkdir(exist_ok=False, parents=True)

        # Initialize preprocessed data directories (training, validation and evaluation).
        pathlib.Path(f"./project_{project_name:s}/data/preprocessed/training/").mkdir(exist_ok=False, parents=True)
        pathlib.Path(f"./project_{project_name:s}/data/preprocessed/validation/").mkdir(exist_ok=False, parents=True)
        pathlib.Path(f"./project_{project_name:s}/data/preprocessed/evaluation/").mkdir(exist_ok=False, parents=True)

        # Initialize temporary files directory.
        pathlib.Path(f"./project_{project_name:s}/temp/").mkdir(exist_ok=False)

        # Initialize tuning results directory.
        pathlib.Path(f"./project_{project_name:s}/tuning/").mkdir(exist_ok=False)

        # Initialize study directory.
        pathlib.Path(f"./project_{project_name:s}/studies/").mkdir(exist_ok=False)

        # Initialize evaluation directory.
        pathlib.Path(f"./project_{project_name:s}/evaluation/").mkdir(exist_ok=False)

        # Initialize external directory.
        pathlib.Path(f"./project_{project_name:s}/external/").mkdir(exist_ok=False)

        # Save project default settings and life cycle manager.
        joblib.dump(value=self.__settings, filename=f"./project_{project_name:s}/settings.pkl")
        joblib.dump(value=self._life_cycle_manager, filename=f"./project_{project_name:s}/life_cycle_manager.pkl")

    @change_project_name_input_guard
    def change_project_name(
        self: Self,
        new_project_name: str
    ) -> None:

        """
        Change the project name.

        Args:
            - new_project_name (str): The new name for the project.
        """

        pathlib.Path(f"./project_{self.__project_name:s}/").rename(f"project_{new_project_name}")
        self.__project_name = new_project_name

    @move_to_previous_stage_input_guard
    def move_to_previous_stage(
        self: Self,
        desired_stage: int,
        force_move: bool=False
    ) -> None:

        """
        Move back the actual stage of the life cycle to the desired stage.

        Args:
            - desired_stage (int): The desired stage to which come back :
                * ProjectLifeCycleManager.SETTINGS_DEFINITION_STAGE
                * ProjectLifeCycleManager.DATA_PREPROCESSING_STAGE
                * ProjectLifeCycleManager.SCALER_FIT_STAGE
                * ProjectLifeCycleManager.MODELS_TUNING_STAGE
            - force_move (bool): Either to force the mouvement to a previous stage (True) or raise a warning a do not take action (False) in case of destructive stage transition. Default = False.
        """

        self._life_cycle_manager.move_to_previous_stage(
            desired_stage=desired_stage,
            path_preprocessed_training_data=pathlib.Path(f"./project_{self.__project_name:s}/data/preprocessed/training/"),
            path_preprocessed_validation_data=pathlib.Path(f"./project_{self.__project_name:s}/data/preprocessed/validation/"),
            path_preprocessed_evaluation_data=pathlib.Path(f"./project_{self.__project_name:s}/data/preprocessed/evaluation/"),
            path_scalers=pathlib.Path(f"./project_{self.__project_name:s}/"),
            path_evaluation_results=pathlib.Path(f"./project_{self.__project_name:s}/evaluation/"),
            path_tuning_results=pathlib.Path(f"./project_{self.__project_name:s}/tuning/"),
            path_studies=pathlib.Path(f"./project_{self.__project_name:s}/studies/"),
            path_temp=pathlib.Path(f"./project_{self.__project_name:s}/temp/"),
            force_move=force_move
        )

        # Save life cycle manager
        joblib.dump(value=self._life_cycle_manager, filename=f"./project_{self.__project_name:s}/life_cycle_manager.pkl")

    def print_general_settings(
        self: Self
    ) -> None:

        """
        Print the general settings:
            - random_seed (int)
            - dtype (str)
            - device (torch.device)
        """

        self.__settings.print_general_settings()

    def print_csv_files_settings(
        self: Self
    ) -> None:

        """
        Print the CSV files settings:
            - csv_sep (str)
            - csv_encoding (str)
        """

        self.__settings.print_csv_files_settings()

    def print_dataset_settings(
        self: Self
    ) -> None:

        """
        Print the dataset settings:
            - n_duplications_per_eval_session (Union[int, str])
            - sessions_id_training (List[int])
            - sessions_id_validation (List[int])
            - sessions_id_evaluation (List[int])
        """

        self.__settings.print_dataset_settings()

    def print_training_settings(
        self: Self
    ) -> None:

        """
        Print the training settings:
            - coeff_sampling_training (int)
            - coeff_sampling_validation (int)
            - n_epochs_sampling (int)
            - n_epochs (int)
            - patience (Union[int, None])
            - num_workers (int)
            - use_mixed_precision (bool)
            - n_epochs_training_checkpoint (int)
        """

        self.__settings.print_training_settings()

    def print_tuning_settings(
        self: Self
    ) -> None:

        """
        Print the tuning settings:
            - n_startup_trials (int)
        """

        self.__settings.print_tuning_settings()

    def print_objective_settings(
        self: Self
    ) -> None:

        """
        Print the objective settings:
            - n_minimum_seconds_operational_gps (int)
            - gps_outage_durations_eval_at_beginning (List[Tuple[str, str]])
            - gps_outage_durations_eval_within (List[Tuple[str, str]])
            - gps_outage_durations_eval_at_end (List[Tuple[str, str]])
        """

        self.__settings.print_objective_settings()

    def print_sensor_settings(
        self: Self
    ) -> None:

        """
        Print the sensor settings:
            - frequency (int)
            - name_accelerometer_x (Union[str, None])
            - name_accelerometer_y (Union[str, None])
            - name_accelerometer_z (Union[str, None])
            - name_gyroscope_x (Union[str, None])
            - name_gyroscope_y (Union[str, None])
            - name_gyroscope_z (Union[str, None])
            - name_magnetometer_x (Union[str, None])
            - name_magnetometer_y (Union[str, None])
            - name_magnetometer_z (Union[str, None])
            - name_velocity_gps_x (Union[str, None])
            - name_velocity_gps_y (Union[str, None])
            - name_velocity_gps_z (Union[str, None])
            - name_position_fusion_x (Union[str, None])
            - name_position_fusion_y (Union[str, None])
            - name_position_fusion_z (Union[str, None])
            - name_velocity_fusion_x (Union[str, None])
            - name_velocity_fusion_y (Union[str, None])
            - name_velocity_fusion_z (Union[str, None])
            - name_orientation_fusion_x (Union[str, None])
            - name_orientation_fusion_y (Union[str, None])
            - name_orientation_fusion_z (Union[str, None])
        """

        self.__settings.print_sensor_settings()

    @ProjectLifeCycleManager.settings_definition_stage_guard
    def set_general_settings(
        self: Self,
        **kwargs: Any
    ) -> None:

        """
        Set the general settings. 
        NB: The function do not required to receive every parameter. Parameter not present will not be modified.

        Args:
            - random_seed (int): A positive integer that is used to ensure determinism of the framework.
            - dtype (str): The dtype of the models and the data (have to be either 'float16', 'float32' or 'float64').
            - device (str): The device on which perform the training and the inference (have to be either 'cuda' for GPU, 'cpu' for CPU or 'auto'
                            in which case the framework select the GPU if available else the CPU).
        """

        self.__settings.set_general_settings(**kwargs)

        # Save project new settings.
        joblib.dump(value=self.__settings, filename=f"./project_{self.__project_name:s}/settings.pkl")

    @ProjectLifeCycleManager.settings_definition_stage_guard
    def set_csv_files_settings(
        self: Self,
        **kwargs: Any
    ) -> None:

        """
        Set the CSV files settings. 
        NB: The function do not required to receive every parameter. Parameter not present will not be modified.

        Args:
            - csv_sep (str): The character used to delimit columns in the CSV files (default = ',').
            - csv_encoding (str): The encoding used to encode the data in the CSV files (default = 'utf-8').
        """

        self.__settings.set_csv_files_settings(**kwargs)

        # Save project new settings.
        joblib.dump(value=self.__settings, filename=f"./project_{self.__project_name:s}/settings.pkl")

    @ProjectLifeCycleManager.settings_definition_stage_guard
    def set_dataset_settings(
        self: Self,
        **kwargs: Any
    ) -> None:

        """
        Set the dataset settings. 
        NB: The function do not required to receive every parameter. Parameter not present will not be modified.

        Args:
            - n_duplications_per_eval_session (Union[int, str]): The number of duplications per evaluation session. If an integer is provided (strictly positive), every session is duplicated the same amount of time.
                                                                 If 'light', 'normal' or 'extensive' is provided, the framework compute an optimized number of duplication different for every session.
            - sessions_id_training (List[int]): The list of the id of the sessions that will be used for the training dataset. Id have to be strictly positive integer.
            - sessions_id_validation (List[int]): The list of the id of the sessions that will be used for the validation dataset. Id have to be strictly positive integer.
            - sessions_id_evaluation (List[int]): The list of the id of the sessions that will be used for the evaluation dataset. Id have to be strictly positive integer.
        """

        self.__settings.set_dataset_settings(**kwargs)

        # Save project new settings.
        joblib.dump(value=self.__settings, filename=f"./project_{self.__project_name:s}/settings.pkl")

    @ProjectLifeCycleManager.settings_definition_stage_guard
    def set_training_settings(
        self: Self,
        **kwargs: Any
    ) -> None:

        """
        Set the training settings. 
        NB: The function do not required to receive every parameter. Parameter not present will not be modified.

        Args:
            - coeff_sampling_training (int): The ratio of sample that will be used as window beginning every epoch for training (e.g. coeff_sampling = 1000 means 1/1000 
                                             of the dataset sample will be used to create a window). Have to be a strictly positive integer
            - coeff_sampling_validation (int): The ratio of sample that will be used as window beginning every epoch for validation (e.g. coeff_sampling = 1000 means 1/1000
                                               of the dataset sample will be used to create a window). Have to be a strictly positive integer.
            - n_epochs_sampling (int): The number of epochs for a complete sampling rotation of the dataset. Have to be a strictly positive integer.
            - n_epochs (int): The number of epochs for which train the models (if early stopping is not triggered). Have to be a strictly positive integer.
            - patience (Union[int, None]): The number of epochs without validation results improvement before early stopping training.
                                           If None, training is never early stopped. If an integer is provided, it has to be strictly positive.
            - num_workers (int): The number of process that will instanciate a dataloader instance to load the data in parallel. 0 means that
                                 the dataloader will be loaded in the main process. A rule of thumb is to use 4 processes by GPU. Have to be a positive integer.
            - use_mixed_precision (bool): Either to use or not float16 mixed precision for training.
            - n_epochs_training_checkpoint (int): The number of epochs of training after which to save a checkpoint. Have to be a strictly positive integer.
        """

        self.__settings.set_training_settings(**kwargs)

        # Save project new settings.
        joblib.dump(value=self.__settings, filename=f"./project_{self.__project_name:s}/settings.pkl")

    @ProjectLifeCycleManager.settings_definition_stage_guard
    def set_tuning_settings(
        self: Self,
        **kwargs: Any
    ) -> None:

        """
        Set the tuning settings. 
        NB: The function do not required to receive every parameter. Parameter not present will not be modified.

        Args:
            - n_startup_trials (int): The number of experiments randomly sampled before using Bayesian optimization to search in the hyperparameter space.
                                      Have to be a strictly positive integer.
        """

        self.__settings.set_tuning_settings(**kwargs)

        # Save project new settings.
        joblib.dump(value=self.__settings, filename=f"./project_{self.__project_name:s}/settings.pkl")

    @ProjectLifeCycleManager.settings_definition_stage_guard
    def set_objective_settings(
        self: Self,
        **kwargs: Any
    ) -> None:

        """
        Set the objective settings. 
        NB: The function do not required to receive every parameter. Parameter not present will not be modified.

        Args:
            - n_minimum_seconds_operational_gps (int): The minimum number of seconds of operational GPS that can be ensured next to every GPS outage.
                                                       Have to be a strictly positive integer
            - gps_outage_durations_eval_at_beginning (List[Tuple[str, str]]): The list of cases of GPS outage durations that will be simulated at the beginning, 
                                                                              following the format [(N1, N2), (N3, N4), ...] to simulate GPS outage from 
                                                                              N1 to N2 seconds and N3 to N4 seconds etc. Format of Nx: "MM:SS".
            - gps_outage_durations_eval_within (List[Tuple[str, str]]): The list of cases of GPS outage durations that will be simulated within the session, 
                                                                        following the format [(N1, N2), (N3, N4), ...] to simulate GPS outage from 
                                                                        N1 to N2 seconds and N3 to N4 seconds etc. Format of Nx: "MM:SS".
            - gps_outage_durations_eval_at_end (List[Tuple[str, str]]): The list of cases of GPS outage durations that will be simulated at the end, 
                                                                        following the format [(N1, N2), (N3, N4), ...] to simulate GPS outage from 
                                                                        N1 to N2 seconds and N3 to N4 seconds etc. Format of Nx: "MM:SS".
        """

        self.__settings.set_objective_settings(**kwargs)

        # Save project new settings.
        joblib.dump(value=self.__settings, filename=f"./project_{self.__project_name:s}/settings.pkl")

    @ProjectLifeCycleManager.settings_definition_stage_guard
    def set_sensor_settings(
        self: Self,
        **kwargs: Any
    ) -> None:

        """
        Print the sensor settings. 
        NB: The function do not required to receive every parameter. Parameter not present will not be modified.

        Args:
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

        self.__settings.set_sensor_settings(**kwargs)

        # Save project new settings.
        joblib.dump(value=self.__settings, filename=f"./project_{self.__project_name:s}/settings.pkl")

    @ProjectLifeCycleManager.settings_definition_stage_guard
    @validate_settings_input_guard
    def validate_settings(
        self: Self,
        force_validation: bool=False
    ) -> None:

        """
        Validate the value of the different setting parameters.

        Args:
            - force_validation (bool): Either to force the validation, even in case of warnings, or not. Default = False.
        """

        warnings, errors = self.__settings.validate_settings(
            path_raw_data=pathlib.Path(f"./project_{self.__project_name:s}/data/raw/")
        )

        # If their are some errors, do not validate the settings.
        if len(errors) > 0:
            
            message_output = f"You have actually {len(errors):d} error(s) inside your settings. The settings cannot be validated. Please correct the errors and call again the validation settings method."

        # If their are some warnings and force_validation is False, do not validate the settings.
        elif len(warnings) > 0 and force_validation == False:
            
            message_output = f"You have actually {len(warnings):d} warning(s) inside your settings. You can either change your settings or you can call again the validation settings method with the argument 'force_validation=True'."

        # If their is no error and no warnings or warnings but force_validation is True, validate the settings.
        else:
            
            message_output = "The project settings have been validated."

            # Increase life cycle stage and save the manager on disk.
            self._life_cycle_manager.move_to_next_stage()
            joblib.dump(value=self._life_cycle_manager, filename=f"./project_{self.__project_name:s}/life_cycle_manager.pkl")

        print(message_output)

        if len(errors) > 0:

            errors_report = "\n\nErrors in the settings:\n"

            for i_error, error in enumerate(errors):
                errors_report += f"\t E{i_error + 1:d}: {error:s}\n"

            print(errors_report)

        if len(warnings) > 0:

            warnings_report = "\n\nWarnings in the settings:\n"

            for i_warning, warning in enumerate(warnings):
                warnings_report += f"\t W{i_warning + 1:d}: {warning:s}\n"

            print(warnings_report)

    @ProjectLifeCycleManager.data_preprocessing_stage_guard
    @preprocess_data_input_guard
    def preprocess_data(
        self: Self,
        verbosity: bool=False
    ) -> None:

        """
        Perform the preprocessing of the data.

        Args:
            - verbosity (bool): Indicate if the framework should print verbose to inform of the progress of the data preprocessing. Default = False.
        """

        DataPreprocessor.preprocess_data(
            csv_sep=self.__settings.csv_files.csv_sep,
            csv_encoding=self.__settings.csv_files.csv_encoding,
            sensor_frequency=self.__settings.sensor.frequency,
            n_minimum_seconds_operational_gps=self.__settings.objective.n_minimum_seconds_operational_gps,
            training_sessions_id=self.__settings.dataset.sessions_id_training,
            validation_sessions_id=self.__settings.dataset.sessions_id_validation,
            evaluation_sessions_id=self.__settings.dataset.sessions_id_evaluation,
            n_duplications_per_eval_session=self.__settings.dataset.n_duplications_per_eval_session,
            gps_outage_durations_eval_at_beginning=self.__settings.objective.gps_outage_durations_eval_at_beginning,
            gps_outage_durations_eval_within=self.__settings.objective.gps_outage_durations_eval_within,
            gps_outage_durations_eval_at_end=self.__settings.objective.gps_outage_durations_eval_at_end,
            columns_to_load=self.__settings.sensor.get_features_name(),
            path_raw_data=pathlib.Path(f"./project_{self.__project_name:s}/data/raw/"),
            path_preprocessed_training_data=pathlib.Path(f"./project_{self.__project_name:s}/data/preprocessed/training/"),
            path_preprocessed_validation_data=pathlib.Path(f"./project_{self.__project_name:s}/data/preprocessed/validation/"),
            path_preprocessed_evaluation_data=pathlib.Path(f"./project_{self.__project_name:s}/data/preprocessed/evaluation/"),
            random_seed=self.__settings.general.random_seed,
            verbosity=verbosity
        )

        # Increase life cycle stage and save the manager on disk.
        self._life_cycle_manager.move_to_next_stage()
        joblib.dump(value=self._life_cycle_manager, filename=f"./project_{self.__project_name:s}/life_cycle_manager.pkl")

    @ProjectLifeCycleManager.scaler_fit_stage_guard
    def fit_scalers(
        self: Self
    ) -> None:

        """
        Fit the scalers on the training dataset.
        """

        Scalers.fit(
            columns_for_accelerometer=self.__settings.sensor.get_features_name_accelerometer(),
            columns_for_gyroscope=self.__settings.sensor.get_features_name_gyroscope(),
            columns_for_magnetometer=self.__settings.sensor.get_features_name_magnetometer(),
            columns_for_velocity_gps=self.__settings.sensor.get_features_name_velocity_gps(),
            columns_for_velocity_fusion=self.__settings.sensor.get_features_name_velocity_fusion(),
            columns_for_orientation_fusion=self.__settings.sensor.get_features_name_orientation_fusion(),
            dtype=self.__settings.general.dtype,
            path_preprocessed_training_data=pathlib.Path(f"./project_{self.__project_name:s}/data/preprocessed/training/"),
            path_scalers=pathlib.Path(f"./project_{self.__project_name:s}/")
        )

        # Increase life cycle stage and save the manager on disk.
        self._life_cycle_manager.move_to_next_stage()
        joblib.dump(value=self._life_cycle_manager, filename=f"./project_{self.__project_name:s}/life_cycle_manager.pkl")

    def __remove_evaluation_results_if_exists(
        self: Self,
        id_config: str
    ) -> None:

        """
        Remove the evaluation results for the asked configuration if they exists.
        This method ensure that in case of tuning continuation (after a configuration has already been tuned and evaluated),
        old evaluation results do not persists on disk.

        Args:
            - id_config (str): A unique id which represents the configuration.
        """

        # Iterate over case folder.
        for folder_case in pathlib.Path(f"./project_{self.__project_name:s}/evaluation/").iterdir():

            # Iterate over subcase folder.
            for folder_subcase in folder_case.iterdir():

                # Construct the path of the evaluation results for this configuration on this subcase.
                path_results_config = pathlib.Path(folder_subcase, f"evaluation_results_{id_config:s}.json")

                # If the path exists, remove the evaluation results.
                if path_results_config.exists():
                    path_results_config.unlink()
                    

    @ProjectLifeCycleManager.models_tuning_stage_guard
    @run_tuning_input_guard
    def run_tuning(
        self: Self,
        n_experiments: int,
        model_name: str,
        use_adversarial: Union[str, None],
        training_type: str,
        coeff_frequency_division: int,
        gps_outage_duration: Tuple[str, str],
        verbosity: int
    ) -> None:

        """
        Start the tuning for the desired configuration.

        Args:
            - n_experiments (int): The number of hyperparameters sets to try. Have to be strictly positive.
            - model_name (str): The name of the model to tune ('CLSTMTFAWB', 'CLSTMTWB', 'STTFAWB', 'STTWB', 'TCANWB').
            - use_adversarial (Union[str, None]): Either to use adversarial example during training on IMU data ('imu'), on all the input ('full') or not (None).
            - training_type (str): The type of training (GPS outage simulation placement). Can be either 'beginning', 'centered', 'end' or 'random'.
            - coeff_frequency_division (int): The coefficient of frequency division before feeding the model. Have to be strictly positive.
            - gps_outage_duration (Tuple[str, str]): The minimal and maximal duration of the GPS outages. Format: ("MM:SS", "MM:SS").
            - verbosity (int): The level of verbosity. Lvl 0 print no verbose, lvl 1 print only final model's metrics and lvl 2 print intermediate (every epoch) model's metrics.
        """

        # If the coefficient of frequency division is not valid, do not start the tuning.
        if not TuningManager.check_coeff_frequency_division_validity(
            frequency=self.__settings.sensor.frequency,
            coeff_frequency_division=coeff_frequency_division,
            n_minimum_seconds_operational_gps=self.__settings.objective.n_minimum_seconds_operational_gps,
        ):
            return None

        # Load scalers and relax points
        scalers = joblib.load(f"./project_{self.__project_name:s}/scalers.pkl")
        relax_points = joblib.load(f"./project_{self.__project_name:s}/relax_points.pkl")

        # Construct id_config
        if use_adversarial is None:
            adv_text = "without_adv"
        elif use_adversarial == "imu":
            adv_text = "with_imu_adv"
        elif use_adversarial == "full":
            adv_text = "with_full_adv"
            
        gps_outage_duration_txt = f"{gps_outage_duration[0][0:2]:s}-{gps_outage_duration[0][3:5]:s}__{gps_outage_duration[1][0:2]:s}-{gps_outage_duration[1][3:5]:s}"
        
        id_config = f"{model_name}__{training_type}__{coeff_frequency_division:d}__{adv_text}__{gps_outage_duration_txt}"

        # Compute min_gps_outage_duration_seconds, max_gps_outage_duration_seconds, and len_window_seconds
        min_gps_outage_duration_seconds = 60 * int(gps_outage_duration[0][0:2]) + int(gps_outage_duration[0][3:5])
        max_gps_outage_duration_seconds = 60 * int(gps_outage_duration[1][0:2]) + int(gps_outage_duration[1][3:5])

        # Remove evaluation results for this configuration if it exists.
        self.__remove_evaluation_results_if_exists(id_config=id_config)

        TuningManager.run_tuning(
            n_experiments=n_experiments,
            n_startup_trials=self.__settings.tuning.n_startup_trials,
            save_every_state_dicts=self.__settings.tuning.save_every_state_dicts,
            scalers=scalers,
            relax_points=relax_points,
            id_config=id_config,
            model_name=model_name,
            use_adversarial=use_adversarial,
            training_type=training_type,
            velocity_loss=self.__settings.training.velocity_loss,
            coeff_frequency_division=coeff_frequency_division,
            frequency=self.__settings.sensor.frequency,
            min_gps_outage_duration_seconds=min_gps_outage_duration_seconds,
            max_gps_outage_duration_seconds=max_gps_outage_duration_seconds,
            n_minimum_seconds_operational_gps=self.__settings.objective.n_minimum_seconds_operational_gps,
            coeff_sampling_training=self.__settings.training.coeff_sampling_training,
            coeff_sampling_validation=self.__settings.training.coeff_sampling_validation,
            n_epochs_sampling=self.__settings.training.n_epochs_sampling,
            n_epochs=self.__settings.training.n_epochs,
            patience=self.__settings.training.patience,
            n_epochs_training_checkpoint=self.__settings.training.n_epochs_training_checkpoint,
            sessions_id_training=self.__settings.dataset.sessions_id_training,
            sessions_id_validation=self.__settings.dataset.sessions_id_validation,
            features_name_accelerometer=self.__settings.sensor.get_features_name_accelerometer(),
            features_name_gyroscope=self.__settings.sensor.get_features_name_gyroscope(),
            features_name_magnetometer=self.__settings.sensor.get_features_name_magnetometer(),
            features_name_velocity_gps=self.__settings.sensor.get_features_name_velocity_gps(),
            features_name_velocity_fusion=self.__settings.sensor.get_features_name_velocity_fusion(),
            features_name_orientation_fusion=self.__settings.sensor.get_features_name_orientation_fusion(),
            num_workers=self.__settings.training.num_workers,
            use_mixed_precision=self.__settings.training.use_mixed_precision,
            dtype=self.__settings.general.dtype,
            device=self.__settings.general.device,
            random_seed=self.__settings.general.random_seed,
            verbosity=verbosity,
            path_data_training=pathlib.Path(f"./project_{self.__project_name:s}/data/preprocessed/training/"),
            path_data_validation=pathlib.Path(f"./project_{self.__project_name:s}/data/preprocessed/validation/"),
            path_temp=pathlib.Path(f"./project_{self.__project_name:s}/temp/"),
            path_tuning_results=pathlib.Path(f"./project_{self.__project_name:s}/tuning/"),
            path_studies=pathlib.Path(f"./project_{self.__project_name:s}/studies/")
        )

    @ProjectLifeCycleManager.models_tuning_stage_guard
    @evaluate_configuration_input_guard
    def evaluate_configuration(
        self: Self,
        model_name: str,
        use_adversarial: Union[str, None],
        training_type: str,
        coeff_frequency_division: int,
        gps_outage_duration: Tuple[str, str],
        external_source: bool=False,
        verbosity: bool=False
    ) -> None:

        """
        Perform the evaluation for the desired configuration.

        Args:
            - model_name (str): The name of the model to tune ('CLSTMTFAWB', 'CLSTMTWB', 'STTFAWB', 'STTWB', 'TCANWB').
            - use_adversarial (Union[str, None]): Either to use adversarial example during training on IMU data ('imu'), on all the input ('full') or not (None).
            - training_type (str): The type of training (GPS outage simulation placement). Can be either 'beginning', 'centered', 'end' or 'random'.
            - coeff_frequency_division (int): The coefficient of frequency division before feeding the model. Have to be strictly positive.
            - gps_outage_duration (Tuple[str, str]): The minimal and maximal duration of the GPS outages. Format: ("MM:SS", "MM:SS").
            - external_source (bool): If True, the configuration will not be loaded from the project but from and external torch script. Used for validation and debug purpose. Default = False.
            - verbosity (bool): Indicate if the framework should print verbose to inform of the progress of the evaluation. Default = False.
        """

        # Load scalers and relax points
        scalers = joblib.load(f"./project_{self.__project_name:s}/scalers.pkl")
        relax_points = joblib.load(f"./project_{self.__project_name:s}/relax_points.pkl")

        # Construct id_config
        if use_adversarial is None:
            adv_text = "without_adv"
        elif use_adversarial == "imu":
            adv_text = "with_imu_adv"
        elif use_adversarial == "full":
            adv_text = "with_full_adv"
            
        gps_outage_duration_txt = f"{gps_outage_duration[0][0:2]:s}-{gps_outage_duration[0][3:5]:s}__{gps_outage_duration[1][0:2]:s}-{gps_outage_duration[1][3:5]:s}"
        
        id_config = f"{model_name}__{training_type}__{coeff_frequency_division:d}__{adv_text}__{gps_outage_duration_txt}"

        if external_source == True:
            path_external_source = pathlib.Path(f"./project_{self.__project_name:s}/external/")
        else:
            path_external_source = None

        EvaluationManager.evaluate_configuration(
            scalers=scalers,
            relax_points=relax_points,
            id_config=id_config,
            model_name=model_name,
            training_type=training_type,
            gps_outage_duration=gps_outage_duration,
            coeff_frequency_division=coeff_frequency_division,
            frequency=self.__settings.sensor.frequency,
            n_minimum_seconds_operational_gps=self.__settings.objective.n_minimum_seconds_operational_gps,
            sessions_id_evaluation=self.__settings.dataset.sessions_id_evaluation,
            features_name_accelerometer=self.__settings.sensor.get_features_name_accelerometer(),
            features_name_gyroscope=self.__settings.sensor.get_features_name_gyroscope(),
            features_name_magnetometer=self.__settings.sensor.get_features_name_magnetometer(),
            features_name_velocity_gps=self.__settings.sensor.get_features_name_velocity_gps(),
            features_name_position_fusion=self.__settings.sensor.get_features_name_position_fusion(),
            features_name_velocity_fusion=self.__settings.sensor.get_features_name_velocity_fusion(),
            features_name_orientation_fusion=self.__settings.sensor.get_features_name_orientation_fusion(),
            num_workers=self.__settings.training.num_workers,
            dtype=self.__settings.general.dtype,
            device=self.__settings.general.device,
            verbosity=verbosity,
            path_evaluation_data=pathlib.Path(f"./project_{self.__project_name:s}/data/preprocessed/evaluation/"),
            path_studies=pathlib.Path(f"./project_{self.__project_name:s}/studies/"),
            path_tuning_results=pathlib.Path(f"./project_{self.__project_name:s}/tuning/"),
            path_evaluation_results=pathlib.Path(f"./project_{self.__project_name:s}/evaluation/"),
            path_external_source=path_external_source
        )

    @ProjectLifeCycleManager.models_tuning_stage_guard
    @validate_models_tuning_input_guard
    def validate_models_tuning(
        self: Self,
        force_validation: bool=False
    ) -> None:

        """
        Validate the stage of models tuning.

        Args:
            - force_validation (bool): Either to force the validation, even in case of warnings, or not. Default = False.
        """

        ids_config_not_evaluated = EvaluationManager.check_models_evaluation(
            gps_outage_durations_eval_at_beginning=self.__settings.objective.gps_outage_durations_eval_at_beginning,
            gps_outage_durations_eval_within=self.__settings.objective.gps_outage_durations_eval_within,
            gps_outage_durations_eval_at_end=self.__settings.objective.gps_outage_durations_eval_at_end,
            path_tuning_results=pathlib.Path(f"./project_{self.__project_name:s}/tuning/"),
            path_evaluation_results=pathlib.Path(f"./project_{self.__project_name:s}/evaluation/")
        )

        # If evaluation has been performed for every model or if force_validation is True, validate the models tuning.
        if len(ids_config_not_evaluated) == 0 or force_validation == True:
            output_message = "The models tuning has been validated!"
            self._life_cycle_manager.move_to_next_stage()
            joblib.dump(value=self._life_cycle_manager, filename=f"./project_{self.__project_name:s}/life_cycle_manager.pkl")
        # Otherwise, just inform the user.
        else:
            output_message = "The models tuning has not been validated because of warnings. You have one or severals tuned configurations that do not have been evaluated. You can either perform the validation for these configurations or call again the models tuning validation method with the argument 'force_validation=True'."

        # If their is missing evaluation, print a report.
        if len(ids_config_not_evaluated) > 0:

            output_message += "\n\nList of the configuration for which the evaluation has not been performed:"

            for id_config in ids_config_not_evaluated:

                model_name = id_config.split("__")[0]
                training_type = id_config.split("__")[1]
                coeff_frequency_division = id_config.split("__")[2]
                adv_txt = " ".join(id_config.split("__")[3].split("_"))
                gps_outage_duration_txt = " to ".join(id_config.split("__")[4:6])

                output_message += f"\n - {model_name} with training type {training_type} {adv_txt} and a frequency division of {coeff_frequency_division} on GPS outage from {gps_outage_duration_txt}"

        print(output_message)

    @ProjectLifeCycleManager.averaging_configuration_stage_guard
    @averaging_configuration_input_guard
    def averaging_configuration(
        self: Self,
        model_selection_levels: List[float],
        external_source: bool=False,
        verbosity: bool=False
    ) -> None:

        """
        Perform the configuration of the averaging.

        Args:
            - model_selection_levels (List[float]): The list of the level for the model selection (e.g. [0.33, 0.66, 1] for 33%, 66% and 100%).
            - external_source (bool): If True, the configuration will not be loaded from the project but from and external torch script. Used for validation and debug purpose. Default = False.
            - verbosity (bool): Indicate if the framework should print verbose to inform of the progress of the evaluation. Default = False.
        """
        
        # Load scalers and relax points
        scalers = joblib.load(f"./project_{self.__project_name:s}/scalers.pkl")
        relax_points = joblib.load(f"./project_{self.__project_name:s}/relax_points.pkl")

        if external_source == True:
            path_external_source = pathlib.Path(f"./project_{self.__project_name:s}/external/")
        else:
            path_external_source = None

        EvaluationManager.evaluate_averaging(
            model_selection_levels=model_selection_levels,
            scalers=scalers,
            relax_points=relax_points,
            frequency=self.__settings.sensor.frequency,
            n_minimum_seconds_operational_gps=self.__settings.objective.n_minimum_seconds_operational_gps,
            sessions_id_evaluation=self.__settings.dataset.sessions_id_evaluation,
            features_name_accelerometer=self.__settings.sensor.get_features_name_accelerometer(),
            features_name_gyroscope=self.__settings.sensor.get_features_name_gyroscope(),
            features_name_magnetometer=self.__settings.sensor.get_features_name_magnetometer(),
            features_name_velocity_gps=self.__settings.sensor.get_features_name_velocity_gps(),
            features_name_position_fusion=self.__settings.sensor.get_features_name_position_fusion(),
            features_name_velocity_fusion=self.__settings.sensor.get_features_name_velocity_fusion(),
            features_name_orientation_fusion=self.__settings.sensor.get_features_name_orientation_fusion(),
            dtype=self.__settings.general.dtype,
            device=self.__settings.general.device,
            verbosity=verbosity,
            path_evaluation_data=pathlib.Path(f"./project_{self.__project_name:s}/data/preprocessed/evaluation/"),
            path_studies=pathlib.Path(f"./project_{self.__project_name:s}/studies/"),
            path_tuning_results=pathlib.Path(f"./project_{self.__project_name:s}/tuning/"),
            path_evaluation_results=pathlib.Path(f"./project_{self.__project_name:s}/evaluation/"),
            path_external_source=path_external_source
        )

        self._life_cycle_manager.move_to_next_stage()
        joblib.dump(value=self._life_cycle_manager, filename=f"./project_{self.__project_name:s}/life_cycle_manager.pkl")

    def display_dashboard(
        self: Self
    ) -> None:

        proc = subprocess.Popen(
            [
                "streamlit", 
                "run", 
                "./posnnet/dashboard/script_dashboard.py",
                f"./project_{self.__project_name:s}/studies/",
                f"./project_{self.__project_name:s}/tuning/",
                f"./project_{self.__project_name:s}/evaluation/"
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )