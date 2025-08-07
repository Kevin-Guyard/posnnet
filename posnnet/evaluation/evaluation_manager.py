import joblib
import json
import numpy as np
import pathlib
import torch
from typing import TypeVar, Type, List, Tuple, Dict, Union

from posnnet.data.evaluation_dataset import EvaluationDataset
from posnnet.evaluation.evaluator import Evaluator
from posnnet.models import GeneralModel
from posnnet.study_manager import StudyManager

import posnnet.data.scalers


class EvaluationManager:

    @classmethod
    def evaluate_configuration(
        cls: Type[TypeVar("EvaluationManager")],
        scalers: posnnet.data.scalers.Scalers,
        relax_points: Dict[str, Dict[str, np.ndarray]],
        id_config: str,
        model_name: str,
        training_type: str,
        gps_outage_duration: Tuple[str, str],
        coeff_frequency_division: int,
        frequency: int,
        n_minimum_seconds_operational_gps: int,
        sessions_id_evaluation: List[int],
        features_name_accelerometer: List[str],
        features_name_gyroscope: List[str],
        features_name_magnetometer: List[str],
        features_name_velocity_gps: List[str],
        features_name_position_fusion: List[str],
        features_name_velocity_fusion: List[str],
        features_name_orientation_fusion: List[str],
        num_workers: int,
        dtype: str,
        device: torch.device,
        verbosity: bool,
        path_evaluation_data: pathlib.Path,
        path_studies: pathlib.Path,
        path_tuning_results: pathlib.Path,
        path_evaluation_results: pathlib.Path,
        path_external_source: Union[pathlib.Path, None]=None
    ) -> None:

        """
        Manage the evaluation of the configuration.

        Args:
            - scalers (posnnet.data.scalers.Scalers): The project Scalers instance.
            - relax_points (Dict[str, Dict[str, np.ndarray]]): The relax points (scaled) of the dataset (velocity GPS = 0 / velocity fusion = 0 / orientation fusion = 0).
            - id_config (str): A unique id which represents the configuration.
            - model_name (str): The name of the model for which perform the trial.
            - training_type (str): The type of training (GPS outage placement). Can be either 'beginning', 'centered', 'end' or 'random'.
            - gps_outage_duration (Tuple[str, str]): The GPS outage duration capacity of the model. Format = ("MM:SS", "MM:SS")
            - coeff_frequency_division (int): The ratio of frequency division (e.g. a window of 2400 samples with a frequency division of 10 leads to a sequence of 240 samples).
            - frequency (int): The original frequency of the sensor.
            - n_minimum_seconds_operational_gps (int): The minimum number of seconds of operational GPS that can be ensured next to every GPS outage.
            - sessions_id_evaluation (List[int]): The list of sessions id to use for the evaluation dataset.
            - features_name_accelerometer (List[str]): The list of features name for the accelerometer (0 <= len(features_name_accelerometer) <= 3).
            - features_name_gyroscope (List[str]): The list of features name for the gyroscope (0 <= len(features_name_gyroscope) <= 3).
            - features_name_magnetometer (List[str]): The list of features name for the magnetometer (0 <= len(features_name_magnetometer) <= 3).
            - features_name_velocity_gps (List[str]): The list of features name for the GPS velocity (0 <= len(features_name_velocity_gps) <= 3).
            - features_name_position_fusion (List[str]): The list of features name for the fusion position (1 <= len(features_name_position_fusion) <= 3).
            - features_name_velocity_fusion (List[str]): The list of features name for the fusion velocity (1 <= len(features_name_velocity_fusion) <= 3).
            - features_name_orientation_fusion (List[str]): The list of features name for the fusion orientation (0 <= len(features_name_orientation_fusion) <= 3).
            - num_workers (int): The number of process that will instanciate a dataloader instance to load the data in parallel. 0 means that*
                                 the dataloader will be loaded in the main process. A rule of thumb is to use 4 processes by GPU.
            - dtype (str): The dtype to use for the model and the data (for example: 'float16', 'float32', 'float64').
            - device (torch.device): The device on which perform the inference.
            - verbosity (bool): Indicate if the framework should print verbose to inform of the progress of the evaluation.
            - path_evaluation_data (pathlib.Path): The path where are stored the evaluation data.
            - path_studies (pathlib.Path): The path where are stored the study object of the tuning.
            - path_tuning_results (pathlib.Path): The path where are stored the tuning results.
            - path_evaluation_results (pathlib.Path): The path where will be stored the evaluation results.
            - path_external_source (Union[pathlib.Path, None]): If provided, the model will not be loaded from the project but from an external torch script.
                                                                Used for debug and validation. Default = None.
        """

        # If no external source path is provided, load the model and params from the project.
        if path_external_source is None:

            # Load the study associated to the configuration.
            study = StudyManager(
                id_config=id_config,
                path_studies=path_studies,
                path_tuning_results=path_tuning_results
            ).load()
    
            # Ensure that at least on trial has been finished.
            try:
                best_trial_number = study.best_trial.number
            except ValueError:
                raise Exception("You have to perform the tuning of the configuration before evaluating it.")
    
            # Load model/training params and state dict.
            path_model_params = pathlib.Path(path_tuning_results, f"{id_config:s}/trial {best_trial_number + 1:03d}/model_params.pkl")
            path_training_params = pathlib.Path(path_tuning_results, f"{id_config:s}/trial {best_trial_number + 1:03d}/training_params.pkl")
            path_state_dict = pathlib.Path(path_tuning_results, f"{id_config:s}/trial {best_trial_number + 1:03d}/state_dict.pt")
    
            model_params = joblib.load(filename=path_model_params)
            training_params = joblib.load(filename=path_training_params)
            model_state_dict = torch.load(f=path_state_dict)
    
            # Load model.
            model = GeneralModel(
                model_name=model_name,
                model_params=model_params,
                model_state_dict=model_state_dict
            )

        # If a path is provided, loaded the torch script and params present in the provided path.
        else:

            model = torch.jit.load(
                f=pathlib.Path(path_external_source, f"{id_config:s}/torch_script_model.pt"), 
                map_location="cpu"
            )
            training_params = joblib.load(filename=pathlib.Path(path_external_source, f"{id_config:s}/training_params.pkl"))
            

        # Get the cases on which evaluate the model.
        cases = {
            "beginning": ["beginning"],
            "centered": ["within"],
            "end": ["end"],
            "random": ["beginning", "within", "end"]
        }.get(training_type)

        # Get the minimum and maximum GPS outage duration that the model can deal with.
        minimum_number_seconds_gpt_outage = 60 * int(gps_outage_duration[0][0:2]) + int(gps_outage_duration[0][3:5])
        maximum_number_seconds_gps_outage = 60 * int(gps_outage_duration[1][0:2]) + int(gps_outage_duration[1][3:5])

        # Compute the length of the window that will be treated.
        len_window = (maximum_number_seconds_gps_outage + n_minimum_seconds_operational_gps) * frequency

        # Iterate over the cases.
        for case in cases:

            path_case = pathlib.Path(path_evaluation_data, f"{case:s}")

            # If their is not data for the case, skip it.
            if not path_case.exists():
                continue

            # Get the subcases.
            subcases = [
                (
                    60 * int(subcase_repo.name.split(" to ")[0][0:2]) + int(subcase_repo.name.split(" to ")[0][3:5]),
                    60 * int(subcase_repo.name.split(" to ")[1][0:2]) + int(subcase_repo.name.split(" to ")[1][3:5]),
                    subcase_repo.name
                )
                for subcase_repo in path_case.iterdir()
                if " to " in subcase_repo.name
            ] # [(n_min_seconds, n_max_seconds, txt), ...]

            # Iterate over the subcases.
            for subcase in subcases:

                # Skip subcase if the  GPS outage duration cannot be handled by the model.
                if subcase[0] < minimum_number_seconds_gpt_outage or subcase[1] > maximum_number_seconds_gps_outage:
                    continue

                # Prepare evaluation dataset.
                dataset_eval = EvaluationDataset(
                    len_window=len_window,
                    coeff_frequency_division=coeff_frequency_division,
                    scaling_type=training_params["scaling_type"],
                    scalers=scalers,
                    sessions_id=sessions_id_evaluation,
                    features_name_accelerometer=features_name_accelerometer,
                    features_name_gyroscope=features_name_gyroscope,
                    features_name_magnetometer=features_name_magnetometer,
                    features_name_velocity_gps=features_name_velocity_gps,
                    features_name_position_fusion=features_name_position_fusion,
                    features_name_velocity_fusion=features_name_velocity_fusion,
                    features_name_orientation_fusion=features_name_orientation_fusion,
                    dtype=getattr(np, dtype),
                    path_data=pathlib.Path(path_case, f"{subcase[2]:s}/")
                )

                # Perform evaluation on the subcase.
                metrics = Evaluator.evaluate_on_subcase(
                    model=model,
                    dataset_eval=dataset_eval,
                    scalers=scalers,
                    coeff_frequency_division=coeff_frequency_division,
                    frequency=frequency,
                    scaling_type=training_params["scaling_type"],
                    relax_points=relax_points,
                    num_workers=num_workers,
                    device=device,
                    dtype=getattr(torch, dtype),
                    verbosity=verbosity,
                    subcase_name=f"Case {case:s} from {subcase[2]:s}"
                )

                # Initialize the path of the results and create the parent folder if it does not exist.
                path_results = pathlib.Path(path_evaluation_results, f"{case:s}/{subcase[2]:s}/evaluation_results_{id_config:s}.json")
                path_results.parent.mkdir(exist_ok=True, parents=True)

                with open(path_results, "w") as result_file:
                    json.dump(obj=metrics, fp=result_file)

    @classmethod
    def check_models_evaluation(
        cls: Type[TypeVar("EvaluationManager")],
        gps_outage_durations_eval_at_beginning: List[Tuple[str, str]],
        gps_outage_durations_eval_within: List[Tuple[str, str]],
        gps_outage_durations_eval_at_end: List[Tuple[str, str]],
        path_tuning_results: pathlib.Path,
        path_evaluation_results: pathlib.Path
    ) -> List[str]:

        """
        Verify if every configuration tuned has been evaluated.

        Args:
            - gps_outage_durations_eval_at_beginning (List[Tuple[str, str]]): The list of cases of GPS outage durations that will be simulated at the beginning, 
                                                                              following the format [(N1, N2), (N3, N4), ...] to simulate GPS outage from 
                                                                              N1 to N2 seconds and N3 to N4 seconds etc. Format of Nx: "MM:SS".
            - gps_outage_durations_eval_within (List[Tuple[str, str]]): The list of cases of GPS outage durations that will be simulated within the session, 
                                                                        following the format [(N1, N2), (N3, N4), ...] to simulate GPS outage from 
                                                                        N1 to N2 seconds and N3 to N4 seconds etc. Format of Nx: "MM:SS".
            - gps_outage_durations_eval_at_end (List[Tuple[str, str]]): The list of cases of GPS outage durations that will be simulated at the end, 
                                                                        following the format [(N1, N2), (N3, N4), ...] to simulate GPS outage from 
                                                                        N1 to N2 seconds and N3 to N4 seconds etc. Format of Nx: "MM:SS".
            - path_tuning_results (pathlib.Path): The path where are stored the tuning results.
            - path_evaluation_results (pathlib.Path): The path where will be stored the evaluation results.

        Returns:
            - ids_config_not_evaluated (List[str]): The list of id of the configurations for which the evaluation has not been performed.
        """

        ids_config_not_evaluated = []

        mapping_training_type_to_cases = {
            "beginning": ["beginning"],
            "centered": ["within"],
            "end": ["end"],
            "random": ["beginning", "within", "end"]
        }
        mapping_case_to_subcases = {
            "beginning": gps_outage_durations_eval_at_beginning,
            "within": gps_outage_durations_eval_within,
            "end": gps_outage_durations_eval_at_end
        }

        # Iterate over the tuning results folder.
        for folder_tuning_results in path_tuning_results.iterdir():

            # If the folder do not contains at least the first trial, skip this configuration.
            if not folder_tuning_results.is_dir() or not pathlib.Path(folder_tuning_results, "trial 001").exists():
                continue

            # Collect information on the configuration.
            id_config = folder_tuning_results.name
            training_type = id_config.split("__")[1] 
            min_gps_outage_duration_config = sum(int(x) * 60**i for i, x in enumerate(reversed(id_config.split("__")[4].split("-")))) # Convert MM-SS to an integer
            max_gps_outage_duration_config = sum(int(x) * 60**i for i, x in enumerate(reversed(id_config.split("__")[5].split("-")))) # Convert MM-SS to an integer

            cases = mapping_training_type_to_cases[training_type]

            for case in cases:

                subcases = mapping_case_to_subcases[case]

                for subcase in subcases:

                    min_gps_outage_duration_subcase = sum(int(x) * 60**i for i, x in enumerate(reversed(subcase[0].split(":")))) # Convert MM-SS to an integer
                    max_gps_outage_duration_subcase = sum(int(x) * 60**i for i, x in enumerate(reversed(subcase[1].split(":")))) # Convert MM-SS to an integer

                    # If the configuration is not adapted for this subcase, skip it.
                    if min_gps_outage_duration_subcase < min_gps_outage_duration_config or max_gps_outage_duration_subcase > max_gps_outage_duration_config:
                        continue

                    # If the evaluation result is not present, add the configuration to the missing list.
                    if not pathlib.Path(path_evaluation_results, case, f"{'-'.join(subcase[0].split(":"))} to {'-'.join(subcase[1].split(":"))}/evaluation_results_{id_config}.json").exists():
                        ids_config_not_evaluated.append(id_config)

        # Ensure uniqueness of the list.
        ids_config_not_evaluated = list(set(ids_config_not_evaluated))

        return ids_config_not_evaluated

    @classmethod
    def evaluate_averaging(
        cls: Type[TypeVar("EvaluationManager")],
        model_selection_levels: List[float],
        scalers: posnnet.data.scalers.Scalers,
        relax_points: Dict[str, Dict[str, np.ndarray]],
        frequency: int,
        n_minimum_seconds_operational_gps: int,
        sessions_id_evaluation: List[int],
        features_name_accelerometer: List[str],
        features_name_gyroscope: List[str],
        features_name_magnetometer: List[str],
        features_name_velocity_gps: List[str],
        features_name_position_fusion: List[str],
        features_name_velocity_fusion: List[str],
        features_name_orientation_fusion: List[str],
        dtype: str,
        device: torch.device,
        verbosity: bool,
        path_evaluation_data: pathlib.Path,
        path_studies: pathlib.Path,
        path_tuning_results: pathlib.Path,
        path_evaluation_results: pathlib.Path,
        path_external_source: Union[pathlib.Path, None]=None
    ) -> None:

        """
        Manage the evaluation of the averaging.

        Args:
            - model_selection_levels (List[float]): The list of the level for the model selection (e.g. [0.33, 0.66, 1] for 33%, 66% and 100%).
            - scalers (posnnet.data.scalers.Scalers): The project Scalers instance.
            - relax_points (Dict[str, Dict[str, np.ndarray]]): The relax points (scaled) of the dataset (velocity GPS = 0 / velocity fusion = 0 / orientation fusion = 0).
            - frequency (int): The original frequency of the sensor.
            - n_minimum_seconds_operational_gps (int): The minimum number of seconds of operational GPS that can be ensured next to every GPS outage.
            - sessions_id_evaluation (List[int]): The list of sessions id to use for the evaluation dataset.
            - features_name_accelerometer (List[str]): The list of features name for the accelerometer (0 <= len(features_name_accelerometer) <= 3).
            - features_name_gyroscope (List[str]): The list of features name for the gyroscope (0 <= len(features_name_gyroscope) <= 3).
            - features_name_magnetometer (List[str]): The list of features name for the magnetometer (0 <= len(features_name_magnetometer) <= 3).
            - features_name_velocity_gps (List[str]): The list of features name for the GPS velocity (0 <= len(features_name_velocity_gps) <= 3).
            - features_name_position_fusion (List[str]): The list of features name for the fusion position (1 <= len(features_name_position_fusion) <= 3).
            - features_name_velocity_fusion (List[str]): The list of features name for the fusion velocity (1 <= len(features_name_velocity_fusion) <= 3).
            - features_name_orientation_fusion (List[str]): The list of features name for the fusion orientation (0 <= len(features_name_orientation_fusion) <= 3).
            - dtype (str): The dtype to use for the model and the data (for example: 'float16', 'float32', 'float64').
            - device (torch.device): The device on which perform the inference.
            - verbosity (bool): Indicate if the framework should print verbose to inform of the progress of the evaluation.
            - path_evaluation_data (pathlib.Path): The path where are stored the evaluation data.
            - path_studies (pathlib.Path): The path where are stored the study object of the tuning.
            - path_tuning_results (pathlib.Path): The path where are stored the tuning results.
            - path_evaluation_results (pathlib.Path): The path where will be stored the evaluation results.
            - path_external_source (Union[pathlib.Path, None]): If provided, the model will not be loaded from the project but from an external torch script.
                                                                Used for debug and validation. Default = None.
        """

        # Iterate over every evaluation cases.
        for case in ["beginning", "within", "end"]:

            path_case = pathlib.Path(path_evaluation_data, case)

            # If the evaluation data case folder does not exist, skip the case.
            if not path_case.exists():
                continue

            # Iterate over subcase
            for subcase_folder in path_case.iterdir():

                # If the folder is not a subcase folder, skip it.
                if not " to " in subcase_folder.name:
                    continue

                # If no evaluation has been performed for this subcase, skip it.
                if not pathlib.Path(path_evaluation_results, f"./{case}/{subcase_folder.name}/").exists():
                    continue

                min_gps_outage_duration_subcase = sum(int(x) * 60**i for i, x in enumerate(reversed(subcase_folder.name.split(" to ")[0].split("-")))) # Convert MM-SS to an integer
                max_gps_outage_duration_subcase = sum(int(x) * 60**i for i, x in enumerate(reversed(subcase_folder.name.split(" to ")[1].split("-")))) # Convert MM-SS to an integer

                # Compute the length of the window that will be treated.
                len_window = (max_gps_outage_duration_subcase + n_minimum_seconds_operational_gps) * frequency

                models = {}
                tuning_ates = {}
                coeffs_frequency_division = {}
                scaling_types = {}

                # If no external source path is provided, load the model from the project tuning.
                if path_external_source is None:

                    # Iterate over study files.
                    for study_file in path_studies.iterdir():
    
                        # If not a study file, skip this file.
                        if not "study" in study_file.name:
                            continue
    
                        id_config = study_file.name[6:-4] # study file name = 'study_{id_config}.pkl'
                        training_type = id_config.split("__")[1]
    
                        min_gps_outage_duration_conf = sum(int(x) * 60**i for i, x in enumerate(reversed(id_config.split("__")[4].split("-")))) # Convert MM-SS to an integer
                        max_gps_outage_duration_conf = sum(int(x) * 60**i for i, x in enumerate(reversed(id_config.split("__")[5].split("-")))) # Convert MM-SS to an integer
    
                        # Skip this configuration if the conf is not able to treat the GPS outage duration of the subcase.
                        if min_gps_outage_duration_subcase < min_gps_outage_duration_conf or max_gps_outage_duration_subcase > max_gps_outage_duration_conf:
                            continue
    
                        # Load the study
                        study = joblib.load(filename=study_file)
    
                        # Collect the best trial. In case of no finished trial (ValueError), skip this configuration.
                        try:
                            best_trial_number = study.best_trial.number
                        except ValueError:
                            continue
    
                        path_tuning_results_conf = pathlib.Path(path_tuning_results, f"{id_config}/trial {best_trial_number + 1:03d}/")
    
                        # Load model.
                        model_name = id_config.split("__")[0]
                        model_params = joblib.load(filename=pathlib.Path(path_tuning_results_conf, "model_params.pkl"))
                        model_state_dict = torch.load(f=pathlib.Path(path_tuning_results_conf, "state_dict.pt"))
                        models[id_config] = GeneralModel(
                            model_name=model_name,
                            model_params=model_params,
                            model_state_dict=model_state_dict
                        )
    
                        # Load scaling type.
                        training_params = joblib.load(filename=pathlib.Path(path_tuning_results_conf, "training_params.pkl"))
                        scaling_types[id_config] = training_params["scaling_type"]
    
                        # Get the coeff of frequency division.
                        coeffs_frequency_division[id_config] = int(id_config.split("__")[2])
    
                        # Load the ATE on validation during tuning.
                        with open(pathlib.Path(path_tuning_results_conf, "results.json"), "r") as results_file:
                            results = json.load(results_file)

                        # If the training type is not random, collect the ATE.
                        if training_type != "random":
                            tuning_ates[id_config] = results["ate"]
                        # Otherwise, collect the ATE on the associated fixed case.
                        else:
                            training_type_associated_to_the_case = {"beginning": "beginning", "within": "centered", "end": "end"}.get(case)
                            tuning_ates[id_config] = results["scores_on_fixed_training_case"][training_type_associated_to_the_case]["ate"]

                # If an external source path is provided, load the external content.
                else:

                    # Iterate over external sources.
                    for external_folder in path_external_source.iterdir():

                        # Ensure this is an externel source folder, otherwise skip it.
                        if not "with" in external_folder.name:
                            continue

                        id_config = external_folder.name
                        training_type = id_config.split("__")[1]

                        min_gps_outage_duration_conf = sum(int(x) * 60**i for i, x in enumerate(reversed(id_config.split("__")[4].split("-")))) # Convert MM-SS to an integer
                        max_gps_outage_duration_conf = sum(int(x) * 60**i for i, x in enumerate(reversed(id_config.split("__")[5].split("-")))) # Convert MM-SS to an integer

                        # Skip this configuration if the conf is not able to treat the GPS outage duration of the subcase.
                        if min_gps_outage_duration_subcase < min_gps_outage_duration_conf or max_gps_outage_duration_subcase > max_gps_outage_duration_conf:
                            continue

                        # Load model.
                        models[id_config] = torch.jit.load(
                            f=pathlib.Path(external_folder, "torch_script_model.pt"),
                            map_location="cpu"
                        )

                        # Load scaling type.
                        training_params = joblib.load(filename=pathlib.Path(external_folder, "training_params.pkl"))
                        scaling_types[id_config] = training_params["scaling_type"]

                        # Get the coeff of frequency division.
                        coeffs_frequency_division[id_config] = int(id_config.split("__")[2])
    
                        # Load the ATE on validation during tuning.
                        with open(pathlib.Path(external_folder, "results.json"), "r") as results_file:
                            results = json.load(results_file)
                        
                        # If the training type is not random, collect the ATE.
                        if training_type != "random":
                            tuning_ates[id_config] = results["ate"]
                        # Otherwise, collect the ATE on the associated fixed case.
                        else:
                            training_type_associated_to_the_case = {"beginning": "beginning", "within": "centered", "end": "end"}.get(case)
                            tuning_ates[id_config] = results["scores_on_fixed_training_case"][training_type_associated_to_the_case]["ate"]

                dataset_eval_normalized = EvaluationDataset(
                    len_window=len_window,
                    coeff_frequency_division=100,
                    scaling_type="normalization",
                    scalers=scalers,
                    sessions_id=sessions_id_evaluation,
                    features_name_accelerometer=features_name_accelerometer,
                    features_name_gyroscope=features_name_gyroscope,
                    features_name_magnetometer=features_name_magnetometer,
                    features_name_velocity_gps=features_name_velocity_gps,
                    features_name_position_fusion=features_name_position_fusion,
                    features_name_velocity_fusion=features_name_velocity_fusion,
                    features_name_orientation_fusion=features_name_orientation_fusion,
                    dtype=getattr(np, dtype),
                    path_data=subcase_folder
                )

                dataset_eval_standardized = EvaluationDataset(
                    len_window=len_window,
                    coeff_frequency_division=100,
                    scaling_type="standardization",
                    scalers=scalers,
                    sessions_id=sessions_id_evaluation,
                    features_name_accelerometer=features_name_accelerometer,
                    features_name_gyroscope=features_name_gyroscope,
                    features_name_magnetometer=features_name_magnetometer,
                    features_name_velocity_gps=features_name_velocity_gps,
                    features_name_position_fusion=features_name_position_fusion,
                    features_name_velocity_fusion=features_name_velocity_fusion,
                    features_name_orientation_fusion=features_name_orientation_fusion,
                    dtype=getattr(np, dtype),
                    path_data=subcase_folder
                )

                # Evaluate averaging.
                metrics_averagings = Evaluator.evaluate_averaging_on_subcase(
                    model_selection_levels=model_selection_levels,
                    models=models,
                    tuning_ates=tuning_ates,
                    dataset_eval_normalized=dataset_eval_normalized,
                    dataset_eval_standardized=dataset_eval_standardized,
                    scalers=scalers,
                    coeffs_frequency_division=coeffs_frequency_division,
                    frequency=frequency,
                    scaling_types=scaling_types,
                    relax_points=relax_points,
                    device=device,
                    dtype=getattr(torch, dtype),
                    verbosity=verbosity,
                    subcase_name=f"Case {case:s} from {subcase_folder.name:s}"
                )

                for model_selection_level, metrics in metrics_averagings.items():
                    
                    # Initialize the path of the results and create the parent folder if it does not exist.
                    path_results = pathlib.Path(path_evaluation_results, f"{case:s}/{subcase_folder.name:s}/evaluation_results_averaging_{model_selection_level!r}.json")

                    with open(path_results, "w") as result_file:
                        json.dump(obj=metrics, fp=result_file)