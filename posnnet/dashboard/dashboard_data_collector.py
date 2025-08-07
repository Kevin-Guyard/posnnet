import joblib
import json
import pathlib
import streamlit as st
from typing import TypeVar, Type, Dict, List, Any

from posnnet.study_manager import StudyManager


class DashboardDataCollector:

    @classmethod
    @st.cache_data(show_spinner="Collecting tuning data…")
    def collect_tuning_data(
        cls: Type[TypeVar("DashboardDataCollector")],
        path_studies: pathlib.Path,
        path_tuning_results: pathlib.Path
    ) -> Dict[str, Dict[str, Dict[str, List[Any]]]]:

        """
        Collect the tuning data and return it as a dictionnary constructed as follows:

            case
            |____ gps_outage_duration
                  |____ Model name: List[str]
                  |____ Coeff freq div: List[int]
                  |____ Adversarial example: List[str]
                  |____ Velocity loss: List[float]
                  |____ ATE (in m): List[float]

        Args:
            - path_studies (pathlib.Path): The path where are stored the study object of the tuning.
            - path_tuning_results (pathlib.Path): The path where are stored the tuning results.

        Returns:
            - tuning_dashboard_data (Dict[str, Dict[str, Dict[str, List[Any]]]]): The data to use for the tuning dashboard.
        """

        tuning_dashboard_data = {}

        # Iterate over study files.
        for study_file in path_studies.iterdir():

            # If the file name does not contains "study_", this is not a study file (can be a cache file, jupyter checkpoint etc). Skip it.
            if not "study_" in study_file.name:
                continue

            # Get the id of the config for this study. Format of the study file name = "study_{id_config}.pkl".
            id_config = study_file.name[6:-4]

            # Load the study using the study manager (the study manager is able to handle corrupted file and reload missing tuning trials).
            study = StudyManager(
                id_config=id_config,
                path_studies=path_studies,
                path_tuning_results=path_tuning_results
            ).load()

            # Get the best trial number. If the study does not contain any trial, it will raise a ValueError. In this case, skip this study.
            try:
                best_trial_number = study.best_trial.number
            except ValueError:
                continue

            # Get the different attributs of the configuration. id_config = "{model_name}__{training_type}__{coeff_frequency_division}__{txt_adv}__{gps_outage_duration}".
            # Model name.
            model_name = id_config.split("__")[0]
            # Training type.
            training_type = id_config.split("__")[1]
            # Coeff frequency division (convert to int).
            coeff_frequency_division = int(id_config.split("__")[2])
            # Adversarial example (format the text).
            adversarial_example_formatting = {"without_adv": "Not used", "with_imu_adv": "On IMU data", "with_full_adv": "On every data"}
            adversarial_example = adversarial_example_formatting.get(id_config.split("__")[3])
            # GPS outage duration (format the text).               .
            gps_outage_duration = f"{":".join(id_config.split("__")[4].split("-")):s} to {":".join(id_config.split("__")[5].split("-")):s}"

            # Load the tuning result for the best trial.
            with open(pathlib.Path(path_tuning_results, f"{id_config:s}/trial {best_trial_number + 1:03d}/results.json"), "r") as best_trial_tuning_results_file:
                best_trial_tuning_results = json.load(fp=best_trial_tuning_results_file)

            # Get the subcase dashboard data dictionnary (initialize an empty dict if the case and subcase do not already exit).
            subcase_dashboard_data = tuning_dashboard_data.setdefault(f"{training_type:s}", {}).setdefault(f"{gps_outage_duration:s}", {})

            # Add the model to the subcase dashboard data (initialize an empty list if the subcase does not already contain any model data).
            subcase_dashboard_data.setdefault("Model name", []).append(model_name)
            subcase_dashboard_data.setdefault("Coeff freq div", []).append(coeff_frequency_division)
            subcase_dashboard_data.setdefault("Adversarial example", []).append(adversarial_example)
            subcase_dashboard_data.setdefault("Velocity loss", []).append(round(best_trial_tuning_results["velocity_loss"], 6))
            subcase_dashboard_data.setdefault("ATE (in m)", []).append(round(best_trial_tuning_results["ate"], 2))

        return tuning_dashboard_data

    @classmethod
    @st.cache_data(show_spinner="Collecting evaluation data…")
    def collect_evaluation_data(
        cls: Type[TypeVar("DashboardDataCollector")],
        path_evaluation_results: pathlib.Path,
    ) -> Dict[str, Dict[str, Dict[str, List[Any]]]]:

        """
        Collect the evaluation data and return it as a dictionnary constructed as follows:
        
            case
            |____ gps_outage_duration
                  |____ Training type: List[str]
                  |____ GPS outage duration: List[str]
                  |____ Model name: List[str]
                  |____ Coeff freq div: List[int]
                  |____ Adversarial example: List[str]
                  |____ AVE (in m/s): List[float]
                  |____ RMVE (%): List[float]
                  |____ STDVE (in m/s): List[float]
                  |____ ATE (in m): List[float]
                  |____ RMTE (%): List[float]
                  |____ STDTE (in m): List[float]
                  |____ RDE: List[float]

        Args:
            - path_evaluation_results (pathlib.Path): The path where are stored the evaluation results.

        Returns:
            - evaluation_dashboard_data (Dict[str, Dict[str, Dict[str, List[Any]]]]): The data to use for the evaluation dashboard.
        """

        evaluation_dashboard_data = {}

        # Iterate over the 3 cases.
        for case in ["beginning", "within", "end"]:
        
            path_case = pathlib.Path(path_evaluation_results, f"{case}/")

            # If the folder of the case does not exist, no model has been evaluate for this case, skip it.
            if not path_case.exists():
                continue

            # Iterate over the subcases of the case.
            for subcase_folder in path_case.iterdir():

                # If the folder name do not contains " to ", this is not a subcase folder, skip it. Subcase folder name format = "MM-SS to MM-SS".
                if not " to " in subcase_folder.name:
                    continue
        
                subcase = ":".join(subcase_folder.name.split("-"))

                # Iterate over every model evaluation results.
                for evaluation_results_file in subcase_folder.iterdir():

                    # If the file name does not contains "evaluation_results_", this is not an evaluation results file (can be a cache file, jupyter checkpoint etc). Skip it.
                    if not "evaluation_results_" in evaluation_results_file.name:
                        continue

                    # If the file is an averaging configuration evaluation result, skip it.
                    if "averaging" in evaluation_results_file.name:
                        continue
        
                    # Get the id of the config for this evaluation results. Format of the evaluation results file name = "evaluation_results_{id_config}.json".
                    id_config = evaluation_results_file.name[19:-5]
        
                    # Get the different attributs of the configuration. id_config = "{model_name}__{training_type}__{coeff_frequency_division}__{txt_adv}__{gps_outage_duration}".
                    # Model name.
                    model_name = id_config.split("__")[0]
                    # Training type.
                    training_type = id_config.split("__")[1].capitalize()
                    # Coeff frequency division (convert to int).
                    coeff_frequency_division = int(id_config.split("__")[2])
                    # Adversarial example (format the text).
                    adversarial_example_formatting = {"without_adv": "Not used", "with_imu_adv": "On IMU data", "with_full_adv": "On every data"}
                    adversarial_example = adversarial_example_formatting.get(id_config.split("__")[3])
                    # GPS outage duration (format the text)    .            .
                    gps_outage_duration = f"{":".join(id_config.split("__")[4].split("-")):s} to {":".join(id_config.split("__")[5].split("-")):s}"
        
                    # Load the evaluation results.
                    with open(evaluation_results_file, "r") as file:
                        evaluation_results = json.load(fp=file)
        
                    # Get the subcase dashboard data dictionnary (initialize an empty dict if the case and subcase do not already exit).
                    subcase_dashboard_data = evaluation_dashboard_data.setdefault(f"{case:s}", {}).setdefault(f"{subcase:s}", {})
        
                    # Add the model to the subcase dashboard data (initialize an empty list if the subcase does not already contain any model data).
                    subcase_dashboard_data.setdefault("Training type", []).append(training_type)
                    subcase_dashboard_data.setdefault("GPS outage duration capacity", []).append(gps_outage_duration)
                    subcase_dashboard_data.setdefault("Model name", []).append(model_name)
                    subcase_dashboard_data.setdefault("Coeff freq div", []).append(coeff_frequency_division)
                    subcase_dashboard_data.setdefault("Adversarial example", []).append(adversarial_example)
                    subcase_dashboard_data.setdefault("AVE (in m/s)", []).append(round(evaluation_results["average_velocity_error"], 4))
                    subcase_dashboard_data.setdefault("RMVE (%)", []).append(round(100 * evaluation_results["relative_maximum_velocity_error"], 2))
                    subcase_dashboard_data.setdefault("STDVE (in m/s)", []).append(round(evaluation_results["std_velocity_error"], 4))
                    subcase_dashboard_data.setdefault("ATE (in m)", []).append(round(evaluation_results["average_trajectory_error"], 2))
                    subcase_dashboard_data.setdefault("RMTE (%)", []).append(round(100 * evaluation_results["relative_maximum_trajectory_error"], 2))
                    subcase_dashboard_data.setdefault("STDTE (in m)", []).append(round(evaluation_results["std_trajectory_error"], 2))
                    subcase_dashboard_data.setdefault("RDE", []).append(round(evaluation_results["relative_distance_error"], 4))
                    subcase_dashboard_data.setdefault("RFTTE (%)", []).append(round(100 * evaluation_results["relative_form_transformed_trajectory_error"], 2))
                    subcase_dashboard_data.setdefault("STE (%)", []).append(round(100 * evaluation_results["scale_error"], 2))
                    subcase_dashboard_data.setdefault("TTE (in m)", []).append(round(evaluation_results["translation_error"], 2))
                    subcase_dashboard_data.setdefault("RTE (in deg)", []).append(round(evaluation_results["rotation_error"], 2))

        return evaluation_dashboard_data

    @classmethod
    @st.cache_data(show_spinner="Collecting averaging data…")
    def collect_averaging_data(
        cls: Type[TypeVar("DashboardDataCollector")],
        path_evaluation_results: pathlib.Path,
    ) -> Dict[str, Dict[str, Dict[str, List[Any]]]]:

        """
        Collect the averaging data and return it as a dictionnary constructed as follows:
        
            case
            |____ gps_outage_duration
                  |____ Level (%): List[int]
                  |____ AVE (in m/s): List[float]
                  |____ RMVE (%): List[float]
                  |____ STDVE (in m/s): List[float]
                  |____ ATE (in m): List[float]
                  |____ RMTE (%): List[float]
                  |____ STDTE (in m): List[float]
                  |____ RDE: List[float]

        Args:
            - path_evaluation_results (pathlib.Path): The path where are stored the evaluation results.

        Returns:
            - averaging_dashboard_data (Dict[str, Dict[str, Dict[str, List[Any]]]]): The data to use for the averaging dashboard.
        """

        averaging_dashboard_data = {}

        # Iterate over the 3 cases.
        for case in ["beginning", "within", "end"]:
        
            path_case = pathlib.Path(path_evaluation_results, f"{case}/")

            # If the folder of the case does not exist, no model has been evaluate for this case, skip it.
            if not path_case.exists():
                continue

            # Iterate over the subcases of the case.
            for subcase_folder in path_case.iterdir():

                # If the folder name do not contains " to ", this is not a subcase folder, skip it. Subcase folder name format = "MM-SS to MM-SS".
                if not " to " in subcase_folder.name:
                    continue
        
                subcase = ":".join(subcase_folder.name.split("-"))

                # Iterate over every model evaluation results.
                for evaluation_results_file in subcase_folder.iterdir():

                    # If the file name does not contains "evaluation_results_", this is not an evaluation results file (can be a cache file, jupyter checkpoint etc). Skip it.
                    if not "evaluation_results_" in evaluation_results_file.name:
                        continue

                    # If the file is not an averaging configuration evaluation result, skip it.
                    if not "averaging" in evaluation_results_file.name:
                        continue

                    model_selection_level = evaluation_results_file.name[29:-5]

                    # Load the evaluation results.
                    with open(evaluation_results_file, "r") as file:
                        evaluation_results = json.load(fp=file)

                    # Get the subcase dashboard data dictionnary (initialize an empty dict if the case and subcase do not already exit).
                    subcase_dashboard_data = averaging_dashboard_data.setdefault(f"{case:s}", {}).setdefault(f"{subcase:s}", {})

                    # Add the model to the subcase dashboard data (initialize an empty list if the subcase does not already contain any model data).
                    subcase_dashboard_data.setdefault("Level (%)", []).append(int(100 * float(model_selection_level)))
                    subcase_dashboard_data.setdefault("AVE (in m/s)", []).append(round(evaluation_results["average_velocity_error"], 4))
                    subcase_dashboard_data.setdefault("RMVE (%)", []).append(round(100 * evaluation_results["relative_maximum_velocity_error"], 2))
                    subcase_dashboard_data.setdefault("STDVE (in m/s)", []).append(round(evaluation_results["std_velocity_error"], 4))
                    subcase_dashboard_data.setdefault("ATE (in m)", []).append(round(evaluation_results["average_trajectory_error"], 2))
                    subcase_dashboard_data.setdefault("RMTE (%)", []).append(round(100 * evaluation_results["relative_maximum_trajectory_error"], 2))
                    subcase_dashboard_data.setdefault("STDTE (in m)", []).append(round(evaluation_results["std_trajectory_error"], 2))
                    subcase_dashboard_data.setdefault("RDE", []).append(round(evaluation_results["relative_distance_error"], 4))
                    subcase_dashboard_data.setdefault("RFTTE (%)", []).append(round(100 * evaluation_results["relative_form_transformed_trajectory_error"], 2))
                    subcase_dashboard_data.setdefault("STE (%)", []).append(round(100 * evaluation_results["scale_error"], 2))
                    subcase_dashboard_data.setdefault("TTE (in m)", []).append(round(evaluation_results["translation_error"], 2))
                    subcase_dashboard_data.setdefault("RTE (in deg)", []).append(round(evaluation_results["rotation_error"], 2))

        return averaging_dashboard_data