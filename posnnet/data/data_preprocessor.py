import numpy as np
import pandas as pd
import pathlib
import random
from typing import TypeVar, Type, Union, List, Tuple


class DataPreprocessor:

    @classmethod
    def preprocess_data(
        cls: Type[TypeVar("DataPreprocessor")],
        csv_sep: str,
        csv_encoding: str,
        sensor_frequency: int,
        n_minimum_seconds_operational_gps: int,
        training_sessions_id: List[int],
        validation_sessions_id: List[int],
        evaluation_sessions_id: List[int],
        n_duplications_per_eval_session: Union[str, int],
        gps_outage_durations_eval_at_beginning: List[Tuple[str, str]],
        gps_outage_durations_eval_within: List[Tuple[str, str]],
        gps_outage_durations_eval_at_end: List[Tuple[str, str]],
        columns_to_load: List[str],
        path_raw_data: pathlib.Path,
        path_preprocessed_training_data: pathlib.Path,
        path_preprocessed_validation_data: pathlib.Path,
        path_preprocessed_evaluation_data: pathlib.Path,
        random_seed: int,
        verbosity: bool
    ) -> None:

        """
        Perform the preprocessing of the data:
            - For training sessions, the sessions data will be loaded (only the usefull columns) and then saved as pickle files.
            - For validatoon sessions, the sessions data will be loaded (only the usefull columns) and then saved as pickle files.
            - For evaluation session, the sessions data will be loaded (only the usefull columns) and then several GPS outage will be simulated 
              for every case and every GPS outage duration (by creating several columns nammed gps_outage_1, gps_outage_2, ...) 
              and finally the data are saved as pickle files.

        Args:
            - csv_sep (str): The character used as delimiter in the CSV files containing the sessions data (e.g. ',').
            - csv_encoding (str): The encoding used to encode the data in the CSV files containing the sessions data (e.g. 'utf-8').
            - sensor_frequency (int): The frequency at which the sensor provide data in Hertz.
            - n_minimum_seconds_operational_gps (int): The minimum number of seconds of operational GPS that can be ensured next to every GPS outage.
            - training_sessions_id (List[int]): The list of the sessions id to use for the training dataset.
            - validation_sessions_id (List[int]): The list of the sessions id to use for the validation dataset.
            - evaluation_sessions_id (List[int]): The list of the sessions id to use for the evaluation dataset.
            - n_duplications_per_eval_session (Union[str, int]): Number of GPS outages simulated for each evaluation session. Can be either
                                                                 an integer (same number for every session) or 'light' or 'normal' or 'extensive'.
            - gps_outage_durations_eval_at_beginning (List[Tuple[str, str]]): The list of cases of GPS outage durations that will be simulated at the beginning, 
                                                                              following the format [(N1, N2), (N3, N4), ...] to simulate GPS outage from 
                                                                              N1 to N2 seconds and N3 to N4 seconds etc. Format of Nx: "MM:SS".
            - gps_outage_durations_eval_within (List[Tuple[str, str]]): The list of cases of GPS outage durations that will be simulated within the session, 
                                                                        following the format [(N1, N2), (N3, N4), ...] to simulate GPS outage from 
                                                                        N1 to N2 seconds and N3 to N4 seconds etc. Format of Nx: "MM:SS".
            - gps_outage_durations_eval_at_end (List[Tuple[str, str]]): The list of cases of GPS outage durations that will be simulated at the end, 
                                                                        following the format [(N1, N2), (N3, N4), ...] to simulate GPS outage from 
                                                                        N1 to N2 seconds and N3 to N4 seconds etc. Format of Nx: "MM:SS".
            - columns_to_load (List[str]): The list of colunms to load in the CSV files.
            - path_raw_data (pathlib.Path): The path of the directory where are stored raw data.
            - path_preprocessed_training_data (pathlib.Path): The path of the directory where will be stored preprocessed training data.
            - path_preprocessed_validation_data (pathlib.Path): The path of the directory where will be stored preprocessed validation data.
            - path_preprocessed_evaluation_data (pathlib.Path): The path of the directory where will be stored preprocessed evaluation data.
            - random_seed (int): A positive integer that is used to ensure determinism of sampling during the data preprocessing.
            - verbosity (bool): Indicate if the framework should print verbose to inform of the progress of the data preprocessing.
        """

        # Set random seed for reproductibility.
        random.seed(a=random_seed, version=2)

        # Read the training sessions (only the usefull columns) and save them as pickle files inside the preprocessed training data directory.
        for i_session, session_id in enumerate(training_sessions_id):
            
            pd.read_csv(
                filepath_or_buffer=pathlib.Path(path_raw_data, f"session_{session_id:d}.csv"),
                sep=csv_sep,
                encoding=csv_encoding,
                usecols=columns_to_load,
            ).to_pickle(
                path=pathlib.Path(path_preprocessed_training_data, f"session_{session_id:d}.pkl"),
                protocol=5
            )

            # Print progress.
            if verbosity == True:
                print(
                    f"Preprocessing of the training data: {100 * (i_session + 1) / len(training_sessions_id):.2f}%", 
                    end="\r" if i_session + 1 < len(training_sessions_id) else "\n"
                )

        # Read the validation sessions (only the usefull columns) and save them as pickle files inside the preprocessed validation data directory.
        for i_session, session_id in enumerate(validation_sessions_id):
            
            pd.read_csv(
                filepath_or_buffer=pathlib.Path(path_raw_data, f"session_{session_id:d}.csv"),
                sep=csv_sep,
                encoding=csv_encoding,
                usecols=columns_to_load,
            ).to_pickle(
                path=pathlib.Path(path_preprocessed_validation_data, f"session_{session_id:d}.pkl"),
                protocol=5
            )

            # Print progress.
            if verbosity == True:
                print(
                    f"Preprocessing of the validation data: {100 * (i_session + 1) / len(validation_sessions_id):.2f}%", 
                    end="\r" if i_session + 1 < len(validation_sessions_id) else "\n"
                )

        # Read evaluation sessions and stock them into a dict (key = session id / value = pandas dataframe with session data).
        dfs_eval = {}
        
        for i_session, session_id in enumerate(evaluation_sessions_id):
            
            dfs_eval[session_id] = pd.read_csv(
                filepath_or_buffer=pathlib.Path(path_raw_data, f"session_{session_id:d}.csv"),
                sep=csv_sep,
                encoding=csv_encoding,
                usecols=columns_to_load,
            )

            # Print progress.
            if verbosity == True:
                print(
                    f"Loading of the evaluation data: {100 * (i_session + 1) / len(evaluation_sessions_id):.2f}%", 
                    end="\r" if i_session + 1 < len(evaluation_sessions_id) else "\n"
                )

        duplications_by_subcase = []
        
        # Transform evaluation sessions.
        # Iterate over the 3 cases: GPS outage at the beginning, within the session and at the end.
        for case, gps_outage_durations in [
            ("beginning", gps_outage_durations_eval_at_beginning),
            ("within", gps_outage_durations_eval_within),
            ("end", gps_outage_durations_eval_at_end)
        ]:

            gps_outage_durations = cls.__convert_string_gps_outage_durations_to_seconds(gps_outage_durations=gps_outage_durations)

            # Iterate over the durations for the case (subcases)
            for n_minimum_seconds_gps_outage, n_maximum_seconds_gps_outage in gps_outage_durations:

                string_minimum_time_gps_outage = f"{n_minimum_seconds_gps_outage // 60:02d}-{n_minimum_seconds_gps_outage % 60:02d}"
                string_maximum_time_gps_outage = f"{n_maximum_seconds_gps_outage // 60:02d}-{n_maximum_seconds_gps_outage % 60:02d}"

                # Initialize the directory where will be stored the subcase data.
                # E.g. /within/00-30 to 01-30/
                path_preprocessed_evaluation_subcase_data = pathlib.Path(
                    path_preprocessed_evaluation_data,
                    f"{case:s}/{string_minimum_time_gps_outage:s} to {string_maximum_time_gps_outage:s}/"
                )
                path_preprocessed_evaluation_subcase_data.mkdir(parents=True, exist_ok=True)

                total_n_duplications = 0

                # Iterate over evaluation sessions
                for i_session, (session_id, df_eval) in enumerate(dfs_eval.items()):

                    # Copy the dataframe (to ensure that the GPS outage simulation is not done on the original dataframe that will be used for every subcase).
                    df_eval = df_eval.copy(deep=True)

                    # Compute the duration of the evaluation session
                    session_duration_in_seconds = df_eval.shape[0] // sensor_frequency

                    # If the session is not long enough for the subcase, skip it
                    if not session_duration_in_seconds > n_minimum_seconds_gps_outage + n_minimum_seconds_operational_gps:
                        # Print progress.
                        if verbosity == True:
                            print(
                                f"GPS outage simulation for case '{case:s}' from {string_minimum_time_gps_outage:s} to {string_maximum_time_gps_outage:s}: {100 * (i_session + 1) / len(evaluation_sessions_id):.2f}%", 
                                end="\r" if i_session + 1 < len(evaluation_sessions_id) else "\n"
                            )
                        continue

                    # Compute the number of duplication for this evaluation session
                    n_duplications = cls.__calculate_n_duplications(
                        case=case,
                        n_minimum_seconds_operational_gps=n_minimum_seconds_operational_gps,
                        n_duplications_per_eval_session=n_duplications_per_eval_session,
                        n_minimum_seconds_gps_outage=n_minimum_seconds_gps_outage,
                        n_maximum_seconds_gps_outage=n_maximum_seconds_gps_outage,
                        session_duration_in_seconds=session_duration_in_seconds
                    )

                    total_n_duplications += n_duplications

                    # Generate the synthetic duplications
                    df_eval = cls.__synthetic_generations_eval_session(
                        case=case,
                        df_eval=df_eval,
                        sensor_frequency=sensor_frequency,
                        n_minimum_seconds_operational_gps=n_minimum_seconds_operational_gps,
                        n_minimum_seconds_gps_outage=n_minimum_seconds_gps_outage,
                        n_maximum_seconds_gps_outage=n_maximum_seconds_gps_outage,
                        n_duplications=n_duplications,
                    )

                    # Save the evaluation session
                    df_eval.to_pickle(
                        path=pathlib.Path(path_preprocessed_evaluation_subcase_data, f"session_{session_id:d}.pkl"), 
                        protocol=5
                    )

                    # Print progress.
                    if verbosity == True:
                        print(
                            f"GPS outage simulation for case '{case:s}' from {string_minimum_time_gps_outage:s} to {string_maximum_time_gps_outage:s}: {100 * (i_session + 1) / len(evaluation_sessions_id):.2f}%", 
                            end="\r" if i_session + 1 < len(evaluation_sessions_id) else "\n"
                        )

                # Add case to report
                duplications_by_subcase.append((case, string_minimum_time_gps_outage, string_maximum_time_gps_outage, total_n_duplications))

                # In case no GPS outages have been simulate on this case (sessions duration too short), remove the evaluation data folder for this subcase.
                if total_n_duplications == 0:
                    path_preprocessed_evaluation_subcase_data.rmdir()

        # Print report.
        print("*" * 100)
        print("*" * 46 + " REPORT " + "*" * 46)
        print("*" * 100)
        for case, string_minimum_time_gps_outage, string_maximum_time_gps_outage, total_n_duplications in duplications_by_subcase:
            if total_n_duplications > 0:
                print(f"Case {case:s} with GPS outage from {string_minimum_time_gps_outage:s} to {string_maximum_time_gps_outage:s}: {total_n_duplications:d} GPS outages simulated.")
        print("*" * 100)

        # If, for some case, no duplications have been generated because the length of the sessions was not enough, inform the user.
        if any([
            total_n_duplications == 0
            for _, _, _, total_n_duplications in duplications_by_subcase
        ]):
            print("*" * 45 + " WARNINGS " + "*" * 45)
            print("*" * 100)
            print("The framework was unable to simulate GPS outage for one or several subcases. The reason is that the duration of your sessions is not enough. For example, if you ask a case with GPS outage that last between 30 seconds and 50 seconds, with 4 seconds of operational GPS, the framework will only be able to simulate GPS outages on sessions for which the duration is at least 34 seconds. Here is the list of the GPS outage subcases that have not been treated:")
            for case, string_minimum_time_gps_outage, string_maximum_time_gps_outage, total_n_duplications in duplications_by_subcase:
                if total_n_duplications == 0:
                    print(f"\t- Case {case:s} with GPS outage from {string_minimum_time_gps_outage:s} to {string_maximum_time_gps_outage:s}")
            print("\nYou have two choices. Option 1: If you are able to provide sessions with longer duration, add these data into the project folder (inside the raw data folder). Then, call Project.move_to_previous_stage(desired_stage=ProjectLifeCycleManager.SETTINGS_DEFINITION_STAGE, force_move=True) to come back to the settings definition. Call the method Project.set_dataset_settings after adding the new sessions to the list that already contains the ones you used this time, validate the settings and proceed again to the data preprocessing. Option 2: If you are not able to provide sessions with longer duration, ignore this message and the subcases will not be treated by the framework.")
            print("*" * 100)
            
                
    @classmethod
    def __convert_string_gps_outage_durations_to_seconds(
        cls: Type[TypeVar("DataPreprocessor")],
        gps_outage_durations: List[Tuple[str, str]]
    ) -> List[Tuple[int, int]]:

        """
        Convert GPS outage duration from format ("MM:SS", "MM:SS") to (60 * MM + SS, 60 * MM + SS).

        Args:
            - gps_outage_durations (List[Tuple[str, str]]): The GPS outage durations with a string format ("MM:SS", "MM:SS").

        Returns:
            - gps_outage_duration_in_seconds(List[Tuple[int, int]]): The GPS outage durations in seconds.
        """

        gps_outage_duration_in_seconds = [
            (
                60 * int(gps_outage_duration[0][0:2]) + int(gps_outage_duration[0][3:5]),
                60 * int(gps_outage_duration[1][0:2]) + int(gps_outage_duration[1][3:5]),
            )
            for gps_outage_duration in gps_outage_durations
        ]

        return gps_outage_duration_in_seconds

    @classmethod
    def __calculate_n_duplications(
        cls: Type[TypeVar("DataPreprocessor")],
        case: str,
        n_minimum_seconds_operational_gps: int,
        n_duplications_per_eval_session: Union[str, int],
        n_minimum_seconds_gps_outage: int,
        n_maximum_seconds_gps_outage: int,
        session_duration_in_seconds: int
    ) -> int:

        """
        Compute the number of GPS outages to simulate on the session. If the argument 'n_duplications_per_eval_session' is an integer, 
        the function will return the same values (assuring the value is superior or equal to 1). If the argument is 'light', 'normal' or 
        'extensive', the optimal number of GPS outages will be computed for the session, taking into account the session duration, the GPS outage
        duration and the minimum operational time of GPS around the GPS outages.

        Args:
            - case (str): The case ('beginning', 'within' or 'end') for which the GPS outages will be simulated.
            - n_minimum_seconds_operational_gps (int): The minimum number of seconds of operational GPS that can be ensured next to every GPS outage.
            - n_duplications_per_eval_session (Union[str, int]): Number of GPS outages simulated for each evaluation session. Can be either
                                                                 an integer (same number for every session) or 'light' or 'normal' or 'extensive'.
            - n_minimum_seconds_gps_outage (int): Minimum number of seconds of GPS outage for the subcase.
            - n_maximum_seconds_gps_outage (int): Maximum number of seconds of GPS outage for the subcase.
            - session_duration_in_seconds (int): Duration of the session in seconds.

        Returns:
            - n_duplications (int): The number of GPS outage to simulate on the session (n_duplications >= 1).
        """

        # If the number of duplications is set to an auto mode, compute the number of duplications for this session, 
        # otherwise use the default value provided by the user.
        if n_duplications_per_eval_session in ["light", "normal", "extensive"]:

            # To dertermine the number of variants for a specific session, several parameters will be used
            # First, we compute the real maximum number of seconds of GPS outage that can be simulated on this session while ensuring at least the minimum of GPS operational time
            # Second, we compute the size of the GPS outage duration range (taking the real maximum number of seconds instead of the user defined maximum)
            # Third, we compute the size of the position range that can be picked (use only for the simulation of GPS outage within the session)
            # E.g. The user define a GPS outage duration between 10 and 30 seconds and at least 4 seconds of GPS operational
            # If the session duration is 25 seconds, the real maximum seconds of GPS outage is min(30, 25 - 4) = 21 seconds,
            # the size of the GPS outage duration range is 21 - 10 = 11 seconds and the size of the position range is 25 - 21 - 4 = 0 seconds
            # If the session duration is 45 seconds, the real maximum seconds of GPS outage is min(30, 45 - 4) = 30 seconds,
            # the size of the GPS outage duration range is 30 - 10 = 20 seconds and the size of the position range is 45 - 30 - 4 = 11 seconds
            real_n_maximum_seconds_gps_outage = min(
                n_maximum_seconds_gps_outage,
                session_duration_in_seconds - n_minimum_seconds_operational_gps
            )
            gps_outage_duration_range_in_seconds = real_n_maximum_seconds_gps_outage - n_minimum_seconds_gps_outage
            position_range_size = session_duration_in_seconds - real_n_maximum_seconds_gps_outage - n_minimum_seconds_operational_gps

            if n_duplications_per_eval_session == "light":
                divisor_gps_outage_duration_range = 10
                divisor_position_range = 120
            elif n_duplications_per_eval_session == "normal":
                divisor_gps_outage_duration_range = 5
                divisor_position_range = 45
            elif n_duplications_per_eval_session == "extensive":
                divisor_gps_outage_duration_range = 2
                divisor_position_range = 15
            else:
                pass # Should not occur

            # Compute n_duplications
            # For beginning and end, n_duplications is the division of the size of the GPS outage duration range by the divisor
            # For within session, it is the sum of the division of the size of the GPS outage duration range by the divisor and the division of the size of position range by the divisor
            if case in ["beginning", "end"]:
                n_duplications = gps_outage_duration_range_in_seconds // divisor_gps_outage_duration_range
            elif case in ["within"]:
                n_duplications = (gps_outage_duration_range_in_seconds // divisor_gps_outage_duration_range) + (position_range_size // divisor_position_range)
            else:
                pass # Should not occur

        else:

            n_duplications = n_duplications_per_eval_session

        # Ensure that n_duplications is superior or equal to 1
        n_duplications = max(n_duplications, 1)

        return n_duplications

    @classmethod
    def __synthetic_generations_eval_session(
        cls: Type[TypeVar("DataPreprocessor")],
        case: str,
        df_eval: pd.DataFrame,
        sensor_frequency: int,
        n_minimum_seconds_operational_gps: int,
        n_minimum_seconds_gps_outage: int,
        n_maximum_seconds_gps_outage: int,
        n_duplications: int
    ) -> pd.DataFrame:

        """
        Perform the generation of n_duplications GPS outage on the session (by creating n_duplications columns
        nammed gps_outage_1, gps_outage_2, etc).

        Args:
            - case (str): The case ('beginning', 'within' or 'end') for which the GPS outages will be simulated.
            - df_eval (pd.DataFrame): The dataframe that contains the session data.
            - sensor_frequency (int): The frequency at which the sensor provide data in Hertz.
            - n_minimum_seconds_operational_gps (int): The minimum number of seconds of operational GPS that can be ensured next to every GPS outage.
            - n_minimum_seconds_gps_outage (int): Minimum number of seconds of GPS outage for the subcase.
            - n_maximum_seconds_gps_outage (int): Maximum number of seconds of GPS outage for the subcase.
            - n_duplications (int): The number of GPS outage to simulate on the session (n_duplications >= 1).

        Returns:
            - df_eval (pd.DataFrame): The input dataframe with simulated GPS outages.
        """

        # Select the synthetic generation based on the schema
        synthetic_generation_eval_function_mapping = {
            "beginning": cls.__synthetic_generation_eval_session_gps_outage_at_beginning,
            "within": cls.__synthetic_generation_eval_session_gps_outage_within,
            "end": cls.__synthetic_generation_eval_session_gps_outage_at_end
        }
        synthetic_generation_eval_function = synthetic_generation_eval_function_mapping.get(case)

        # Iterate n_duplications time to create synthetic duplications
        for i_duplication in range(1, n_duplications + 1):

            df_eval = synthetic_generation_eval_function(
                df_eval=df_eval,
                sensor_frequency=sensor_frequency,
                n_minimum_seconds_operational_gps=n_minimum_seconds_operational_gps,
                n_minimum_seconds_gps_outage=n_minimum_seconds_gps_outage,
                n_maximum_seconds_gps_outage=n_maximum_seconds_gps_outage,
                i_duplication=i_duplication
            )

        return df_eval

    @classmethod
    def __synthetic_generation_eval_session_gps_outage_at_beginning(
        cls: Type[TypeVar("DataPreprocessor")],
        df_eval: pd.DataFrame,
        sensor_frequency: int,
        n_minimum_seconds_operational_gps: int,
        n_minimum_seconds_gps_outage: int,
        n_maximum_seconds_gps_outage: int,
        i_duplication: int
    ) -> pd.DataFrame:

        """
        Perform the generation of one GPS outage at the beginning of the session.

        Args:
            - df_eval (pd.DataFrame): The dataframe that contains the session data.
            - sensor_frequency (int): The frequency at which the sensor provide data in Hertz.
            - n_minimum_seconds_operational_gps (int): The minimum number of seconds of operational GPS that can be ensured next to every GPS outage.
            - n_minimum_seconds_gps_outage (int): Minimum number of seconds of GPS outage for the subcase.
            - n_maximum_seconds_gps_outage (int): Maximum number of seconds of GPS outage for the subcase.
            - i_duplication (int): The number to allocate to the GPS outage.

        Returns:
            - df_eval (pd.DataFrame): The input dataframe with the simulated GPS outage.
        """

        # Compute the minimum and maximum number of samples for the GPS outage
        n_samples_minimum_gps_outage = n_minimum_seconds_gps_outage * sensor_frequency
        n_samples_maximum_gps_outage = n_maximum_seconds_gps_outage * sensor_frequency
        # Compute the minimum number of samples to let on the side after the GPS outage
        minimum_samples_side = n_minimum_seconds_operational_gps * sensor_frequency

        # Ensure that the maximum number of samples for the GPS outage allow to let at least minimum_samples_side samples of operational GPS on the side
        n_samples_maximum_gps_outage = min(n_samples_maximum_gps_outage, df_eval.shape[0] - minimum_samples_side)

        # Sample random number and create a new column gps_outage (1 = GPS outage / 0 = GPS operational)
        n_samples_gps_outage = random.randint(a=n_samples_minimum_gps_outage, b=n_samples_maximum_gps_outage)
        df_eval[f"gps_outage_{i_duplication:d}"] = 0
        df_eval.loc[: n_samples_gps_outage - 1, f"gps_outage_{i_duplication:d}"] = 1

        return df_eval

    @classmethod
    def __synthetic_generation_eval_session_gps_outage_within(
        cls: Type[TypeVar("DataPreprocessor")],
        df_eval: pd.DataFrame,
        sensor_frequency: int,
        n_minimum_seconds_operational_gps: int,
        n_minimum_seconds_gps_outage: int,
        n_maximum_seconds_gps_outage: int,
        i_duplication: int
    ) -> pd.DataFrame:

        """
        Perform the generation of one GPS outage within the session.

        Args:
            - df_eval (pd.DataFrame): The dataframe that contains the session data.
            - sensor_frequency (int): The frequency at which the sensor provide data in Hertz.
            - n_minimum_seconds_operational_gps (int): The minimum number of seconds of operational GPS that can be ensured next to every GPS outage.
            - n_minimum_seconds_gps_outage (int): Minimum number of seconds of GPS outage for the subcase.
            - n_maximum_seconds_gps_outage (int): Maximum number of seconds of GPS outage for the subcase.
            - i_duplication (int): The number to allocate to the GPS outage.

        Returns:
            - df_eval (pd.DataFrame): The input dataframe with the simulated GPS outage.
        """

        # Compute the minimum and maximum number of samples for the GPS outage
        n_samples_minimum_gps_outage = n_minimum_seconds_gps_outage * sensor_frequency
        n_samples_maximum_gps_outage = n_maximum_seconds_gps_outage * sensor_frequency
        # Compute the minimum number of samples to let on the side before and after the GPS outage
        minimum_samples_side = n_minimum_seconds_operational_gps * sensor_frequency

        # Ensure that the maximum number of samples for the GPS outage allow to let at least minimum_samples_side samples of operational GPS on the side
        n_samples_maximum_gps_outage = min(n_samples_maximum_gps_outage, df_eval.shape[0] - minimum_samples_side)

        # Sample random numbers and create a new column gps_outage (1 = GPS outage / 0 = GPS operational)
        n_samples_gps_outage = random.randint(a=n_samples_minimum_gps_outage, b=n_samples_maximum_gps_outage)
        position_gps_outage = random.randint(a=minimum_samples_side // 2, b=df_eval.shape[0] - n_samples_gps_outage - minimum_samples_side // 2)
        df_eval[f"gps_outage_{i_duplication:d}"] = 0
        df_eval.loc[position_gps_outage : position_gps_outage + n_samples_gps_outage - 1, f"gps_outage_{i_duplication:d}"] = 1

        return df_eval

    @classmethod
    def __synthetic_generation_eval_session_gps_outage_at_end(
        cls: Type[TypeVar("DataPreprocessor")],
        df_eval: pd.DataFrame,
        sensor_frequency: int,
        n_minimum_seconds_operational_gps: int,
        n_minimum_seconds_gps_outage: int,
        n_maximum_seconds_gps_outage: int,
        i_duplication: int
    ) -> pd.DataFrame:

        """
        Perform the generation of one GPS outage at the end of the session.

        Args:
            - df_eval (pd.DataFrame): The dataframe that contains the session data.
            - sensor_frequency (int): The frequency at which the sensor provide data in Hertz.
            - n_minimum_seconds_operational_gps (int): The minimum number of seconds of operational GPS that can be ensured next to every GPS outage.
            - n_minimum_seconds_gps_outage (int): Minimum number of seconds of GPS outage for the subcase.
            - n_maximum_seconds_gps_outage (int): Maximum number of seconds of GPS outage for the subcase.
            - i_duplication (int): The number to allocate to the GPS outage.

        Returns:
            - df_eval (pd.DataFrame): The input dataframe with the simulated GPS outage.
        """

        # Compute the minimum and maximum number of samples for the GPS outage
        n_samples_minimum_gps_outage = n_minimum_seconds_gps_outage * sensor_frequency
        n_samples_maximum_gps_outage = n_maximum_seconds_gps_outage * sensor_frequency
        # Compute the minimum number of samples to let on the side before the GPS outage
        minimum_samples_side = n_minimum_seconds_operational_gps * sensor_frequency

        # Ensure that the maximum number of samples for the GPS outage allow to let at least minimum_samples_side samples of operational GPS on the side
        n_samples_maximum_gps_outage = min(n_samples_maximum_gps_outage, df_eval.shape[0] - minimum_samples_side)

        # Sample random number and create a new column gps_outage (1 = GPS outage / 0 = GPS operational)
        n_samples_gps_outage = random.randint(a=n_samples_minimum_gps_outage, b=n_samples_maximum_gps_outage)
        df_eval[f"gps_outage_{i_duplication:d}"] = 0
        df_eval.loc[df_eval.shape[0] - n_samples_gps_outage:, f"gps_outage_{i_duplication:d}"] = 1

        return df_eval