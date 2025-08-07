import joblib
import json
import optuna
import pathlib
from typing import Self

class StudyManager:

    def __init__(
        self: Self,
        id_config: str,
        path_studies: pathlib.Path,
        path_tuning_results: pathlib.Path
    ) -> None:

        """
        Initialize a StudyManager instance that manages the loading and saving of optuna study object on disk.

        Args:
            - id_config (str): A unique id which represents the configuration.
            - path_studies (pathlib.Path): The path where are stored the studies.
            - path_tuning_results (pathlib.Path): The path where are stored the tuning results.
        """

        # Initialize the path to the last and old study files.
        self.path_study = pathlib.Path(path_studies, f"study_{id_config:s}.pkl")

        # Initialize the path to the tuning results
        self.path_config_tuning_results = pathlib.Path(path_tuning_results, f"./{id_config:s}/")

    def save(
        self: Self,
        study: optuna.study.Study
    ) -> None:

        """
        Save the study on disk.

        Args:
            - study (optuna.study.Study): The optuna study object to save on disk.
        """

        # Save the new study.
        joblib.dump(value=study, filename=self.path_study)

    def load(
        self: Self
    ) -> optuna.study.Study:

        """
        Load the study object of the configuration if it exists on disk, otherwise create it.

        Return:
            - study (optuna.study.Study): The optuna study object.
        """

        # Load the study on disk.
        try:
            study = joblib.load(filename=self.path_study)
        # Exception in case of no study or corrupted file. Create a new study.
        except:
            study = optuna.create_study(
                direction="minimize"
            )

        # Check if some trials on disk are not in the study.
        # Missing trials can happen in two scenarios:
        #  - Corrupted file (so the study has been recreated)
        #  - Tuning process finish after trial saving but before study saving.
        i_trial = len(study.trials)

        # Iterate while some trials folder of trial not present in the study are found.
        while pathlib.Path(self.path_config_tuning_results, f"trial {i_trial + 1:03d}").exists():

            path_trial = pathlib.Path(self.path_config_tuning_results, f"trial {i_trial + 1:03d}")

            # Try to load the results and the trial object of this trial
            try:
                with open(pathlib.Path(path_trial, "results.json"), "r") as results_file:
                    results = json.load(results_file)
                trial_on_disk = joblib.load(filename=pathlib.Path(path_trial, "trial.pkl"))
            # In case of exception (if results or trial files are not present / corrupted), remove the folder of the disk and continue.
            except:
                shutil.rmtree(path=path_trial)
                i_trial += 1
                continue

            # Create a new trial object and add it to the study as a finished trial.
            trial_to_add = optuna.trial.create_trial(
                state=optuna.trial.TrialState.COMPLETE,
                params=trial_on_disk.params,
                distributions=trial_on_disk.distributions,
                value=results["velocity_loss"],
            )
            study.add_trial(trial=trial_to_add)

            # Pass to next trial
            i_trial += 1

        return study