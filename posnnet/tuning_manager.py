import joblib
import json
import numpy as np
import optuna
import pathlib
import random
import torch
from typing import TypeVar, Type, List, Union, Dict

import posnnet.data.dataset
from posnnet.data.dataset import Dataset
import posnnet.data.scalers
from posnnet.hp_spaces import HpSpace
import posnnet.hp_spaces
from posnnet.models import GeneralModel
from posnnet.study_manager import StudyManager
from posnnet.training.trainer import Trainer


class TuningManager:

    @classmethod
    def __objective(
        cls: Type[TypeVar("TuningManager")],
        best_score: Union[float, None],
        save_every_state_dicts: bool,
        trial: optuna.trial.Trial,
        trainer: posnnet.training.trainer.Trainer,
        scalers: posnnet.data.scalers.Scalers,
        hp_space: posnnet.hp_spaces.HpSpace,
        id_config: str,
        model_name: int,
        input_size: int,
        output_size: int,
        len_window: int,
        coeff_frequency_division: int,
        coeff_sampling_training: int,
        coeff_sampling_validation: int,
        n_epochs_sampling: int,
        sessions_id_training: List[int],
        sessions_id_validation: List[int],
        features_name_accelerometer: List[str],
        features_name_gyroscope: List[str],
        features_name_magnetometer: List[str],
        features_name_velocity_gps: List[str],
        features_name_velocity_fusion: List[str],
        features_name_orientation_fusion: List[str],
        dtype: str,
        random_seed: int,
        path_data_training: pathlib.Path,
        path_data_validation: pathlib.Path,
        path_tuning_results: pathlib.Path
    ) -> float:

        """
        Perform one trial experimentation (objective function for optuna framework).

        Args:
            - trial (optuna.trial.Trial): The optuna trial object of the actual trial.
            - best_score (Union[float, None]): The actual best score for this configuration. None = first set tried.
            - save_every_state_dicts (bool): Either to save or not every model state dicts (if False, only the best model state dict is kept).
            - trainer (posnnet.training.trainer.Trainer): Instance of Trainer that manage the training and scoring.
            - scalers (posnnet.data.scalers.Scalers): The project Scalers instance.
            - hp_space (posnnet.hp_spaces.HpSpace): Instance of HpSpace that manage the hyperparameter sampling.
            - id_config (str): A unique id which represents the configuration.
            - model_name (str): The name of the model for which perform the trial.
            - input_size (int): The number of features of the sensor.
            - output_size (int): The number of fusion velocity axis (number of predicted dimensions).
            - len_window (int): The length of a window provided to the neurel network in samples (before frequency division).
            - coeff_frequency_division (int): The ratio of frequency division (e.g. a window of 2400 samples with a frequency division of 10 leads to a sequence of 240 samples).
            - coeff_sampling_training (int): The ratio of sample that will be used as window beginning every epoch for the training dataset (e.g. coeff_sampling = 1000 means 1/1000 of the dataset sample will be used to create a window).
            - coeff_sampling_validation (int): The ratio of sample that will be used as window beginning every epoch for the validation dataset (e.g. coeff_sampling = 1000 means 1/1000 of the dataset sample will be used to create a window).
            - n_epochs_sampling (int): The number of epochs for a complete sampling rotation of the dataset.
            - sessions_id_training (List[int]): The list of sessions id to use for the training dataset.
            - sessions_id_validation (List[int]): The list of sessions id to use for the validation dataset.
            - features_name_accelerometer (List[str]): The list of features name for the accelerometer (0 <= len(features_name_accelerometer) <= 3).
            - features_name_gyroscope (List[str]): The list of features name for the gyroscope (0 <= len(features_name_gyroscope) <= 3).
            - features_name_magnetometer (List[str]): The list of features name for the magnetometer (0 <= len(features_name_magnetometer) <= 3).
            - features_name_velocity_gps (List[str]): The list of features name for the GPS velocity (0 <= len(features_name_velocity_gps) <= 3).
            - features_name_velocity_fusion (List[str]): The list of features name for the fusion velocity (1 <= len(features_name_velocity_fusion) <= 3).
            - features_name_orientation_fusion (List[str]): The list of features name for the fusion orientation (0 <= len(features_name_orientation_fusion) <= 3).
            - dtype (str): The dtype to use for the model and the data (for example: 'float16', 'float32', 'float64').
            - random_seed (int): A positive integer that is used to ensure determinism during GPS outage simulation generation.
            - path_data_training (pathlib.Path): The path where are stored the training data.
            - path_data_validation (pathlib.Path): The path where are stored the validation data.
            - path_tuning_results (pathlib.Path): The path where will be stored the tuning results.

        Returns:
            - best_loss_achieved (float): The best (lower) velocity loss achieved in validation.
        """

        # Initialize model and training params.
        model_params = {
            "input_size": input_size,
            "output_size": output_size
        }
        training_params = {}

        # Sample model and training hyperparameters.
        model_params = hp_space.sample_model_hp(trial=trial, model_params=model_params)
        training_params = hp_space.sample_training_hp(trial=trial, training_params=training_params)

        # Set random seed to ensure determinism during model creation.
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)

        # Initialize the model.
        model = GeneralModel(
            model_name=model_name,
            model_params=model_params
        )

        # Initialize training and validation dataset.
        dataset_train = Dataset(
            len_window=len_window,
            coeff_frequency_division=coeff_frequency_division,
            coeff_sampling=coeff_sampling_training,
            n_epochs_sampling=n_epochs_sampling,
            scaling_type=training_params["scaling_type"],
            scalers=scalers,
            sessions_id=sessions_id_training,
            features_name_accelerometer=features_name_accelerometer,
            features_name_gyroscope=features_name_gyroscope,
            features_name_magnetometer=features_name_magnetometer,
            features_name_velocity_gps=features_name_velocity_gps,
            features_name_velocity_fusion=features_name_velocity_fusion,
            features_name_orientation_fusion=features_name_orientation_fusion,
            dtype=getattr(np, dtype),
            path_data=path_data_training
        )
        dataset_val = Dataset(
            len_window=len_window,
            coeff_frequency_division=coeff_frequency_division,
            coeff_sampling=coeff_sampling_validation,
            n_epochs_sampling=n_epochs_sampling,
            scaling_type=training_params["scaling_type"],
            scalers=scalers,
            sessions_id=sessions_id_validation,
            features_name_accelerometer=features_name_accelerometer,
            features_name_gyroscope=features_name_gyroscope,
            features_name_magnetometer=features_name_magnetometer,
            features_name_velocity_gps=features_name_velocity_gps,
            features_name_velocity_fusion=features_name_velocity_fusion,
            features_name_orientation_fusion=features_name_orientation_fusion,
            dtype=getattr(np, dtype),
            path_data=path_data_validation
        )

        # Perform the training and scoring.
        best_loss_achieved, associated_ate, best_model_state_dict, batch_size_train, n_accumulations, scores_on_fixed_training_case = trainer.train_and_score(
            model=model,
            dataset_train=dataset_train,
            dataset_val=dataset_val,
            batch_size=training_params["batch_size"],
            learning_rate=training_params["learning_rate"],
            weight_decay=training_params["weight_decay"],
            alpha=training_params["alpha"],
            beta=training_params.get("beta", None),
            epsilon=training_params.get("epsilon", None),
            scaling_type=training_params["scaling_type"]
        )

        # Add the effective batch size and number of gradient accumulations during training.
        training_params.update({
            "batch_size_train": batch_size_train,
            "n_accumulations": n_accumulations
        })

        results = {
            "velocity_loss": best_loss_achieved,
            "ate": associated_ate
        }

        if scores_on_fixed_training_case is not None:
            results["scores_on_fixed_training_case"] = scores_on_fixed_training_case

        # Create a subdirectory to store results, model state dict and parameters.
        path_tuning_result = pathlib.Path(path_tuning_results, f"./{id_config:s}/trial {trial.number + 1:03d}/")
        path_tuning_result.mkdir(parents=True)

        # Save results
        with open(pathlib.Path(path_tuning_result, "results.json"), "w") as file_results:
            json.dump(obj=results, fp=file_results)

        # Save state dict only if save_every_state_dicts == True or if first trial or if a new best score has been achieved.
        if save_every_state_dicts or best_score is None or best_loss_achieved < best_score:
            torch.save(obj=best_model_state_dict, f=pathlib.Path(path_tuning_result, "state_dict.pt"))

        # Save parameters.
        joblib.dump(value=model_params, filename=pathlib.Path(path_tuning_result, "model_params.pkl"))
        joblib.dump(value=training_params, filename=pathlib.Path(path_tuning_result, "training_params.pkl"))

        # Save the trial
        joblib.dump(value=trial, filename=pathlib.Path(path_tuning_result, "trial.pkl"))

        return best_loss_achieved

    @classmethod
    def run_tuning(
        cls: Type[TypeVar("TuningManager")],
        n_experiments: int,
        n_startup_trials: int,
        save_every_state_dicts: bool,
        scalers: posnnet.data.scalers.Scalers,
        relax_points: Dict[str, Dict[str, np.ndarray]],
        id_config: str,
        model_name: str,
        use_adversarial: Union[str, None],
        training_type: str,
        velocity_loss: str,
        coeff_frequency_division: int,
        frequency: int,
        n_minimum_seconds_operational_gps: int,
        min_gps_outage_duration_seconds: int,
        max_gps_outage_duration_seconds: int,
        coeff_sampling_training: int,
        coeff_sampling_validation: int,
        n_epochs_sampling: int,
        n_epochs: int,
        patience: Union[int, None],
        n_epochs_training_checkpoint: int,
        sessions_id_training: List[int],
        sessions_id_validation: List[int],
        features_name_accelerometer: List[str],
        features_name_gyroscope: List[str],
        features_name_magnetometer: List[str],
        features_name_velocity_gps: List[str],
        features_name_velocity_fusion: List[str],
        features_name_orientation_fusion: List[str],
        num_workers: int,
        use_mixed_precision: bool,
        dtype: str,
        device: torch.device,
        random_seed: int,
        verbosity: int,
        path_data_training: pathlib.Path,
        path_data_validation: pathlib.Path,
        path_temp: pathlib.Path,
        path_tuning_results: pathlib.Path,
        path_studies: pathlib.Path
    ) -> None:

        """
        Perform the tuning of an architecture.

        Args:
            - n_experiments (int): The number of experiments (trials) to perform.
            - n_startup_trials (int): The number of experiments randomly sampled before using Bayesian optimization to search in the hyperparameter space.
            - save_every_state_dicts (bool): Either to save or not every model state dicts (if False, only the best model state dict is kept).
            - scalers (posnnet.data.scalers.Scalers): The project Scalers instance.
            - relax_points (Dict[str, Dict[str, np.ndarray]]): The relax points (scaled) of the dataset (velocity GPS = 0 / velocity fusion = 0 / orientation fusion = 0).
            - id_config (str): A unique id which represents the configuration.
            - model_name (str): The name of the model for which perform the trial.
            - use_adversarial (Union[str, None]): Either to use adversarial example during training on IMU data ('imu'), on all the input ('full') or not (None).
            - training_type (str): The type of training (GPS outage placement). Can be either 'beginning', 'centered', 'end' or 'random'.
            - velocity_loss (str): The type of loss function used to optimize and score the model. Can be either 'mae' or 'mse'.
            - coeff_frequency_division (int): The ratio of frequency division (e.g. a window of 2400 samples with a frequency division of 10 leads to a sequence of 240 samples).
            - frequency (int): The original frequency of the sensor.
            - n_minimum_seconds_operational_gps (int): The minimum number of seconds of operational GPS that can be ensured next to every GPS outage.
            - min_gps_outage_duration_seconds (int): The minimal length of a GPS outage in seconds.
            - max_gps_outage_duration_seconds (int): The maximal length of a GPS outage in seconds.
            - coeff_sampling_training (int): The ratio of sample that will be used as window beginning every epoch for the training dataset (e.g. coeff_sampling = 1000 means 1/1000 of the dataset sample will be used to create a window).
            - coeff_sampling_validation (int): The ratio of sample that will be used as window beginning every epoch for the validation dataset (e.g. coeff_sampling = 1000 means 1/1000 of the dataset sample will be used to create a window).
            - n_epochs_sampling (int): The number of epochs for a complete sampling rotation of the dataset.
            - n_epochs (int): The number of epochs for which train the models (if early stopping is not triggered).
            - patience (Union[int, None]): The number of epochs without validation results improvement before early stopping training.
                                           If None, training is never early stopped.
            - n_epochs_training_checkpoint (int): The number of epochs of training after which to save a checkpoint.
            - sessions_id_training (List[int]): The list of sessions id to use for the training dataset.
            - sessions_id_validation (List[int]): The list of sessions id to use for the validation dataset.
            - features_name_accelerometer (List[str]): The list of features name for the accelerometer (0 <= len(features_name_accelerometer) <= 3).
            - features_name_gyroscope (List[str]): The list of features name for the gyroscope (0 <= len(features_name_gyroscope) <= 3).
            - features_name_magnetometer (List[str]): The list of features name for the magnetometer (0 <= len(features_name_magnetometer) <= 3).
            - features_name_velocity_gps (List[str]): The list of features name for the GPS velocity (0 <= len(features_name_velocity_gps) <= 3).
            - features_name_velocity_fusion (List[str]): The list of features name for the fusion velocity (1 <= len(features_name_velocity_fusion) <= 3).
            - features_name_orientation_fusion (List[str]): The list of features name for the fusion orientation (0 <= len(features_name_orientation_fusion) <= 3).
            - num_workers (int): The number of process that will instanciate a dataloader instance to load the data in parallel. 0 means that*
                                 the dataloader will be loaded in the main process. A rule of thumb is to use 4 processes by GPU.
            - use_mixed_precision (bool): Either to use or not float16 mixed precision for training.
            - dtype (str): The dtype to use for the model and the data (for example: 'float16', 'float32', 'float64').
            - device (torch.device): The device on which perform the training and inference.
            - random_seed (int): A positive integer that is used to ensure determinism during GPS outage simulation generation.
            - verbosity (int): The level of verbosity: 0 = critical error / 1 = final results verbose / 2 = verbose every epochs.
            - path_data_training (pathlib.Path): The path where are stored the training data.
            - path_data_validation (pathlib.Path): The path where are stored the validation data.
            - path_temp (pathlib.Path): The path where temporary files are stored.
            - path_tuning_results (pathlib.Path): The path where will be stored the tuning results.
            - path_studies (pathlib.Path): The path where will be stored the study object of the tuning.
        """

        # Remove optuna message except critical if verbosity < 1
        if verbosity < 1:
            optuna.logging.set_verbosity(optuna.logging.CRITICAL)

        # Initialize the study manager
        study_manager = StudyManager(
            id_config=id_config,
            path_studies=path_studies,
            path_tuning_results=path_tuning_results,
        )

        # Load the study object if exists (resume tuning) or create a new study.
        study = study_manager.load()

        # Compute the input and output sizes.
        input_size = len(features_name_accelerometer) + \
                     len(features_name_gyroscope) + \
                     len(features_name_magnetometer) + \
                     len(features_name_velocity_gps) + \
                     len(features_name_velocity_fusion) + \
                     len(features_name_orientation_fusion)
        output_size = len(features_name_velocity_fusion)


        # Compute the different lengths.
        len_window = (max_gps_outage_duration_seconds + n_minimum_seconds_operational_gps) * frequency # Multiply the window duration with frequency to have the number of sample of a window.
        min_len_window = (min_gps_outage_duration_seconds + n_minimum_seconds_operational_gps) * frequency # Multiply the minimal window duration with frequency to have the number of sample of a window.
        len_seq = len_window // coeff_frequency_division # Divide the number of sample of a window by the frequency division coeff to have the input length.
        min_len_seq = min_len_window // coeff_frequency_division # Divide the number of sampler of the minimal window by the frequency division coeff to have the minimal input length.
        min_len_gps_outage = min_gps_outage_duration_seconds * frequency // coeff_frequency_division
        max_len_gps_outage = max_gps_outage_duration_seconds * frequency // coeff_frequency_division        

        # Initialize the hyperparameters space.
        hp_space = HpSpace(
            model_name=model_name,
            use_adversarial=use_adversarial,
            min_len_seq=min_len_seq
        )

        # Initialize trainer.
        trainer = Trainer(
            id_config=id_config,
            scalers=scalers,
            relax_points=relax_points,
            use_adversarial=use_adversarial,
            training_type=training_type,
            velocity_loss=velocity_loss,
            len_seq=len_seq,
            min_len_gps_outage=min_len_gps_outage,
            max_len_gps_outage=max_len_gps_outage,
            coeff_frequency_division=coeff_frequency_division,
            frequency=frequency,
            n_epochs=n_epochs,
            patience=patience,
            n_epochs_training_checkpoint=n_epochs_training_checkpoint,
            random_seed=random_seed,
            num_workers=num_workers,
            use_mixed_precision=use_mixed_precision,
            device=device,
            dtype=getattr(torch, dtype),
            verbosity=verbosity >= 2,
            path_temp=path_temp
        )

        # Iterate until N successfull experiments have been conducted.
        while len([trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]) < n_experiments:

            # Set the sampler with different random seed each trial.
            # This is done to ensure that, when a study is reconstruct (in case of corrupted file that cannot be loaded),
            # the sampler will not select again the same sets.
            study.sampler = optuna.samplers.TPESampler(
                n_startup_trials=n_startup_trials,
                seed=random_seed + len(study.trials)
            )

            try:
                best_score = study.best_value
                best_trial_number = study.best_trial.number
            except:
                best_score = None
                best_trial_number = None

            # Perform one experiment.
            study.optimize(
                func=lambda trial: cls.__objective(
                    trial=trial,
                    best_score=best_score,
                    save_every_state_dicts=save_every_state_dicts,
                    trainer=trainer,
                    scalers=scalers,
                    hp_space=hp_space,
                    id_config=id_config,
                    model_name=model_name,
                    input_size=input_size,
                    output_size=output_size,
                    len_window=len_window,
                    coeff_frequency_division=coeff_frequency_division,
                    coeff_sampling_training=coeff_sampling_training,
                    coeff_sampling_validation=coeff_sampling_validation,
                    n_epochs_sampling=n_epochs_sampling,
                    sessions_id_training=sessions_id_training,
                    sessions_id_validation=sessions_id_validation,
                    features_name_accelerometer=features_name_accelerometer,
                    features_name_gyroscope=features_name_gyroscope,
                    features_name_magnetometer=features_name_magnetometer,
                    features_name_velocity_gps=features_name_velocity_gps,
                    features_name_velocity_fusion=features_name_velocity_fusion,
                    features_name_orientation_fusion=features_name_orientation_fusion,
                    dtype=dtype,
                    random_seed=random_seed,
                    path_data_training=path_data_training,
                    path_data_validation=path_data_validation,
                    path_tuning_results=path_tuning_results
                ),
                n_trials=1,
                timeout=None,
                n_jobs=1,
                catch=[Exception]
            )

            # In case of new best score, delete the state dict of the previous best model
            if best_score is not None and study.best_value < best_score:
                path_previous_best_state_dict = pathlib.Path(path_tuning_results, f"./{id_config:s}/trial {best_trial_number + 1:03d}/state_dict.pt")
                path_previous_best_state_dict.unlink()

            # Save the study object on disk.
            study_manager.save(study=study)

    @classmethod
    def check_coeff_frequency_division_validity(
        cls: Type[TypeVar("TuningManager")],
        frequency: int,
        coeff_frequency_division: int,
        n_minimum_seconds_operational_gps: int,
    ) -> bool:

        """
        Determine if the coeff frquency division is valid regarding the specification of the project.
        To be valid, coeff_freq have to validate:
            - coeff_frequency_division <= n_minimum_seconds_operational_gps * frequency // 2
            - coeff_frequency_division <= frequency
            - frequency % coeff_frequency_division == 0

        Args:
            - frequency (int): The original frequency of the sensor.
            - coeff_frequency_division (int): The ratio of frequency division (e.g. a window of 2400 samples with a frequency division of 10 leads to a sequence of 240 samples).
            - n_minimum_seconds_operational_gps (int): The minimum number of seconds of operational GPS that can be ensured next to every GPS outage.

        Returns:
            - is_coeff_valid (bool): Indicates if the coefficient of frequency division is valid or not.
        """

        coeff_max = min(n_minimum_seconds_operational_gps * frequency // 2, frequency)
        
        if coeff_frequency_division > coeff_max:

            is_coeff_valid = False
            print(f"The selected coefficient of frequency division is not valid. To be valid, the coefficient has to validate both 'coeff_frequency_division <= n_minimum_seconds_operational_gps * frequency // 2' and 'coeff_frequency_division <= frequency'. In your case, the coefficient has to be inferior or equal to {coeff_max:d}.")

        elif frequency % coeff_frequency_division != 0:

            is_coeff_valid = False
            print(f"The selected coefficient of frequency division is not valid. To be valid, the coefficient has to validate 'frequency % coeff_frequency_division == 0'. You selected a coefficient value of {coeff_frequency_division:d} and you have a frequency of {frequency:d} Hz, leading to a rest in the frequency division.")

        else:

            is_coeff_valid = True

        return is_coeff_valid