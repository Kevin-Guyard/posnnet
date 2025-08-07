import joblib
import pathlib
import torch
from typing import Self, Dict, Tuple, Union

import posnnet.models
import posnnet.training.score_manager


class TrainingCheckpoint:

    def __init__(
        self: Self,
        id_config: str,
        n_epochs_training_checkpoint: int,
        path_temp: pathlib.Path
    ) -> None:

        """
        Initialize an instance of TrainingCheckpoint which manage the loading and saving of checkpoint during training
        to restore the training state in case of interruptions.

        Args:
            - id_config (str): A unique id which represents the configuration.
            - n_epochs_training_checkpoint (int): The number of epochs of training after which to save a checkpoint.
            - path_temp (pathlib.Path): The path for temporary files.
        """

        # Memorize parameters.
        self.n_epochs_training_checkpoint = n_epochs_training_checkpoint

        # Initialize the path to the checkpoint and the temporary checkpoint.
        self.path_checkpoint = pathlib.Path(path_temp, f"training_checkpoint_{id_config:s}.pkl")
        self.path_old_checkpoint = pathlib.Path(path_temp, f"training_checkpoint_{id_config:s}_old.pkl")

    def save(
        self: Self,
        model: posnnet.models.GeneralModel,
        optimizer: torch.optim.AdamW,
        mixed_precision_scaler: torch.amp.GradScaler,
        best_model_state_dict: Dict[str, torch.Tensor],
        score_manager: posnnet.training.score_manager.ScoreManager,
        i_epoch: int,
        len_seq: int,
        n_epochs: int,
        patience: Union[int, None],
        random_seed: int,
        velocity_loss: str,
        batch_size_train: int,
        n_accumulations: int,
        batch_size_val: int,
        learning_rate: float,
        weight_decay: float,
        alpha: float,
    ) -> None:

        """
        Save a checkpoint (at condition that no checkpoint has been saved for n_epochs_training_checkpoint epochs).

        Args:
            - model (posnnet.models.GeneralModel): The model which is trained.
            - optimizer (torch.optim.AdamW): The optimizer used to optimize the model during training.
            - mixed_precision_scaler (torch.amp.GradScaler): The scaler used to scale / unscale gradient during training with mixed precision.
            - best_model_state_dict (Dict[str, torch.Tensor]): The state dict of the model during the best epoch achieved.
            - score_manager (posnnet.training.score_manager.ScoreManager): The score manager that store the training and validation metrics.
            - i_epoch (int): The number of the actual epoch.
            - len_seq (int): The length of the sequence provided in input of the model.
            - n_epochs (int): The number of epochs of training if early stopping does not occur.
            - patience (Union[int, None]): The number of epochs without validation results improvement before early stopping training.
                                           If None, training is never early stopped.
            - random_seed (int): A positive integer that is used to ensure determinism of during training.
            - velocity_loss (str): The type of loss function, either 'mae' or 'mse'.
            - batch_size_train (int): The batch size of the dataloader during training.
            - n_accumulations (int): The number of gradient accumulations before performing a model optimisation step.
            - batch_size_val (int): The batch size of the dataloader during validation.
            - learning_rate (float): The learning rate parameter of the optimizer.
            - weight_decay (float): The weight decay parameter of the optimizer.
            - alpha (float): The weight of the loss for the masked part of the sequence.
        """

        # Save the checkpoint only each n_epochs_training_checkpoint epochs.
        if i_epoch % self.n_epochs_training_checkpoint == self.n_epochs_training_checkpoint - 1:

            # Create a dict that contains every information required to restart the training with reproductibility guarantee.
            training_checkpoint = {
                "i_epoch": i_epoch + 1,
                "state_dicts": {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "mixed_precision_scaler": mixed_precision_scaler.state_dict(),
                    "best_model": best_model_state_dict
                },
                "score_manager": score_manager,
                "training_params": {
                    "len_seq": len_seq,
                    "n_epochs": n_epochs,
                    "patience": patience,
                    "random_seed": random_seed,
                    "velocity_loss": velocity_loss,
                    "batch_size_train": batch_size_train,
                    "n_accumulations": n_accumulations,
                    "batch_size_val": batch_size_val,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "alpha": alpha
                }
            }

            # If a checkpoint already exists, rename it to keep it until the new one is saved.
            if self.path_checkpoint.exists():
                self.path_checkpoint.rename(target=self.path_old_checkpoint)

            # Save the checkpoint
            joblib.dump(value=training_checkpoint, filename=self.path_checkpoint)

            # Now that the new checkpoint is saved, remove the old one.
            self.path_old_checkpoint.unlink(missing_ok=True) # Missing ok = True so even is their was not an older checkpoint, no error is raised.

    def load_or_initialize(
        self: Self,
        model: posnnet.models.GeneralModel,
        optimizer: torch.optim.AdamW,
        mixed_precision_scaler: torch.amp.GradScaler,
        len_seq: int,
        n_epochs: int,
        patience: Union[int, None],
        random_seed: int,
        velocity_loss: str,
        batch_size_train: int,
        n_accumulations: int,
        batch_size_val: int,
        learning_rate: float,
        weight_decay: float,
        alpha: float,
    ) -> Tuple[int, Union[Dict[str, torch.Tensor], None], posnnet.training.score_manager.ScoreManager]:

        """
        If a checkpoint for the configuration exists, load it, otherwise initialize the parameters with default values.

        Args:
            - model (posnnet.models.GeneralModel): The model which is trained.
            - optimizer (torch.optim.AdamW): The optimizer used to optimize the model during training.
            - mixed_precision_scaler (torch.amp.GradScaler): The scaler used to scale / unscale gradient during training with mixed precision.
            - len_seq (int): The length of the sequence provided in input of the model.
            - n_epochs (int): The number of epochs of training if early stopping does not occur.
            - patience (Union[int, None]): The number of epochs without validation results improvement before early stopping training.
                                           If None, training is never early stopped.
            - random_seed (int): A positive integer that is used to ensure determinism of during training.
            - velocity_loss (str): The type of loss function, either 'mae' or 'mse'.
            - batch_size_train (int): The batch size of the dataloader during training.
            - n_accumulations (int): The number of gradient accumulations before performing a model optimisation step.
            - batch_size_val (int): The batch size of the dataloader during validation.
            - learning_rate (float): The learning rate parameter of the optimizer.
            - weight_decay (float): The weight decay parameter of the optimizer.
            - alpha (float): The weight of the loss for the masked part of the sequence.

        Returns:
            - i_start_epoch (int): The number of the epoch at which start the training. If no checkpoint, default = 0.
            - best_model_state_dict (Union[Dict[str, torch.Tensor], None]): The state dict of the model for the best epoch achieved. If no checkpoint, default = None.
            - score_manager (posnnet.training.score_manager.ScoreManager): The score manager which store training and validation metrics. If no checkpoint, a new one
                                                                           is initialized with no history.
        """

        # No checkpoint exists (Normal case for a new model).
        if not self.path_checkpoint.exists() and not self.path_old_checkpoint.exists():

            training_checkpoint = None

        # A checkpoint exists (Normal case for a training restart that has already saved checkpoints).
        elif self.path_checkpoint.exists() and not self.path_old_checkpoint.exists():

            # Try to load the checkpoint.
            try:
                training_checkpoint = joblib.load(filename=self.path_checkpoint)
            # In case of error (for example corrupted file), remove the checkpoint file from the disk and restart training from beginning.
            except:
                training_checkpoint = None
                self.path_checkpoint.unlink()

        # Only an old checkpoint exists (The training process stopped during checkpoint saving, before the start of the saving on disk of the new checkpoint).
        elif not self.path_checkpoint.exists() and self.path_old_checkpoint.exists():

            # Try to load the old checkpoint and rename it as the last one.
            try:
                training_checkpoint = joblib.load(filename=self.path_old_checkpoint)
                self.path_old_checkpoint.rename(target=self.path_checkpoint)
            # In case of error (old checkpoint file corrupted), remove the old checkpoint file from disk and restart training from beginning.
            except:
                training_checkpoint = None
                self.path_old_checkpoint.unlink()

        # Both checkpoint and old checkpoints exist (The training process stopped during new checkpoint saving on disk or during old checkpoint removing).
        elif self.path_checkpoint.exists() and self.path_old_checkpoint.exists():

            # Try to load the last checkpoint and then remove the older
            try:
                training_checkpoint = joblib.load(filename=self.path_checkpoint)
                self.path_old_checkpoint.unlink()
            # In case of error (last checkpoint file corrupted), remove the last checkpoint file from disk.
            except:
                self.path_checkpoint.unlink()
                # Try to load the older checkpoint and rename it as the last one.
                try:
                    training_checkpoint = joblib.load(filename=self.path_old_checkpoint)
                    self.path_old_checkpoint.rename(target=self.path_checkpoint)
                except:
                    # In case of error (old checkpoint file corrupted), remove thd old checkpoint file from disk and restart the training from beginning.
                    training_checkpoint = None
                    self.path_old_checkpoint.unlink()

        # Should not occur.
        else:
            pass

        # If a checkpoint has been loaded, check if the training params present in the checkpoint (the training params used for training the model 
        # before interruption) are identical as the actual training params. If not, checkpoint will not be restored and training will be restart from beginning.
        if training_checkpoint is not None:
            
            try:
                
                assert training_checkpoint.get("training_params").get("len_seq") == len_seq
                assert training_checkpoint.get("training_params").get("n_epochs") == n_epochs
                assert training_checkpoint.get("training_params").get("patience") == patience
                assert training_checkpoint.get("training_params").get("random_seed") == random_seed
                assert training_checkpoint.get("training_params").get("velocity_loss") == velocity_loss
                assert training_checkpoint.get("training_params").get("batch_size_train") == batch_size_train
                assert training_checkpoint.get("training_params").get("n_accumulations") == n_accumulations
                assert training_checkpoint.get("training_params").get("batch_size_val") == batch_size_val
                assert training_checkpoint.get("training_params").get("learning_rate") == learning_rate
                assert training_checkpoint.get("training_params").get("weight_decay") == weight_decay
                assert training_checkpoint.get("training_params").get("alpha") == alpha
                
            except:
                
                training_checkpoint = None

        # If a checkpoint has been loaded and training params are identical, restore state
        if training_checkpoint is not None:

            # Try to restore state
            try:
            
                i_start_epoch = training_checkpoint.get("i_epoch")
                model.load_state_dict(training_checkpoint.get("state_dicts").get("model"))
                optimizer.load_state_dict(training_checkpoint.get("state_dicts").get("optimizer"))
                mixed_precision_scaler.load_state_dict(training_checkpoint.get("state_dicts").get("mixed_precision_scaler"))
                best_model_state_dict = training_checkpoint.get("state_dicts").get("best_model")
                score_manager = training_checkpoint.get("score_manager")

            # In case of error (for example if the model architecture is not the same), restart from beginning
            except:

                training_checkpoint = None
            
        # If no checkpoint (or no training params identical or impossible to restore the state), start from beginning.
        if training_checkpoint is None:
            
            i_start_epoch = 0
            best_model_state_dict = None
            score_manager = posnnet.training.score_manager.ScoreManager(patience=patience)

        return i_start_epoch, best_model_state_dict, score_manager

    def delete_checkpoint_file(
        self: Self
    ) -> None:

        """
        Remove the checkpoint file associated to the configuration if it exists.
        """

        self.path_checkpoint.unlink(missing_ok=True)