import contextlib
from copy import deepcopy
import gc
import numpy as np
import pathlib
import random
import torch
from typing import Self, Tuple, Dict, Union

import posnnet.data.dataset
import posnnet.models

from posnnet.training.batch_size_configurator import BatchSizeConfigurator
from posnnet.training.masks_manager import MasksManager
from posnnet.training.score_manager import ScoreManager
from posnnet.training.training_checkpoint import TrainingCheckpoint
from posnnet.training.trajectory_error_computer import TrajectoryErrorComputer


class Trainer:

    def __init__(
        self: Self,
        id_config: str,
        scalers: posnnet.data.scalers.Scalers,
        relax_points: Dict[str, Dict[str, np.ndarray]],
        use_adversarial: Union[str, None],
        training_type: str,
        velocity_loss: str,
        len_seq: int,
        min_len_gps_outage: int,
        max_len_gps_outage: int,
        coeff_frequency_division: int,
        frequency: int,
        n_epochs: int,
        patience: Union[int, None],
        n_epochs_training_checkpoint: int,
        random_seed: int,
        num_workers: int,
        use_mixed_precision: bool,
        device: torch.device,
        dtype: torch.dtype,
        verbosity: bool,
        path_temp: pathlib.Path
    ) -> None:

        """
        Initialize a Trainer instance that manage the training and scoring of models.

        Args:
            - id_config (str): A unique id which represents the configuration.
            - scalers (posnnet.data.scalers.Scalers): The project Scalers instance.
            - relax_points (Dict[str, Dict[str, np.ndarray]]): The relax points (scaled) of the dataset (velocity GPS = 0 / velocity fusion = 0 / orientation fusion = 0).
            - use_adversarial (Union[str, None]): Either to use adversarial example during training on IMU data ('imu'), on all the input ('full') or not (None).
            - training_type (str): The type of training (GPS outage placement). Can be either 'beginning', 'centered', 'end' or 'random'.
            - velocity_loss (str): The type of loss function used to optimize and score the model. Can be either 'mae' or 'mse'.
            - len_seq (int): The length of the sequence provided to the model after frequency division.
            - min_len_gps_outage (int): The minimal length of a GPS outage (after frequency division).
            - max_len_gps_outage (int): The maximal length of a GPS outage (after frequency division).
            - coeff_frequency_division (int): The factor of frequency division.
            - frequency (int): The original frequency of the sensor.
            - n_epochs (int): The number of epochs for which train the models (if early stopping is not triggered).
            - patience (Union[int, None]): The number of epochs without validation results improvement before early stopping training.
                                           If None, training is never early stopped.
            - n_epochs_training_checkpoint (int): The number of epochs of training after which to save a checkpoint.
            - random_seed (int): A positive integer that is used to ensure determinism during GPS outage simulation generation.
            - num_workers (int): The number of process that will instanciate a dataloader instance to load the data in parallel. 0 means that*
                                 the dataloader will be loaded in the main process. A rule of thumb is to use 4 processes by GPU.
            - use_mixed_precision (bool): Either to use or not float16 mixed precision for training.
            - device (torch.Tensor): The torch device used for training and inference.
            - dtype (torch.dtype): The dtype of the network and the data.
            - verbosity (bool): Determine if training and validation metrics are displayed or not in the standard output (used for debug / follow the training).
            - path_temp (pathlib.Path): The path where temporary files are stored.
        """

        # Memorize parameters.
        self.use_adversarial = use_adversarial
        self.len_seq = len_seq
        self.n_epochs = n_epochs
        self.patience = patience
        self.random_seed = random_seed
        self.num_workers = num_workers
        self.use_mixed_precision = use_mixed_precision
        self.device = device
        self.dtype = dtype
        self.verbosity = verbosity
        self.velocity_loss = velocity_loss
        self.coeff_frequency_division = coeff_frequency_division
        self.training_type = training_type

        # Initialize the criterion for loss computation.
        if velocity_loss == "mae":
            self.criterion = torch.nn.L1Loss(reduction="sum")
        elif velocity_loss == "mse":
            self.criterion = torch.nn.MSELoss(reduction="sum")
        else:
            raise NotImplementedError(f"No loss implemented for {velocity_loss}.")

        # Initialize the masks manager.
        self.masks_manager = MasksManager(
            training_type=training_type,
            len_seq=len_seq,
            min_len_gps_outage=min_len_gps_outage,
            max_len_gps_outage=max_len_gps_outage,
            relax_points=relax_points
        )

        # Initialize the training checkpoint manager.
        self.training_checkpoint = TrainingCheckpoint(
            id_config=id_config,
            n_epochs_training_checkpoint=n_epochs_training_checkpoint,
            path_temp=path_temp
        )

        # Initialize the trajectory error computer.
        self.trajectory_error_computer = TrajectoryErrorComputer(
            scalers=scalers,
            coeff_frequency_division=coeff_frequency_division,
            frequency=frequency
        )

        # Initialize autocast context (for mixed precision).
        if use_mixed_precision:
            self.autocast_context = torch.autocast(device_type=device.type, dtype=torch.float16)
        else:
            self.autocast_context = contextlib.nullcontext()

    def train_and_score(
        self: Self,
        model: posnnet.models.GeneralModel,
        dataset_train: posnnet.data.dataset.Dataset,
        dataset_val: posnnet.data.dataset.Dataset,
        batch_size: int,
        learning_rate: float,
        weight_decay: float,
        alpha: float,
        beta: Union[float, None],
        epsilon: Union[float, None],
        scaling_type: str
    ) -> Tuple[float, float, Dict[str, torch.Tensor], int, int, Union[Dict[str, Dict[str, float]], None]]:

        """
        Perform the training and the scoring of a model.

        Args:
            - model (posnnet.models.GeneralModel): The model for which perform the validation epoch.
            - dataset_train (posnnet.data.dataset.Dataset): The dataset that contains training data.
            - dataset_val (posnnet.data.dataset.Dataset): The dataset that contains validation data.
            - batch_size (int): The asked batch size for training (either in one shoot or with gradient accumulations if GPU memory isn't able to manage the full batch).
            - learning_rate (float): The learning rate that will be used by the AdamW optimizer.
            - weight_decay (float): The weight decay that will be used by the AdamW optimizer.
            - alpha (float): The weight of the loss for the masked part of the sequence.
            - beta (Union[float, None]): The weight of the adversarial loss in the final loss during training (Ignored when adversarial training is not used, set to None).
            - epsilon (Union[float, None]): The weight of the gradient used to compute adversarial example during training (Ignored when adversarial training is not used, set to None).
            - scaling_type (int): Either 'normalization' for a min-max scaling between -1 and 1 or 'standardization' for a standard scaling with zero mean and unit std.

        Returns:
            - best_loss_achieved (float): The best (lowest) velocity loss achieved during validation.
            - associated_ate (float): The average trajectory error of the epoch when the model achieved the lowest velocity loss
            - best_model_state_dict (Dict[str, torch.Tensor]): The state dict of the model when it reaches the best score.
            - batch_size_train (int): The effective batch size used during training.
            - n_accumulations (int): The number of gradient accumulations before optimization step (n_accumulations = batch_size // batch_size_train).
            - scores_on_fixed_training_case (Union[Dict[str, Dict[str, float]], None]): A dictionnary with the structure {training_type: {'velocity_loss': x, 'ate': y}}. None if the training type is not 'random'.
        """

        # Create optimizer and scaler for mixed precision.
        optimizer = torch.optim.AdamW(
            params=model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        mixed_precision_scaler = torch.amp.GradScaler(self.device.type)

        # Transfer the model to the device (GPU or CPU) and cast the dtype.
        model = model.to(device=self.device, dtype=self.dtype)

        # Search a batch size configuration for training and validation.
        batch_size_train, n_accumulations, batch_size_val = BatchSizeConfigurator.find_train_val_batch_size(
            model=model,
            len_seq=self.len_seq,
            batch_size=batch_size,
            n_axes_by_data_sources=dataset_train.get_n_axes_by_data_sources(),
            device=self.device,
            use_mixed_precision=self.use_mixed_precision
        )

        # If a training checkpoint exists, restore the training state. Otherwise, initialize to the training to the beginnning.
        i_start_epoch, best_model_state_dict, score_manager = self.training_checkpoint.load_or_initialize(
            model=model,
            optimizer=optimizer,
            mixed_precision_scaler=mixed_precision_scaler,
            len_seq=self.len_seq,
            n_epochs=self.n_epochs,
            patience=self.patience,
            random_seed=self.random_seed,
            velocity_loss=self.velocity_loss,
            batch_size_train=batch_size_train,
            n_accumulations=n_accumulations,
            batch_size_val=batch_size_val,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            alpha=alpha
        )

        # Set random seed.
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

        # Create dataloaders.
        dataloader_train = torch.utils.data.DataLoader(
            dataset=dataset_train,
            batch_size=batch_size_train,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
            drop_last=True
        )
        dataloader_val = torch.utils.data.DataLoader(
            dataset=dataset_val,
            batch_size=batch_size_val,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            drop_last=False
        )   

        for i_epoch in range(i_start_epoch, self.n_epochs):

            # Clean GPU memory.
            if self.device.type == "cuda":
                gc.collect()
                torch.cuda.empty_cache()

            # Perform a training epoch.
            loss_velocity_masked_train, loss_velocity_unmasked_train, average_trajectory_error_train = self.__training_epoch(
                model=model,
                optimizer=optimizer,
                mixed_precision_scaler=mixed_precision_scaler,
                dataloader_train=dataloader_train,
                i_epoch=i_epoch,
                alpha=alpha,
                beta=beta,
                epsilon=epsilon,
                n_accumulations=n_accumulations,
                scaling_type=scaling_type
            )
            score_manager.push_training_metrics(
                loss_velocity_masked_train=loss_velocity_masked_train,
                loss_velocity_unmasked_train=loss_velocity_unmasked_train,
                average_trajectory_error_train=average_trajectory_error_train
            )

            # Perform a validation epoch.
            loss_velocity_masked_val, loss_velocity_unmasked_val, average_trajectory_error_val = self.__validation_epoch(
                model=model,
                dataloader_val=dataloader_val,
                scaling_type=scaling_type
            )
            score_manager.push_validation_metrics(
                loss_velocity_masked_val=loss_velocity_masked_val,
                loss_velocity_unmasked_val=loss_velocity_unmasked_val,
                average_trajectory_error_val=average_trajectory_error_val
            )

            # Print verbose.
            if self.verbosity:
                score_manager.print_verbose()

            # Evaluate early stopping and stop training if their is not any improvement during the last x 
            # epochs, where x is the patience.
            if score_manager.evaluate_early_stopping_criterion():
                break

            # Evaluate if the model has improved.
            if score_manager.has_model_improved():
                best_model_state_dict = deepcopy(model.state_dict())

            # Save a training checkpoint file (only one epoch each n_epochs_training_checkpoint).
            self.training_checkpoint.save(
                model=model,
                optimizer=optimizer,
                mixed_precision_scaler=mixed_precision_scaler,
                best_model_state_dict=best_model_state_dict,
                score_manager=score_manager,
                i_epoch=i_epoch,
                len_seq=self.len_seq,
                n_epochs=self.n_epochs,
                patience=self.patience,
                random_seed=self.random_seed,
                velocity_loss=self.velocity_loss,
                batch_size_train=batch_size_train,
                n_accumulations=n_accumulations,
                batch_size_val=batch_size_val,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                alpha=alpha
            )  

        # If the training type is 'random', perform a validation epoch with 'beginning', 'centered' and 'end' to have comparable values.
        if self.training_type == "random":

            scores_on_fixed_training_case = self.__score_on_fixed_case(
                model=model,
                dataloader_val=dataloader_val,
                scaling_type=scaling_type,
                best_model_state_dict=best_model_state_dict
            ) 

        else:

            scores_on_fixed_training_case = None

        # Remove training checkpoint file.
        self.training_checkpoint.delete_checkpoint_file()

        best_loss_achieved, associated_ate = score_manager.get_best_score()

        return best_loss_achieved, associated_ate, best_model_state_dict, batch_size_train, n_accumulations, scores_on_fixed_training_case

    def __score_on_fixed_case(
        self: Self,
        model: posnnet.models.GeneralModel,
        dataloader_val: torch.utils.data.DataLoader,
        scaling_type: str,
        best_model_state_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, Dict[str, float]]:

        """
        Score the model on the validation dataset for the 3 fixed cases. Usefull to have a comparison for 'random' training type with others fixed training types.

        Args:
            - model (posnnet.models.GeneralModel): The model for which perform the validation epoch.
            - dataloader_val (torch.utils.data.DataLoader): The pytorch dataloader that is used to get the validation data.
            - scaling_type (int): Either 'normalization' for a min-max scaling between -1 and 1 or 'standardization' for a standard scaling with zero mean and unit std.
            - best_model_state_dict (Dict[str, torch.Tensor]): The state dict of the model when it reaches the best score.

        Returns:
            - scores_on_fixed_training_case (Dict[str, Dict[str, float]]): A dictionnary with the structure {training_type: {'velocity_loss': x, 'ate': y}}.
        """

        scores_on_fixed_training_case = {}

        # Restore best model state dict.
        model.load_state_dict(state_dict=best_model_state_dict)

        # Iterate over fixed training type.
        for training_type in ["beginning", "centered", "end"]:

            self.masks_manager.training_type = training_type

            loss_velocity_masked_val, _, average_trajectory_error_val = self.__validation_epoch(
                model=model,
                dataloader_val=dataloader_val,
                scaling_type=scaling_type
            )
            scores_on_fixed_training_case[training_type] = {
                "velocity_loss": loss_velocity_masked_val,
                "ate": average_trajectory_error_val
            }

        return scores_on_fixed_training_case
            
    def __training_epoch(
        self: Self,
        model: posnnet.models.GeneralModel,
        optimizer: torch.optim.AdamW,
        mixed_precision_scaler: torch.amp.GradScaler,
        dataloader_train: torch.utils.data.DataLoader,
        i_epoch: int,
        alpha: float,
        beta: Union[float, None],
        epsilon: Union[float, None],
        n_accumulations: int,
        scaling_type: str
    ) -> Tuple[float, float, float]:

        """
        Perform one training epoch.

        Args:
            - model (posnnet.models.GeneralModel): The model for which perform the validation epoch.
            - optimizer (torch.optim.AdamW): The AdamW optimizer used to optimize the network during training.
            - mixed_precision_scaler (torch.amp.GradScaler): The scaler used for mixed precision during training (not the same than the scalers for the dataset).
            - dataloader_train (torch.utils.data.DataLoader): The pytorch dataloader that is used to get the training data.
            - i_epoch (int): The number of the actual epoch.
            - alpha (float): The weight of the loss for the masked part of the sequence.
            - beta (Union[float, None]): The weight of the adversarial loss in the final loss during training (Ignored when adversarial training is not used, set to None).
            - epsilon (Union[float, None]): The weight of the gradient used to compute adversarial example during training (Ignored when adversarial training is not used, set to None).
            - n_accumulations (int): The number of gradient accumulations before performing an optimization step.
            - scaling_type (int): Either 'normalization' for a min-max scaling between -1 and 1 or 'standardization' for a standard scaling with zero mean and unit std.

        Returns:
            - loss_velocity_masked_train (float): Velocity loss on the masked part of the sequence for the training dataset (mean over every masked points).
            - loss_velocity_unmasked_train (float): Velocity loss on the unmasked part of the sequence for the training dataset (mean over every unmasked points).
            - average_trajectory_error_train (float): Average trajectory error (ATE) for the training dataset (mean over every masked timestamp)
        """

        # Set random seed to have different masks from one epoch to another.
        torch.manual_seed(self.random_seed + i_epoch)
        np.random.seed(self.random_seed + i_epoch)
        random.seed(self.random_seed + i_epoch)

        # Set the dataset sampler.
        dataloader_train.dataset.set_epoch(i_epoch=i_epoch)

        # Set the model in training mode.
        model = model.train()

        # Initialize metrics.
        loss_velocity_masked_train = 0
        loss_velocity_unmasked_train = 0
        trajectory_error_train = 0
        size_masked_train = 0
        size_unmasked_train = 0
        n_timestamps_masked_train = 0

        # Reset optimizer gradient (set to none for performance purpose).
        optimizer.zero_grad(set_to_none=True)

        # Initialize gradient accumulation counter
        gradient_accumultations_counter = 0

        for i_batch, (x_accelerometer, x_gyroscope, x_magnetometer, x_velocity_gps, x_velocity_fusion, x_orientation_fusion, y_velocity_fusion, y_velocity_fusion_unscaled_undivided) in enumerate(dataloader_train):

            # Generate random masks.
            masks = self.masks_manager.generate_masks(batch_size=x_velocity_gps.size(dim=0))

            # Generate a tensor to indicate the masks to the model.
            x_mask = masks.clone().detach()

            # Mask GPS and fusion data during the GPS outage simulation (where masks == True).
            x_velocity_gps, x_velocity_fusion, x_orientation_fusion = self.masks_manager.apply_masks(
                masks=masks,
                x_velocity_gps=x_velocity_gps,
                x_velocity_fusion=x_velocity_fusion,
                x_orientation_fusion=x_orientation_fusion,
                scaling_type=scaling_type
            )

            # Transfer the data to the device.
            x_accelerometer = x_accelerometer.to(device=self.device)
            x_gyroscope = x_gyroscope.to(device=self.device)
            x_magnetometer = x_magnetometer.to(device=self.device)
            x_velocity_gps = x_velocity_gps.to(device=self.device)
            x_velocity_fusion = x_velocity_fusion.to(device=self.device)
            x_orientation_fusion = x_orientation_fusion.to(device=self.device)
            x_mask = x_mask.to(device=self.device)
            y_velocity_fusion = y_velocity_fusion.to(device=self.device)

            # Extend masks to match the tarket dimension (for loss computation) and transfer to the device.
            masks = masks.repeat(repeats=(1, 1, y_velocity_fusion.size(dim=-1)))
            size_masked = masks.sum().item()
            size_unmasked = (~ masks).sum().item()
            masks = masks.to(device=self.device)

            # Set the data tensors as requiring a gradient to be able to compute adversarial example.
            if self.use_adversarial is None:
                xs_requiring_grad = [] # No gradient required because training do not use adversarial example.
            elif self.use_adversarial == "imu":
                xs_requiring_grad = [x_accelerometer, x_gyroscope, x_magnetometer] # IMU data.           
            elif self.use_adversarial == "full":
                xs_requiring_grad = [x_accelerometer, x_gyroscope, x_magnetometer, x_velocity_gps, x_velocity_fusion, x_orientation_fusion] # IMU + GPS + fusion data.             
            else:
                pass # Should not occur.

            for x_requiring_grad in xs_requiring_grad:
                x_requiring_grad.requires_grad = True

            # Use mixed precision with 16 bits to speed up training and reduce memory requirements.
            with self.autocast_context:

                # Forward pass
                y_velocity_pred = model(
                    x_accelerometer=x_accelerometer,
                    x_gyroscope=x_gyroscope,
                    x_magnetometer=x_magnetometer,
                    x_velocity_gps=x_velocity_gps,
                    x_velocity_fusion=x_velocity_fusion,
                    x_orientation_fusion=x_orientation_fusion,
                    x_mask=x_mask
                )

                # Loss computation.
                # Instead of using the mean reduction of pytorch loss that will average taking the number of sample,
                # the loss is averaged using the number of masked and unmasked samples in the input.
                loss_velocity_masked = self.criterion(
                    y_velocity_pred.masked_fill(mask= ~ masks, value=0),
                    y_velocity_fusion.masked_fill(mask= ~ masks, value=0)
                ) / size_masked
                loss_velocity_unmasked = self.criterion(
                    y_velocity_pred.masked_fill(mask=masks, value=0),
                    y_velocity_fusion.masked_fill(mask=masks, value=0)
                ) / size_unmasked

                # Velocity loss is the weighted sum of the velocity loss on masked part and on unmasked part.
                loss_velocity = alpha * loss_velocity_masked + (1 - alpha) * loss_velocity_unmasked

            # If the loss is undefined (nan), skip this batch
            if torch.isnan(loss_velocity):
                continue

            # If adversarial examples are not used, the final loss is only the loss on original examples.
            if self.use_adversarial is None:

                # Backward on the velocity loss (after scaling for mixed precision).
                final_loss = loss_velocity
                
                if self.use_mixed_precision:
                    mixed_precision_scaler.scale(final_loss).backward()
                else:
                    final_loss.backward()

            # If adversarial examples are used.
            else:

                # Backward on the velocity loss (after scaling for mixed precision). Retain the graph to compute adversarial examples.
                if self.use_mixed_precision:
                    mixed_precision_scaler.scale(loss_velocity).backward(retain_graph=True)
                else:
                    loss_velocity.backward(retain_graph=True)

                # Compute adversarial example.
                x_accelerometer_adv = (x_accelerometer + epsilon * x_accelerometer.grad.sign()).detach()
                x_gyroscope_adv = (x_gyroscope + epsilon * x_gyroscope.grad.sign()).detach()
                x_magnetometer_adv = (x_magnetometer + epsilon * x_magnetometer.grad.sign()).detach()
                x_mask_adv = x_mask.detach()
                
                if self.use_adversarial == "imu":
                    x_velocity_gps_adv = x_velocity_gps.detach()
                    x_velocity_fusion_adv = x_velocity_fusion.detach()
                    x_orientation_fusion_adv = x_orientation_fusion.detach()
                elif self.use_adversarial == "full":
                    x_velocity_gps_adv = (x_velocity_gps + epsilon * x_velocity_gps.grad.sign()).detach()
                    x_velocity_fusion_adv = (x_velocity_fusion + epsilon * x_velocity_fusion.grad.sign()).detach()
                    x_orientation_fusion_adv = (x_orientation_fusion + epsilon * x_orientation_fusion.grad.sign()).detach()

                 #Use mixed precision with 16 bits to speed up training and reduce memory requirements.
                with self.autocast_context:
    
                    # Forward pass on adversarial example.
                    y_velocity_pred_adv = model(
                        x_accelerometer=x_accelerometer_adv,
                        x_gyroscope=x_gyroscope_adv,
                        x_magnetometer=x_magnetometer_adv,
                        x_velocity_gps=x_velocity_gps_adv,
                        x_velocity_fusion=x_velocity_fusion_adv,
                        x_orientation_fusion=x_orientation_fusion_adv,
                        x_mask=x_mask
                    )
    
                    # loss computation on adversarial example.
                    # Instead of using the mean reduction of pytorch loss that will average taking the number of sample,
                    # the loss is averaged using the number of masked and unmasked samples in the input.
                    loss_velocity_masked_adv = self.criterion(
                        y_velocity_pred_adv.masked_fill(mask= ~ masks, value=0),
                        y_velocity_fusion.masked_fill(mask= ~ masks, value=0)
                    ) / size_masked
                    loss_velocity_unmasked_adv = self.criterion(
                        y_velocity_pred_adv.masked_fill(mask=masks, value=0),
                        y_velocity_fusion.masked_fill(mask=masks, value=0)
                    ) / size_unmasked
    
                    # Velocity loss on adversarial example is the weighted sum of the velocity loss on masked part and on unmasked part.
                    loss_velocity_adv = alpha * loss_velocity_masked_adv + (1 - alpha) * loss_velocity_unmasked_adv

                # If the adversarial loss is undefined (nan), skip this batch
                if torch.isnan(loss_velocity_adv):
                    continue

                # Final loss is the velocity loss on the original examples + a ponderation of the velocity loss on adversarial examples.
                final_loss = loss_velocity + beta * loss_velocity_adv

                # Backward the final loss (after scaling for mixed precision).
                if self.use_mixed_precision:
                    mixed_precision_scaler.scale(final_loss).backward()
                else:
                    final_loss.backward()

            gradient_accumultations_counter += 1

            # If their are n_accumulations of gradients, perform a step of optimization.
            if gradient_accumultations_counter == n_accumulations:

                if self.use_mixed_precision:
                    mixed_precision_scaler.step(optimizer)
                    mixed_precision_scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                gradient_accumultations_counter = 0

            # Add batch loss.
            loss_velocity_masked_train += loss_velocity_masked.item() * size_masked
            loss_velocity_unmasked_train += loss_velocity_unmasked.item() * size_unmasked
            size_masked_train += size_masked
            size_unmasked_train += size_unmasked
            n_timestamps_masked_train += self.coeff_frequency_division * size_masked / y_velocity_fusion_unscaled_undivided.size(dim=-1) # One timestamp = N axis, multiply by self.coeff_frequency_division because ATE is computed on original frequency so there is self.coeff_frequency_division more timestamp.

            # Compute trajectory error.
            trajectory_error_train += self.trajectory_error_computer.compute(
                y_velocity_fusion_unscaled_undivided=y_velocity_fusion_unscaled_undivided,
                y_velocity_pred=y_velocity_pred,
                masks=masks
            )

        # Divide the sum of batches loss by the total size of the masks.
        loss_velocity_masked_train = loss_velocity_masked_train / size_masked_train
        loss_velocity_unmasked_train = loss_velocity_unmasked_train / size_unmasked_train
        average_trajectory_error_train = trajectory_error_train / n_timestamps_masked_train

        return loss_velocity_masked_train, loss_velocity_unmasked_train, average_trajectory_error_train

    def __validation_epoch(
        self: Self,
        model: posnnet.models.GeneralModel,
        dataloader_val: torch.utils.data.DataLoader,
        scaling_type: str
    ) -> Tuple[float, float, float]:

        """
        Perform one validation epoch.

        Args:
            - model (posnnet.models.GeneralModel): The model for which perform the validation epoch.
            - dataloader_val (torch.utils.data.DataLoader): The pytorch dataloader that is used to get the validation data.
            - scaling_type (int): Either 'normalization' for a min-max scaling between -1 and 1 or 'standardization' for a standard scaling with zero mean and unit std.

        Returns:
            - loss_velocity_masked_val (float): Velocity loss on the masked part of the sequence for the validation dataset (mean over every masked points).
            - loss_velocity_unmasked_val (float): Velocity loss on the unmasked part of the sequence for the validation dataset (mean over every unmasked points).
            - average_trajectory_error_val (float): Average trajectory error (ATE) for the validation dataset (mean over every masked timestamp)
        """

        # Set random seed to have same masks from one epoch to another.
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

        # Set the model in validation mode.
        model = model.eval()

        # Initialize metrics.
        loss_velocity_masked_val = 0
        loss_velocity_unmasked_val = 0
        trajectory_error_val = 0
        size_masked_val = 0
        size_unmasked_val = 0
        n_timestamps_masked_val = 0

        for i_batch, (x_accelerometer, x_gyroscope, x_magnetometer, x_velocity_gps, x_velocity_fusion, x_orientation_fusion, y_velocity_fusion, y_velocity_fusion_unscaled_undivided) in enumerate(dataloader_val):

            # Generate random masks
            masks = self.masks_manager.generate_masks(batch_size=x_velocity_gps.size(dim=0))

            # Generate a tensor to indicate the masks to the model.
            x_mask = masks.clone().detach()

            # Mask GPS and fusion data during the GPS outage simulation (where masks == True).
            x_velocity_gps, x_velocity_fusion, x_orientation_fusion = self.masks_manager.apply_masks(
                masks=masks,
                x_velocity_gps=x_velocity_gps,
                x_velocity_fusion=x_velocity_fusion,
                x_orientation_fusion=x_orientation_fusion,
                scaling_type=scaling_type
            )

            # Transfer the data to the device.
            x_accelerometer = x_accelerometer.to(device=self.device)
            x_gyroscope = x_gyroscope.to(device=self.device)
            x_magnetometer = x_magnetometer.to(device=self.device)
            x_velocity_gps = x_velocity_gps.to(device=self.device)
            x_velocity_fusion = x_velocity_fusion.to(device=self.device)
            x_orientation_fusion = x_orientation_fusion.to(device=self.device)
            x_mask = x_mask.to(device=self.device)
            y_velocity_fusion = y_velocity_fusion.to(device=self.device)

            # Extend masks to match the tarket dimension (for loss computation) and transfer to the device.
            masks = masks.repeat(repeats=(1, 1, y_velocity_fusion.size(dim=-1)))
            size_masked = masks.sum().item()
            size_unmasked = (~ masks).sum().item()
            masks = masks.to(device=self.device)

            with torch.no_grad():

                # Forward pass
                y_velocity_pred = model(
                    x_accelerometer=x_accelerometer,
                    x_gyroscope=x_gyroscope,
                    x_magnetometer=x_magnetometer,
                    x_velocity_gps=x_velocity_gps,
                    x_velocity_fusion=x_velocity_fusion,
                    x_orientation_fusion=x_orientation_fusion,
                    x_mask=x_mask
                )

            # loss computation.
            # Instead of using the mean reduction of pytorch loss that will average taking the number of sample,
            # the loss is averaged using the number of masked and unmasked samples in the input.
            loss_velocity_masked = self.criterion(
                y_velocity_pred.masked_fill(mask= ~ masks, value=0),
                y_velocity_fusion.masked_fill(mask= ~ masks, value=0)
            ) / size_masked
            loss_velocity_unmasked = self.criterion(
                y_velocity_pred.masked_fill(mask=masks, value=0),
                y_velocity_fusion.masked_fill(mask=masks, value=0)
            ) / size_unmasked

            # Add batch loss.
            loss_velocity_masked_val += loss_velocity_masked.item() * size_masked
            loss_velocity_unmasked_val += loss_velocity_unmasked.item() * size_unmasked
            size_masked_val += size_masked
            size_unmasked_val += size_unmasked
            n_timestamps_masked_val += self.coeff_frequency_division * size_masked / y_velocity_fusion_unscaled_undivided.size(dim=-1) # One timestamp = N axis, multiply by self.coeff_frequency_division because ATE is computed on original frequency so there is self.coeff_frequency_division more timestamp.

            # Compute trajectory error.
            trajectory_error_val += self.trajectory_error_computer.compute(
                y_velocity_fusion_unscaled_undivided=y_velocity_fusion_unscaled_undivided,
                y_velocity_pred=y_velocity_pred,
                masks=masks
            )

        # Divide the sum of batches loss by the total size of the masks.
        loss_velocity_masked_val = loss_velocity_masked_val / size_masked_val
        loss_velocity_unmasked_val = loss_velocity_unmasked_val / size_unmasked_val
        average_trajectory_error_val = trajectory_error_val / n_timestamps_masked_val

        return loss_velocity_masked_val, loss_velocity_unmasked_val, average_trajectory_error_val