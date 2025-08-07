import contextlib
from copy import deepcopy
import gc
import torch
from typing import TypeVar, Type, Dict, Tuple

import posnnet.models


class BatchSizeConfigurator:

    @classmethod
    def __is_training_configuration_fitting(
        cls: Type[TypeVar("BatchSizeConfigurator")],
        model: posnnet.models.GeneralModel,
        len_seq: int,
        batch_size_train: int,
        n_axes_by_data_sources: Dict[str, int],
        device: torch.device,
        use_mixed_precision: bool
    ) -> bool:

        """
        Try to proceed to a training pass with the asked batch size to determine if the configuration
        fit in memory with this batch size.

        Args:
            - model (posnnet.models.GeneralModel): The model for which evaluate the configuration.
            - len_seq (int): The size of the input sequence.
            - batch_size_train (int): The size of the batch for the training pass.
            - n_axes_by_data_sources (Dict[str, int]): A dictionnary that contains the keys the data sources ('accelerometer', 
                                                       'gyroscope', 'magnetometer', 'velocity_gps', 'velocity_fusion', 
                                                       'orientation_fusion'). For every key, the value of the dictionnary 
                                                       should be the number of axes of the sensor for the sources associated.
            - device (torch.device): The device on which perform the training pass.
            - use_mixed_precision (bool): Either to use or not float16 mixed precision for training.

        Returns:
            - fit (bool): True if the configuration fit in memory for a training pass, False otherwise.
        """

        # Initialize autocast context (for mixed precision).
        if use_mixed_precision:
            autocast_context = torch.autocast(device_type=device.type, dtype=torch.float16)
        else:
            autocast_context = contextlib.nullcontext()

        # Clear GPU memory
        if device.type == "cuda":
            gc.collect()
            torch.cuda.empty_cache()

        try:

            # Initialize optimizer, scaler and criterion.
            optimizer = torch.optim.AdamW(params=model.parameters())
            scaler = torch.amp.GradScaler(device=device.type)
            criterion = torch.nn.MSELoss()

            # Initialize a batch of data.
            x_accelerometer = torch.rand(size=(batch_size_train, len_seq, n_axes_by_data_sources.get("accelerometer")), device=device)
            x_gyroscope = torch.rand(size=(batch_size_train, len_seq, n_axes_by_data_sources.get("gyroscope")), device=device)
            x_magnetometer = torch.rand(size=(batch_size_train, len_seq, n_axes_by_data_sources.get("magnetometer")), device=device)
            x_velocity_gps = torch.rand(size=(batch_size_train, len_seq, n_axes_by_data_sources.get("velocity_gps")), device=device)
            x_velocity_fusion = torch.rand(size=(batch_size_train, len_seq, n_axes_by_data_sources.get("velocity_fusion")), device=device)
            x_orientation_fusion = torch.rand(size=(batch_size_train, len_seq, n_axes_by_data_sources.get("orientation_fusion")), device=device)
            x_mask = torch.rand(size=(batch_size_train, len_seq, 1), device=device)
            y_velocity = torch.rand(size=(batch_size_train, len_seq, n_axes_by_data_sources.get("velocity_fusion")), device=device)

            # If the device is CUDA and their is less than 20 % of the GPU memory available, raise an OutOfMemoryError.
            if device.type == "cuda":
                free_mem, total_mem = torch.cuda.mem_get_info()
                if free_mem / total_mem < 0.2:
                    raise torch.cuda.OutOfMemoryError()

            # Use mixed precision if asked.
            with autocast_context:

                # Forward pass.
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
                loss = criterion(y_velocity_pred, y_velocity)

            # Backward on the total loss.
            scaler.scale(loss).backward()

            # Optimizer step.
            scaler.step(optimizer)
            optimizer.zero_grad(set_to_none=True)  

            # If the device is CUDA and their is less than 20 % of the GPU memory available, raise an OutOfMemoryError.
            if device.type == "cuda":
                free_mem, total_mem = torch.cuda.mem_get_info()
                if free_mem / total_mem < 0.2:
                    raise torch.cuda.OutOfMemoryError()

            # Success, return True
            fit = True

        # Memory does not satisfy the requirements for this configuration and batch size, return False.
        except Exception:
            
            fit = False

        return fit

    @classmethod
    def __is_validation_configuration_fitting(
        cls: Type[TypeVar("BatchSizeConfigurator")],
        model: posnnet.models.GeneralModel,
        len_seq: int,
        batch_size_val: int,
        n_axes_by_data_sources: Dict[str, int],
        device: torch.device,
        use_mixed_precision: bool
    ) -> bool:

        """
        Try to proceed to a validation pass with the asked batch size to determine if the configuration
        fit in memory with this batch size.

        Args:
            - model (posnnet.models.GeneralModel): The model for which evaluate the configuration.
            - len_seq (int): The size of the input sequence.
            - batch_size_val (int): The size of the batch for the validation pass.
            - n_axes_by_data_sources (Dict[str, int]): A dictionnary that contains the keys the data sources ('accelerometer', 
                                                       'gyroscope', 'magnetometer', 'velocity_gps', 'velocity_fusion', 
                                                       'orientation_fusion'). For every key, the value of the dictionnary 
                                                       should be the number of axes of the sensor for the sources associated.
            - device (torch.device): The device on which perform the validation pass.
            - use_mixed_precision (bool): Either to use or not float16 mixed precision for training.

        Returns:
            - fit (bool): True if the configuration fit in memory for a validation pass, False otherwise.
        """

        # Initialize autocast context (for mixed precision).
        if use_mixed_precision:
            autocast_context = torch.autocast(device_type=device.type, dtype=torch.float16)
        else:
            autocast_context = contextlib.nullcontext()

        # Clear GPU memory
        if device.type == "cuda":
            gc.collect()
            torch.cuda.empty_cache()

        try:

            # Initialize criterion
            criterion = torch.nn.MSELoss()

            # Initialize a batch of data.
            x_accelerometer = torch.rand(size=(batch_size_val, len_seq, n_axes_by_data_sources.get("accelerometer")), device=device)
            x_gyroscope = torch.rand(size=(batch_size_val, len_seq, n_axes_by_data_sources.get("gyroscope")), device=device)
            x_magnetometer = torch.rand(size=(batch_size_val, len_seq, n_axes_by_data_sources.get("magnetometer")), device=device)
            x_velocity_gps = torch.rand(size=(batch_size_val, len_seq, n_axes_by_data_sources.get("velocity_gps")), device=device)
            x_velocity_fusion = torch.rand(size=(batch_size_val, len_seq, n_axes_by_data_sources.get("velocity_fusion")), device=device)
            x_orientation_fusion = torch.rand(size=(batch_size_val, len_seq, n_axes_by_data_sources.get("orientation_fusion")), device=device)
            x_mask = torch.rand(size=(batch_size_val, len_seq, 1), device=device)
            y_velocity = torch.rand(size=(batch_size_val, len_seq, n_axes_by_data_sources.get("velocity_fusion")), device=device)

            # If the device is CUDA and their is less than 20 % of the GPU memory available, raise an OutOfMemoryError.
            if device.type == "cuda":
                free_mem, total_mem = torch.cuda.mem_get_info()
                if free_mem / total_mem < 0.2:
                    raise torch.cuda.OutOfMemoryError()

            # Use mixed precision if asked.
            with autocast_context:

                # Forward pass.
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
                loss = criterion(y_velocity_pred, y_velocity)

            # If the device is CUDA and their is less than 20 % of the GPU memory available, raise an OutOfMemoryError.
            if device.type == "cuda":
                free_mem, total_mem = torch.cuda.mem_get_info()
                if free_mem / total_mem < 0.2:
                    raise torch.cuda.OutOfMemoryError()

            # Success, return True
            fit = True

        # Memory does not satisfy the requirements for this configuration and batch size, return False.
        except Exception:
            
            fit = False

        return fit

    @classmethod
    def find_train_val_batch_size(
        cls: Type[TypeVar("BatchSizeConfigurator")],
        model: posnnet.models.GeneralModel,
        len_seq: int,
        batch_size: int,
        n_axes_by_data_sources: Dict[str, int],
        device: torch.device,
        use_mixed_precision: bool
    ) -> Tuple[int, int, int]:

        """
        Find the batch size for training and validation and the number of accumulation for training.
        E.g. If a batch size of 64 is asked but the system can only handle batch size of 16 for training,
        batch_size_train = 16 and n_accumulations = 4.
        
        Args:
            - model (posnnet.models.GeneralModel): The model for which evaluate the configuration.
            - len_seq (int): The size of the input sequence.
            - batch_size (int): The asked size of the batch for the training.
            - n_axes_by_data_sources (Dict[str, int]): A dictionnary that contains the keys the data sources ('accelerometer', 
                                                       'gyroscope', 'magnetometer', 'velocity_gps', 'velocity_fusion', 
                                                       'orientation_fusion'). For every key, the value of the dictionnary 
                                                       should be the number of axes of the sensor for the sources associated.
            - device (torch.device): The device on which perform the training and validation pass.
            - use_mixed_precision (bool): Either to use or not float16 mixed precision for training.

        Returns:
            - batch_size_train (int): The batch size the system can handle for training.
            - n_accumulations (int): The number of accumulations before performing model optimisation to satisfy the
                                     asked batch size with the actual maximum batch size the system can handle for training.
            - batch_size_val (int): The batch size the system can handle for validation.
        """

        # Keep a backup of the model state dict to restore it at the end of the search.
        model_state_dict = deepcopy(model.state_dict())

        # Set the model in training mode.
        model = model.train()

        batch_size_train = batch_size
        n_accumulations_train = 1

        # While the batch size is superior or equal to 1 and the configuration does not
        # fit with the actual batch size, divide the batch size by 2 and double the 
        # number of accumulations.
        while batch_size_train >= 1 and not cls.__is_training_configuration_fitting(
            model=model,
            len_seq=len_seq,
            batch_size_train=batch_size_train,
            n_axes_by_data_sources=n_axes_by_data_sources,
            device=device,
            use_mixed_precision=use_mixed_precision
        ):

            batch_size_train //= 2
            n_accumulations_train *= 2

        # If the batch size is inferior to 1, the configuration does not fit with any batch size.
        if batch_size_train < 1:
            raise Exception("Impossible to find a batch size for the training configuration.")

        # Set the model to eval mode.
        model = model.eval()

        batch_size_val = 2048

        # While the batch size is superior or equal to 1 and the configuration does not
        # fit with the actual batch size, divide the batch size by 2 and double the 
        # number of accumulations.
        while batch_size_val >= 1 and not cls.__is_validation_configuration_fitting(
            model=model,
            len_seq=len_seq,
            batch_size_val=batch_size_val,
            n_axes_by_data_sources=n_axes_by_data_sources,
            device=device,
            use_mixed_precision=use_mixed_precision
        ):

            batch_size_val //= 2

        # If the batch size is inferior to 1, the configuration does not fit with any batch size.
        if batch_size_val < 1:
            raise Exception("Impossible to find a batch size for the validation configuration.")

        # Restore the model state
        model.load_state_dict(state_dict=model_state_dict)

        return batch_size_train, n_accumulations_train, batch_size_val