import joblib
import numpy as np
import pathlib
import random
import torch

from comparison.source.ctin.dataset import Dataset
from comparison.source.ctin.model import CTIN
from comparison.source.ctin.ctin_loss import CTINLoss


def ctin_training(
    sensor_frequency: float,
    beta: float,
    random_seed: int,
    d_model: int,
    batch_size: int
) -> None:

    PATIENCE = 30
    N_EPOCHS = 300

    # Initialize training and validation dataset.
    dataset_train = Dataset(
        sensor_frequency=sensor_frequency,
        beta=beta,
        path_data=pathlib.Path("./project_framework_validation/data/preprocessed/training/"),
    )
    
    dataset_val = Dataset(
        sensor_frequency=sensor_frequency,
        beta=beta,
        path_data=pathlib.Path("./project_framework_validation/data/preprocessed/validation/"),
    )

    # Set random seed before instantiate model.
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    model = CTIN(input_dim=6, out_dim=3, d_model=d_model).float().cuda()

    # Instantiate dataloaders.
    dataloader_train = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        shuffle=True,
        drop_last=True
    )
    dataloader_val = torch.utils.data.DataLoader(
        dataset=dataset_val,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
        drop_last=False
    )

    # Instantiate optimizer, lr scheduler and criterion for loss.
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-6)

    loss_fc = CTINLoss()

    # If a training checkpoint exists, load it.
    if pathlib.Path(f"./comparison/models_data/ctin_{d_model}_{batch_size}_checkpoint.pkl").exists():
    
        checkpoint = joblib.load(f"./comparison/models_data/ctin_{d_model}_{batch_size}_checkpoint.pkl")
        losses_val = checkpoint["losses_val"]
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        i_start_epoch = checkpoint["i_epoch"] + 1

    # Otherwise, initialize to default values.
    else:
    
        i_start_epoch = 1
        losses_val = []

    # Iterate for N epochs.
    for i_epoch in range(i_start_epoch, N_EPOCHS + 1):

        # Initialize random seed with the epoch number.
        torch.manual_seed(random_seed + i_epoch)
        np.random.seed(random_seed + i_epoch)
        random.seed(random_seed + i_epoch)

        # Initialize epoch loss values.
        loss_train = 0
        loss_val = 0
        n_samples_train = 0
        n_samples_val = 0

        # Set the model in training mode and clean optimizer.
        model.train()
        optimizer.zero_grad(set_to_none=True)

        # Iterate over training dataset.
        for x, y_vel, y_pos in dataloader_train:

            # Transfer data to GPU.
            x, y_vel, y_pos = x.cuda(), y_vel.cuda(), y_pos.cuda()

            # Forward.
            y_vel_pred, y_cov_pred = model(x)

            # Compute loss
            loss = loss_fc(y_vel_pred, y_cov_pred, y_vel, y_pos)
            loss_train += loss.item() * len(x)
            n_samples_train += len(x)
            
            # Backward and optimize model.
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # Compute training epoch loss.
        loss_train /= n_samples_train

        # Set the model in evaluation mode.
        model.eval()

        # Iterate over training dataset.
        for x, y_vel, y_pos in dataloader_val:

            # Transfer data to GPU.
            x, y_vel, y_pos = x.cuda(), y_vel.cuda(), y_pos.cuda()

            # Forward.
            with torch.no_grad():
                y_vel_pred, y_cov_pred = model(x)

            # Compute loss
            loss = loss_fc(y_vel_pred, y_cov_pred, y_vel, y_pos)
            loss_val += loss.item() * len(x)
            n_samples_val += len(x)

        # Compute validation epoch loss.
        loss_val /= n_samples_val

        # If a new best validation loss is reached, save the model state dict.
        if len(losses_val) == 0 or loss_val < np.min(losses_val):
            torch.save(model.state_dict(), f"./comparison/models_data/ctin_{d_model}_{batch_size}_state_dict.pt")

        losses_val.append(loss_val)

        # Save a checkpoint every 5 epochs.
        if i_epoch % 5 == 0:
            checkpoint = {
                "i_epoch": i_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "losses_val": losses_val
            }
            joblib.dump(value=checkpoint, filename=f"./comparison/models_data/ctin_{d_model}_{batch_size}_checkpoint.pkl")

        # Print verbose.
        print(f"Epoch {i_epoch:03d}: loss train = {loss_train:.3f} loss val = {loss_val:.3f} {'NEW BEST' if loss_val == np.min(losses_val) else '':s}")

        if i_epoch > PATIENCE and not np.min(losses_val[-PATIENCE:]) < np.min(losses_val[:-PATIENCE]):
            break