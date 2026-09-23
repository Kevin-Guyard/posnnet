import joblib
import numpy as np
import pathlib
import random
import torch

from comparison.source.ronin.dataset import Dataset
from comparison.source.ronin.model_resnet1d import ResNet1D, BasicBlock1D, FCOutputModule
from comparison.source.ronin.model_temporal import LSTMSeqNetwork, TCNSeqNetwork


def ronin_training(
    model_type: str,
    sensor_frequency: float,
    beta: float,
    random_seed: int
) -> None:

    # Initialize training and validation dataset.
    dataset_train = Dataset(
        model_type=model_type,
        sensor_frequency=sensor_frequency,
        beta=beta,
        path_data=pathlib.Path("./project_framework_validation/data/preprocessed/training/"),
        random_seed=random_seed
    )
    
    dataset_val = Dataset(
        model_type=model_type,
        sensor_frequency=sensor_frequency,
        beta=beta,
        path_data=pathlib.Path("./project_framework_validation/data/preprocessed/validation/"),
        random_seed=random_seed
    )

    # Set random seed before instantiate model.
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    # Instantiate model.
    if model_type == "LSTM":
        model = LSTMSeqNetwork(
            input_size=6,
            out_size=3,
            device=torch.device("cuda"),
            dropout=0.2
        ).float().cuda()
    elif model_type == "TCN":
        model = TCNSeqNetwork(
            input_channel=6,
            output_channel=3,
            kernel_size=3,
            layer_channels=[16, 32, 64, 128, 72, 36],
            dropout=0.2
        ).float().cuda()
    elif model_type == "ResNet1D":
        model = ResNet1D(
            num_inputs=6,
            num_outputs=3,
            block_type=BasicBlock1D,
            group_sizes=[2, 2, 2, 2],
            base_plane=64,
            output_block=FCOutputModule,
            kernel_size=3,
            **{'fc_dim': 512, 'in_dim': 4, 'dropout': 0.5, 'trans_planes': 128}
        ).float().cuda()

    # Collect parameters.
    batch_size = {"LSTM": 72, "TCN": 72, "ResNet1D": 128}.get(model_type)
    learning_rate = {"LSTM": 3e-4, "TCN": 3e-4, "ResNet1D": 1e-4}.get(model_type)
    lr_factor = {"LSTM": 0.75, "TCN": 0.75, "ResNet1D": 0.1}.get(model_type)
    n_epochs = {"LSTM": 300, "TCN": 200, "ResNet1D": 100}.get(model_type)

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
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=lr_factor, patience=10)
    criterion = torch.nn.MSELoss(reduction="mean")

    # Initialize loss function.
    loss_fc_lstm_tcn = lambda y_pred, y: criterion(y_pred.sum(dim=1), y.sum(dim=1))
    loss_fc_resnet1d = lambda y_pred, y: criterion(y_pred, y)
    loss_fc = lambda y_pred, y: loss_fc_lstm_tcn(y_pred, y) if model_type in ["LSTM", "TCN"] else loss_fc_resnet1d(y_pred, y)

    # If a training checkpoint exists, load it.
    if pathlib.Path(f"./comparison/models_data/ronin_{model_type.lower():s}_checkpoint.pkl").exists():
    
        checkpoint = joblib.load(f"./comparison/models_data/ronin_{model_type.lower():s}_checkpoint.pkl")
        best_loss_val = checkpoint["best_loss_val"]
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        i_start_epoch = checkpoint["i_epoch"] + 1

    # Otherwise, initialize to default values.
    else:
    
        i_start_epoch = 1
        best_loss_val = np.inf

    # Iterate for N epochs.
    for i_epoch in range(i_start_epoch, n_epochs + 1):

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
        for x, y in dataloader_train:

            # For ResNet1D, swap axes.
            if model_type == "ResNet1D":
                x = x.swapaxes(1, 2)

            # Transfer data to GPU.
            x, y = x.cuda(), y.cuda()

            # Forward.
            y_pred = model(x)

            # Compute loss.
            loss = loss_fc(y_pred, y)
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

        # Iterate over validation dataset.
        for x, y in dataloader_val:

            # For ResNet1D, swap axes.
            if model_type == "ResNet1D":
                x = x.swapaxes(1, 2)

            # Transfer data to GPU.
            x, y = x.cuda(), y.cuda()

            # Forward.
            with torch.no_grad():
                y_pred = model(x)

            # Compute loss.
            loss = loss_fc(y_pred, y)
            loss_val += loss.item() * len(x)
            n_samples_val += len(x)

        # Compute validation epoch loss.
        loss_val /= n_samples_val

        # If a new best validation loss is reached, save the model state dict.
        if loss_val < best_loss_val:
            best_loss_val = loss_val
            torch.save(model.state_dict(), f"./comparison/models_data/ronin_{model_type.lower():s}_state_dict.pt")

        # Step for LR scheduler.
        lr_scheduler.step(loss_val)

        # Save a checkpoint every 5 epochs.
        if i_epoch % 5 == 0:
            checkpoint = {
                "i_epoch": i_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                "best_loss_val": best_loss_val
            }
            joblib.dump(value=checkpoint, filename=f"./comparison/models_data/ronin_{model_type.lower():s}_checkpoint.pkl")

        # Print verbose.
        print(f"Epoch {i_epoch:03d}: loss train = {loss_train:.3f} loss val = {loss_val:.3f} {'NEW BEST' if loss_val == best_loss_val else '':s}")