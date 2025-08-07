import numpy as np
import torch
from typing import Self, Tuple, Dict


class MasksManager:

    def __init__(
        self: Self,
        training_type: str,
        len_seq: int,
        min_len_gps_outage: int,
        max_len_gps_outage: int,
        relax_points: Dict[str, Dict[str, np.ndarray]]
    ) -> None:

        """
        Initialize a MasksManager instance that hanlde masks generation and application.

        Args:
            - training_type (str): The type of training (GPS outage placement). Can be either 'beginning', 'centered', 'end' or 'random'.
            - len_seq (int): The length of the sequence.
            - min_len_gps_outage (int): The minimum length of a GPS outage.
            - max_len_gps_outage (int): The maximum length of a GPS outage.
            - relax_points (Dict[str, Dict[str, np.ndarray]]): The relax points (scaled) of the dataset (velocity GPS = 0 / velocity fusion = 0 / orientation fusion = 0).
        """

        # Memorize parameters.
        self.training_type = training_type
        self.len_seq = len_seq
        self.min_len_gps_outage = min_len_gps_outage
        self.max_len_gps_outage = max_len_gps_outage

        # Memorize relax points.
        self.relax_points = relax_points

    def generate_masks(
        self: Self,
        batch_size: int
    ) -> torch.Tensor:

        """
        Generate the masks to simulate the GPS outages.

        Args:
            - batch_size (int): The number of sample in a batch.

        Returns:
            - masks (torch.Tensor): A boolean tensor that indicate the GPS outages of size = (batch_size, len_seq, 1).
        """

        # Pick a random GPS outage duration (different for every sample of the batch).
        sizes_mask = np.random.randint(
            low=self.min_len_gps_outage,
            high=self.max_len_gps_outage + 1,
            size=batch_size
        )

        if self.training_type == "beginning":

            # The size after the GPS outage is the size of the sequence - the size of the GPS outage.
            sizes_unmask = self.len_seq - sizes_mask

            # Construct the masks by stacking every mask sample.
            masks = torch.stack([
                torch.cat([
                    torch.ones(size=(size_mask, 1), dtype=torch.bool),
                    torch.zeros(size=(size_unmask, 1), dtype=torch.bool)
                ], dim=0)
                for size_mask, size_unmask in zip(sizes_mask, sizes_unmask)
            ], dim=0)

        elif self.training_type == "centered":

            # The left and right size are the half of the length of the sequence - the size of the GPS outage.
            sizes_left = (self.len_seq - sizes_mask) // 2
            sizes_right = self.len_seq - sizes_mask - sizes_left

            # Construct the masks by stacking every mask sample.
            masks = torch.stack([
                torch.cat([
                    torch.zeros(size=(size_left, 1), dtype=torch.bool),
                    torch.ones(size=(size_mask, 1), dtype=torch.bool),
                    torch.zeros(size=(size_right, 1), dtype=torch.bool)
                ], dim=0)
                for size_mask, size_left, size_right in zip(sizes_mask, sizes_left, sizes_right) 
            ], dim=0)

        elif self.training_type == "end":

            # The size before the GPS outage is the size of the sequence - the size of the GPS outage.
            sizes_unmask = self.len_seq - sizes_mask

            # Construct the masks by stacking every mask sample.
            masks = torch.stack([
                torch.cat([
                    torch.zeros(size=(size_unmask, 1), dtype=torch.bool),
                    torch.ones(size=(size_mask, 1), dtype=torch.bool)
                ], dim=0)
                for size_mask, size_unmask in zip(sizes_mask, sizes_unmask)
            ], dim=0)

        elif self.training_type == "random":

            # The left size is pick random between 0 and the size of the seq - the size of the GPS outage.
            # The right size is the remaining size.
            sizes_left = np.random.randint(
                low=0,
                high=self.len_seq - sizes_mask + 1,
                size=batch_size
            )
            sizes_right = self.len_seq - sizes_mask - sizes_left

            # Construct the masks by stacking every mask sample.
            masks = torch.stack([
                torch.cat([
                    torch.zeros(size=(size_left, 1), dtype=torch.bool),
                    torch.ones(size=(size_mask, 1), dtype=torch.bool),
                    torch.zeros(size=(size_right, 1), dtype=torch.bool)
                ], dim=0)
                for size_mask, size_left, size_right in zip(sizes_mask, sizes_left, sizes_right) 
            ], dim=0)

        return masks

    def apply_masks(
        self: Self,
        masks: torch.Tensor,
        x_velocity_gps: torch.Tensor,
        x_velocity_fusion: torch.Tensor,
        x_orientation_fusion: torch.Tensor,
        scaling_type: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        """
        Mask the GPS and fusion data.

        Args:
            - masks (torch.Tensor): A boolean tensor that indicate the GPS outages of size = (batch_size, len_seq, 1).
            - x_velocity_gps (torch.Tensor): A tensor that contains the GPS velocity data.
            - x_velocity_fusion (torch.Tensor): A tensor that contains the fusion velocity data.
            - x_orientation_fusion (torch.Tensor): A tensor that contains the fusion orientation data.
            - scaling_type (int): Either 'normalization' for a min-max scaling between -1 and 1 or 'standardization' for a standard scaling with zero mean and unit std.

        Returns:
            - x_velocity_gps (torch.Tensor): The GPS velocity data masked during the GPS outages.
            - x_velocity_fusion (torch.Tensor): The fusion velocity data masked during the GPS outages.
            - x_orientation_fusion (torch.Tensor): The fusion orientation data masked during the GPS outages.
        """

        # Mask GPS velocity data if the sensor provide at least GPS velocity on one axe.
        if x_velocity_gps.size(dim=-1) > 0:
            x_velocity_gps = torch.cat([
                x_velocity_gps[:, :, idx : idx + 1].masked_fill(mask=masks, value=self.relax_points[scaling_type]["velocity_gps"][idx])
                for idx in range(x_velocity_gps.size(dim=-1))
            ], dim=-1)

        # Mask fusion velocity data
        x_velocity_fusion = torch.cat([
            x_velocity_fusion[:, :, idx : idx + 1].masked_fill(mask=masks, value=self.relax_points[scaling_type]["velocity_fusion"][idx])
            for idx in range(x_velocity_fusion.size(dim=-1))
        ], dim=-1)

        # Mask fusion orientation data if the sensor provide at least fusion orientation on one axe.
        if x_orientation_fusion.size(dim=-1) > 0:
            x_orientation_fusion = torch.cat([
                x_orientation_fusion[:, :, idx : idx + 1].masked_fill(mask=masks, value=self.relax_points[scaling_type]["orientation_fusion"][idx])
                for idx in range(x_orientation_fusion.size(dim=-1))
            ], dim=-1)

        return x_velocity_gps, x_velocity_fusion, x_orientation_fusion