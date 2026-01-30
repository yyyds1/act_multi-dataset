from abc import ABC, abstractmethod
import os
import numpy as np
import torch

class BaseDataset(torch.utils.data.Dataset, ABC):
    def __init__(
        self,
        *,
        episode_ids, 
        dataset_dir, 
        camera_names, 
        norm_stats,
    ):
        super.__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats

    @abstractmethod
    def __getitem__(self, idx):
        """
        Get an episode form the dataset.
        
        Parameters:
            idx: index of the episode
        
        Returns:
            A Sequence (obs, act)

            obs: A dict with keys:
                1. "joint_pose": Franka Panda 7D joint angle
                2. "eepose": Franka Panda 7D end effector pose in root frame, consist of transition (x, y, z) and orientation quat (w, x, y, z)
                3. "gripper_pose": Franka Panda 2D gripper state 
                4. "images": RGB images of side view and wrist view, normalize to [0, 1]
                5. "ref_point": end effector reference 7D poses in root frame

            act: A tensor contains action
        """
        pass

    @staticmethod
    def compute_reference_points(pose: torch.Tensor, time_stamp: torch.Tensor, fps: int, seq_len: int) -> torch.Tensor:
        """
        Comupte the reference points from poses sampled from demo.Consisted of 3 steps:
        1. Add Gaussion noise to the sampled poses.
        2. Generate a continous reference trajactory from given poses.
        3. Sample the reference points from the trajactory.
        
        Parameters:
            pose: key poses sampled from demo, [batch_size, point_num, 7]
            time_stamp: the time stamp of the key poses, [batch_size, point_num]
            fps: fps of demo
            seq_len: length of the reference points to be generated.

        Returns:
             a tensor contains the reference points, [batch_size, seq_len, 7].
        """
        B, N, D = pose.shape
        device = pose.device

        # 1. Add Gaussian noise to the sampled poses
        noise_std = 0.01
        noisy_pose = pose + torch.randn_like(pose) * noise_std

        # 2. Generate target query times
        # We want to sample at t = 0, 1/fps, 2/fps ...
        query_times = torch.arange(seq_len, device=device, dtype=pose.dtype) / fps
        # Broadcast to batch size: [B, seq_len]
        query_times = query_times.unsqueeze(0).expand(B, -1)

        # 3. Locate interpolation intervals
        # Find indices such that time_stamp[i-1] <= query_time < time_stamp[i]
        right_idx = torch.searchsorted(time_stamp, query_times, right=True)
        
        # Clamp indices to valid range [1, N-1] so we can always access left neighbor
        right_idx = torch.clamp(right_idx, 1, N - 1)
        left_idx = right_idx - 1

        # 4. Gather data for interpolation
        # Get timestamps at boundaries [B, seq_len]
        t_left = torch.gather(time_stamp, 1, left_idx)
        t_right = torch.gather(time_stamp, 1, right_idx)

        # Get poses at boundaries. Need to expand indices to [B, seq_len, D]
        left_idx_exp = left_idx.unsqueeze(-1).expand(-1, -1, D)
        right_idx_exp = right_idx.unsqueeze(-1).expand(-1, -1, D)
        
        p_left = torch.gather(noisy_pose, 1, left_idx_exp)
        p_right = torch.gather(noisy_pose, 1, right_idx_exp)

        # 5. Linear Interpolation (Lerp)
        dt = t_right - t_left
        # Avoid division by zero
        dt = torch.where(dt < 1e-6, torch.ones_like(dt) * 1e-6, dt)
        
        alpha = (query_times - t_left) / dt
        # Clamp alpha to [0, 1] to hold edge values if query_times exceed available range
        alpha = torch.clamp(alpha, 0.0, 1.0).unsqueeze(-1)

        reference_points = torch.lerp(p_left, p_right, alpha)
        
        return reference_points
