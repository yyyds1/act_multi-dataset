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
    def compute_reference_points(pose, time_stamp, fps):
        """
        Comupte the reference points from poses sampled from demo.Consisted of 3 steps:
        1. Add noise to the sampled poses.
        2. Generate a continous reference trajactory from given poses.
        3. Sample the reference points from the trajactory.
        
        Parameters:
            pose: key poses sampled from demo
            time_stamp: the time stamp of the key poses
            fps: fps of demo

        Returns:
             a tensor contains the reference points.
        """
        # TODO
