from .base import BaseDataset
import os
import numpy as np
import torch
import h5py
import json

class LiberoDataset(BaseDataset):
    def __init__(
        self,
        *,
        dataset_dir, 
        episode_ids = None, 
        camera_names = None, 
        norm_stats = None,
    ):
        super().__init__(
            episode_ids=episode_ids,
            dataset_dir=dataset_dir,
            camera_names=camera_names,
            norm_stats=norm_stats,
        )
        # Additional initialization for LiberoDataset can be added here

    def __getitem__(self, idx):
        """
        Get an episode from the Libero dataset.
        
        Parameters:
            idx: index of the episode
        
        Returns:
            A Sequence (obs, act)

            obs: A dict with keys:
                1. "joint_pose": Franka Panda 7D joint angle, [seq_len, 7]
                2. "eepose": Franka Panda 7D end effector pose in root frame, consist of transition (x, y, z) and orientation quat (w, x, y, z), [seq_len, 7]
                3. "gripper_pose": Franka Panda 2D gripper state, [seq_len, 2]
                4. "images": RGB images of side view and wrist view, normalize to [0, 1], [seq_len, H, W, 3]
                5. "ref_point": end effector reference 7D poses in root frame, [seq_len, 7]

            act: A tensor contains action
        """
        episode_id = self.episode_ids[idx]
        with h5py.File(self.dataset_dir, "r") as f:
            if f"data/{episode_id}" in f:
                group_key = f"data/{episode_id}"
            elif f"data/demo_{episode_id}" in f:
                group_key = f"data/demo_{episode_id}"
            else:
                raise KeyError(f"Episode {episode_id} not found in {self.dataset_dir}")
            
            demo_grp = f[group_key]
            actions = demo_grp["actions"][()]
            
            obs_grp = demo_grp["obs"]
            joint_pose = obs_grp["joint_states"][()]
            eepose = obs_grp["ee_states"][()]
            gripper_pose = obs_grp["gripper_states"][()]
            
            images = {}
            for cam in self.camera_names:
                if cam in obs_grp:
                    images[cam] = obs_grp[cam][()]
                elif f"{cam}_rgb" in obs_grp:
                    images[cam] = obs_grp[f"{cam}_rgb"][()]
                elif cam == "eye_in_hand" and "robot0_eye_in_hand_rgb" in obs_grp:
                    images[cam] = obs_grp["robot0_eye_in_hand_rgb"][()]
                else:
                    raise KeyError(f"Camera {cam} not found in observations")
            
            if "env_args" in f["data"].attrs:
                env_args = json.loads(f["data"].attrs["env_args"])
                fps = env_args.get("control_freq", 20)
            else:
                fps = 20
        
        seq_len = actions.shape[0]
        
        # Sample key points
        interval = int(fps * 0.2)
        if interval < 1: interval = 1
        key_point_indices = set(range(0, seq_len, interval))
        
        # Add points where gripper state was modified
        if gripper_pose.ndim == 1:
            gripper_pose_2d = gripper_pose[:, None]
        else:
            gripper_pose_2d = gripper_pose
            
        delta = np.abs(gripper_pose_2d[1:] - gripper_pose_2d[:-1]).sum(axis=1)
        change_indices = np.where(delta > 1e-4)[0] + 1
        key_point_indices.update(change_indices.tolist())
        
        sorted_indices = sorted(list(key_point_indices))
        sorted_indices = [x for x in sorted_indices if x < seq_len]
        if not sorted_indices:
            sorted_indices = [0]
            
        # Compute reference points
        key_poses = torch.from_numpy(eepose[sorted_indices]).float().unsqueeze(0)
        key_times = torch.tensor([x / fps for x in sorted_indices], dtype=torch.float32).unsqueeze(0)
        
        ref_point = self.compute_reference_points(key_poses, key_times, fps, seq_len)
        ref_point = ref_point.squeeze(0)
        
        # Prepare obs dict
        obs = {
            "joint_pose": torch.from_numpy(joint_pose).float(),
            "eepose": torch.from_numpy(eepose).float(),
            "gripper_pose": torch.from_numpy(gripper_pose).float(),
            "images": {k: (torch.from_numpy(v).float() / 255.0) for k, v in images.items()},
            "ref_point": ref_point
        }
        
        act = torch.from_numpy(actions).float()
        
        # Normalize if stats are provided
        if self.norm_stats is not None:
            if "qpos_mean" in self.norm_stats:
                obs["joint_pose"] = (obs["joint_pose"] - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
            if "action_mean" in self.norm_stats:
                act = (act - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        
        return obs, act
