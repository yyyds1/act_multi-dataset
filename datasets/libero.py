from .base import BaseDataset
import os
import numpy as np
import torch
import h5py
import json
from scipy.spatial.transform import Rotation as R
from .decoraters import register_dataset

@register_dataset("libero")
class LiberoDataset(BaseDataset):
    def __init__(
        self,
        *,
        dataset_dir, 
        episode_ids = None, 
        camera_names = None,
        norm_stats = None,
        history_obs_len = 0,
    ):
        self.history_obs_len = history_obs_len
        resolved_episode_ids = episode_ids
        resolved_camera_names = camera_names

        if episode_ids is None or camera_names is None:
            with h5py.File(dataset_dir, "r") as f:
                data_grp = f["data"]
                episode_keys = list(data_grp.keys())

                if episode_ids is None:
                    parsed_ids = []
                    for key in episode_keys:
                        if key.startswith("demo_"):
                            suffix = key[len("demo_"):]
                            parsed_ids.append(int(suffix) if suffix.isdigit() else suffix)
                        elif key.isdigit():
                            parsed_ids.append(int(key))
                        else:
                            parsed_ids.append(key)

                    def sort_key(value):
                        return (0, value) if isinstance(value, int) else (1, str(value))

                    resolved_episode_ids = sorted(parsed_ids, key=sort_key)

                if camera_names is None:
                    episode_key = None
                    for key in sorted(episode_keys):
                        episode_key = key
                        break

                    if episode_key is None:
                        raise ValueError(f"No episodes found in {dataset_dir}")

                    obs_grp = data_grp[episode_key]["obs"]
                    cam_candidates = []
                    for key in obs_grp.keys():
                        if key in {"joint_states", "ee_states", "gripper_states"}:
                            continue
                        if key == "robot0_eye_in_hand_rgb":
                            cam_candidates.append("eye_in_hand")
                        elif key.endswith("_rgb"):
                            cam_candidates.append(key[:-4])
                        else:
                            cam_candidates.append(key)

                    seen = set()
                    resolved_camera_names = []
                    for cam in cam_candidates:
                        if cam not in seen:
                            resolved_camera_names.append(cam)
                            seen.add(cam)

        super().__init__(
            episode_ids=resolved_episode_ids,
            dataset_dir=dataset_dir,
            camera_names=resolved_camera_names,
            norm_stats=norm_stats,
        )

        self.episodes = []
        with h5py.File(self.dataset_dir, "r") as f:
            if "env_args" in f["data"].attrs:
                env_args = json.loads(f["data"].attrs["env_args"])
                fps = env_args.get("control_freq", 20)
            else:
                fps = 20
            
            for episode_id in self.episode_ids:
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

                if "ee_pos" in obs_grp and "ee_ori" in obs_grp:
                    ee_pos = obs_grp["ee_pos"][()]
                    ee_ori = obs_grp["ee_ori"][()]
                    # Convert axis-angle (3) to Quaternion (4) -> (w, x, y, z)
                    r = R.from_rotvec(ee_ori)
                    quat_xyzw = r.as_quat()
                    quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]
                    eepose = np.concatenate([ee_pos, quat_wxyz], axis=1)
                elif "ee_states" in obs_grp:
                    raw_ee = obs_grp["ee_states"][()]
                    if raw_ee.shape[1] == 6:
                        ee_pos = raw_ee[:, :3]
                        ee_ori = raw_ee[:, 3:]
                        r = R.from_rotvec(ee_ori)
                        quat_xyzw = r.as_quat()
                        quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]
                        eepose = np.concatenate([ee_pos, quat_wxyz], axis=1)
                    else:
                        eepose = raw_ee
                else:
                    raise KeyError("Could not find ee_states or ee_pos/ee_ori in observation")

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
                

                
                self.episodes.append({
                    "actions": actions,
                    "joint_pose": joint_pose,
                    "eepose": eepose,
                    "gripper_pose": gripper_pose,
                    "images": images,
                    "fps": fps
                })

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, idx):
        """
        Randomly get a frame in the episode with index idx from the dataset.
        
        Parameters:
            idx: index of the episode
        
        Returns:
            A Sequence (obs, history_obs, actions, is_pad) where:

            obs: A dict with keys:
                1. "joint_pose": Franka Panda 7D joint angle
                2. "eepose": Franka Panda 7D end effector pose in root frame, consist of transition (x, y, z) and orientation quat (w, x, y, z)
                3. "gripper_pose": Franka Panda 2D gripper state 
                4. "images": RGB images of side view and wrist view, normalize to [0, 1]
                5. "ref_point": end effector reference 7D poses in root frame

            history_obs: A dict with the same keys as obs, but contains the history observations before the current obs. The length of history is determined by history_obs_len.
            act: A tensor contains action
            is_pad: A boolean tensor indicating padding, [seq_len]
        """
        episode_data = self.episodes[idx]
        
        actions = episode_data["actions"]
        joint_pose = episode_data["joint_pose"]
        eepose = episode_data["eepose"]
        gripper_pose = episode_data["gripper_pose"]
        images = episode_data["images"]
        fps = episode_data["fps"]
        
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
        
        # Sample random timestep
        start_ts = np.random.choice(seq_len)
        
        # Construct observations at start_ts
        obs = {
            "joint_pose": torch.from_numpy(joint_pose[start_ts]).float(),
            "eepose": torch.from_numpy(eepose[start_ts]).float(),
            "gripper_pose": torch.from_numpy(gripper_pose[start_ts]).float(),
            "images": {k: (torch.from_numpy(v[start_ts]).float() / 255.0) for k, v in images.items()},
            "ref_point": ref_point[start_ts]
        }
        
        # Construct history_obs
        history_obs = {
            "joint_pose": [],
            "eepose": [],
            "gripper_pose": [],
            "images": {k: [] for k in images.keys()},
            "ref_point": []
        }
        
        if self.history_obs_len > 0:
            for i in range(1, self.history_obs_len + 1):
                idx = max(0, start_ts - i)
                history_obs["joint_pose"].append(torch.from_numpy(joint_pose[idx]).float())
                history_obs["eepose"].append(torch.from_numpy(eepose[idx]).float())
                history_obs["gripper_pose"].append(torch.from_numpy(gripper_pose[idx]).float())
                for k, v in images.items():
                    history_obs["images"][k].append(torch.from_numpy(v[idx]).float() / 255.0)
                history_obs["ref_point"].append(ref_point[idx])
                
            # Stack the history lists
            history_obs["joint_pose"] = torch.stack(history_obs["joint_pose"])
            history_obs["eepose"] = torch.stack(history_obs["eepose"])
            history_obs["gripper_pose"] = torch.stack(history_obs["gripper_pose"])
            for k in history_obs["images"]:
                history_obs["images"][k] = torch.stack(history_obs["images"][k])
            history_obs["ref_point"] = torch.stack(history_obs["ref_point"])
        
        # Construct actions from start_ts
        action = actions[start_ts:]
        action_len = len(action)
        
        padded_action = np.zeros_like(actions)
        padded_action[:action_len] = action
        
        is_pad = np.zeros(seq_len)
        is_pad[action_len:] = 1
        
        act = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()
        
        # Normalize if stats are provided
        if self.norm_stats is not None:
            if "qpos_mean" in self.norm_stats:
                obs["joint_pose"] = (obs["joint_pose"] - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
            if "action_mean" in self.norm_stats:
                act = (act - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        
        return obs, history_obs, act, is_pad
