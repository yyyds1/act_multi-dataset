
import os
import sys
import argparse
import numpy as np
import torch
import cv2
import pybullet as p
import pybullet_data
import tempfile
import h5py

# Add parent directory to sys.path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from datasets.factory import DataFactory
# Import datasets package to trigger auto-registration
import datasets

# Ensure LiberoDataset is registered logic
if "libero" not in DataFactory._registry:
    try:
        from datasets.libero import LiberoDataset
        DataFactory.register("libero", LiberoDataset)
    except ImportError:
        pass

def load_panda_urdf(assets_dir):
    # Load URDF and replace package://Panda with absolute path
    urdf_path = os.path.join(assets_dir, 'Panda', 'panda.urdf')
    with open(urdf_path, 'r') as f:
        urdf_content = f.read()
    
    # Replace package://Panda with absolute path
    panda_assets_path = os.path.join(assets_dir, 'Panda')
    urdf_content = urdf_content.replace('package://Panda', panda_assets_path)
    
    # Create temp file
    temp_urdf = tempfile.NamedTemporaryFile(delete=False, suffix='.urdf', mode='w')
    temp_urdf.write(urdf_content.encode('utf-8'))
    temp_urdf.close()
    return temp_urdf.name

def main(args):
    dataset_name = args.dataset
    dataset_path_arg = args.dataset_dir
    
    if not os.path.exists(dataset_path_arg):
        print(f"Dataset path does not exist: {dataset_path_arg}")
        return

    # Initialize PyBullet
    cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # Load Panda
    assets_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../assets'))
    panda_urdf_path = load_panda_urdf(assets_dir)
    try:
        panda_id = p.loadURDF(panda_urdf_path, useFixedBase=True)
    except Exception as e:
        print(f"Failed to load URDF: {e}")
        os.unlink(panda_urdf_path)
        return
    finally:
        if os.path.exists(panda_urdf_path):
             os.unlink(panda_urdf_path)

    # Reset view
    p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=[0, 0, 0.5])

    # --------------------------------------------------------
    # Determine Episodes and Cameras
    # --------------------------------------------------------
    with h5py.File(dataset_path_arg, 'r') as f:
        # Get all keys that look like episodes
        keys = list(f['data'].keys())
        # Sort keys
        def get_key_idx(k):
            # Extract number from demo_10 or 10
            if k.startswith('demo_'):
                try: return int(k.split('_')[1])
                except: return k
            elif k.isdigit():
                return int(k)
            else:
                return k
        
        # We sort them to have a deterministic index mapping
        all_episode_keys = sorted(keys, key=get_key_idx)
        
        # Select episodes
        selected_keys = []
        if args.episode_idx is not None:
            # args.episode_idx is a list of integers
            for idx in args.episode_idx:
                if 0 <= idx < len(all_episode_keys):
                    selected_keys.append(all_episode_keys[idx])
                else:
                    print(f"Warning: Episode index {idx} out of range (0-{len(all_episode_keys)-1})")
        else:
            # Random episode
            import random
            selected_keys = [random.choice(all_episode_keys)]
        
        if not selected_keys:
            print("No episodes selected.")
            p.disconnect()
            return
            
        # Determine Cameras from first selected episode
        first_ep_key = selected_keys[0]
        obs_grp = f[f"data/{first_ep_key}/obs"]
        
        # Identify available cameras
        available_cameras = []
        for k in obs_grp.keys():
            # Heuristic to identify camera images
            if k.endswith('_rgb') or k.endswith('_image'):
                dataset_cam_name = k
                # Infer the logical name to pass to LiberoDataset
                # LiberoDataset __getitem__ checks `cam in obs_grp` OR `f"{cam}_rgb" in obs_grp`
                # So we prefer the base name without _rgb if possible
                if k.endswith('_rgb'):
                     logical_name = k[:-4]
                elif k.endswith('_image'):
                     logical_name = k[:-6]
                else:
                     logical_name = k
                
                # However, if we pass logical name, LiberoDataset must find it.
                # If obs_grp has "agentview_rgb", and we pass "agentview", LiberoDataset checks "agentview" (False) then "agentview_rgb" (True).
                # So "agentview" is safe.
                # If obs_grp has "robot0_eye_in_hand_rgb", passing "eye_in_hand" works if logic handles it.
                # LiberoDataset snippet logic: 
                # elif cam == "eye_in_hand" and "robot0_eye_in_hand_rgb" in obs_grp: ...
                
                # Check for specific Libero mapping
                if k == "robot0_eye_in_hand_rgb":
                     available_cameras.append("eye_in_hand")
                else:
                     available_cameras.append(logical_name)
    
        # Filter cameras based on arguments
        final_cameras = []
        if args.camera_idx is not None:
             for cam_arg in args.camera_idx:
                 if cam_arg in available_cameras:
                     final_cameras.append(cam_arg)
                 else:
                     # Check if it was formatted differently?
                     print(f"Warning: Camera {cam_arg} not found in available: {available_cameras}")
        else:
             final_cameras = available_cameras
        
        if not final_cameras:
            print("No cameras selected or found. Visualizing render only.")
            
    # --------------------------------------------------------
    # Load Dataset and Visualize
    # --------------------------------------------------------
    
    # We create one dataset instance containing the selected episodes
    dataset = DataFactory.create_data(
        dataset_name,
        dataset_dir=dataset_path_arg,
        episode_ids=selected_keys,
        camera_names=final_cameras,
        norm_stats=None
    )
    
    # Create valid ref point visual
    ref_sphere_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.03, rgbaColor=[1, 0, 0, 1])
    ref_body_id = p.createMultiBody(baseVisualShapeIndex=ref_sphere_id)
    
    print(f"Visualizing {len(dataset)} episodes...")
    
    for i in range(len(dataset)):
        obs, act = dataset[i]
        ep_key = selected_keys[i]
        print(f"Rendering Episode {ep_key}...")
        
        # Unpack Data
        # Ensure we convert to numpy
        joint_pose = obs['joint_pose'].numpy()
        ref_points = obs['ref_point'].numpy()
        
        seq_len = joint_pose.shape[0]
        
        output_file = f"visualize_{dataset_name}_{ep_key}.mp4"
        
        # Setup Video Writer
        # Render Size
        render_w, render_h = 640, 480
        
        # Inspect Image Size
        cam_h, cam_w = 0, 0
        if final_cameras:
            # Check dimensions of first camera
            # obs['images'][cam] shape?
            first_cam = final_cameras[0]
            img_sample = obs['images'][first_cam][0] # T, C, H, W or T, H, W, C
            # PyTorch from_numpy preserves shape. Libero usually (T, H, W, C).
            # But we should verify. 
            # If shape[-1] == 3, it is HWC. If shape[-3] == 3 (and ndim 4), it is CHW?
            # Actually obs['images'] values are tensors.
            # If LiberoDataset doesn't transpose, it is whatever h5py gave.
            # Assuming HWC from h5py.
            
            if img_sample.shape[-1] == 3: # HWC
                 cam_h, cam_w = img_sample.shape[0], img_sample.shape[1]
            elif img_sample.shape[0] == 3: # CHW
                 cam_h, cam_w = img_sample.shape[1], img_sample.shape[2]
            else:
                 # Fallback
                 cam_h, cam_w = img_sample.shape[0], img_sample.shape[1]

        # Layout: Render Left, Stack of Cameras Right
        # Total Width = RenderW + CamW
        # Total Height = max(RenderH, CamH * NumCams)
        
        num_cams = len(final_cameras)
        total_w = render_w + (cam_w if num_cams > 0 else 0)
        total_h = max(render_h, (cam_h * num_cams) if num_cams > 0 else 0)
        
        # Ensure divisible by 2 for video codec
        if total_w % 2 != 0: total_w += 1
        if total_h % 2 != 0: total_h += 1
        
        fps = 20.0
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_file, fourcc, fps, (total_w, total_h))
        
        for t in range(seq_len):
            # 1. Update Simulation
            jp = joint_pose[t] # 7D usually
            # Set Panda joints (0-6)
            for j in range(min(7, len(jp))):
                p.resetJointState(panda_id, j, jp[j])
                
            # Update Gripper? 
            # If jp has more than 7 dims, assume gripper. 
            # Usually qpos is 7 (joints) + 2 (gripper).
            if len(jp) > 7:
                 gripper_val = jp[7] # width?
                 # Panda has 2 finger joints: panda_finger_joint1, panda_finger_joint2
                 # Indices: usually 9, 10 in full body, but loaded fixed base:
                 # getNumJoints
                 # panda_finger_joint1 is 9, panda_finger_joint2 is 10 (approx)
                 # We can just ignore visual fidelity of gripper for now if indices vary.
                 pass

            # Update Ref Point
            rp = ref_points[t] # 7 floats
            if len(rp) >= 3:
                p.resetBasePositionAndOrientation(ref_body_id, rp[:3], [0,0,0,1])

            # 2. Render
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[0, 0, 0.5],
                distance=1.2,
                yaw=90,
                pitch=-30,
                roll=0,
                upAxisIndex=2
            )
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=60,
                aspect=float(render_w)/render_h,
                nearVal=0.1,
                farVal=100.0
            )
            (_, _, px, _, _) = p.getCameraImage(
                width=render_w,
                height=render_h,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL
            )
            rgb_render = np.array(px, dtype=np.uint8).reshape((render_h, render_w, 4))[:, :, :3]
            rgb_render = cv2.cvtColor(rgb_render, cv2.COLOR_RGB2BGR)
            
            # 3. Process Camera Images
            # Stack them vertically
            cam_stack = []
            if final_cameras:
                for c_name in final_cameras:
                    img_t = obs['images'][c_name][t] # Tensor
                    # Convert to numpy [H, W, 3] BGR
                    img_np = img_t.numpy()
                    
                    # Handle CHW vs HWC
                    if img_np.shape[0] == 3: # CHW
                        img_np = np.transpose(img_np, (1, 2, 0))
                    
                    # Normalize / Scale ?
                    # LiberoDataset divides by 255.0. So it is 0..1 float.
                    img_np = (img_np * 255).astype(np.uint8)
                    
                    # RGB to BGR
                    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                    
                    # Resize to fixed cam_w, cam_h if needed (should match)
                    if img_bgr.shape[0] != cam_h or img_bgr.shape[1] != cam_w:
                         img_bgr = cv2.resize(img_bgr, (cam_w, cam_h))
                         
                    cam_stack.append(img_bgr)
            
            # Combine
            canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)
            
            # Place Render (Centers vertically)
            y_off_render = (total_h - render_h) // 2
            canvas[y_off_render:y_off_render+render_h, 0:render_w] = rgb_render
            
            # Place Cams
            if cam_stack:
                # Stack vertical
                full_cam_col = np.concatenate(cam_stack, axis=0) # (Num*H, W, 3)
                # If total height of cams < total_h, center it
                h_cams_total = full_cam_col.shape[0]
                y_off_cams = (total_h - h_cams_total) // 2
                
                # Careful with bounds
                target_h = min(total_h, h_cams_total)
                canvas[y_off_cams:y_off_cams+target_h, render_w:render_w+cam_w] = full_cam_col[:target_h, :]

            out.write(canvas)
            
        out.release()
        print(f"Saved video: {output_file}")


    p.disconnect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset (e.g. libero)')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Path to the .hdf5 dataset file')
    parser.add_argument('--episode_idx', type=int, nargs='+', default=None, help='List of episode indices to visualize')
    parser.add_argument('--camera_idx', type=str, nargs='+', default=None, help='List of camera names to visualize')
    
    args = parser.parse_args()
    main(args)
