
import h5py
import pybullet as p
import pybullet_data
import os
import numpy as np
import tempfile
from scipy.spatial.transform import Rotation as R

def get_transform(pos, quat_xyzw):
    T = np.eye(4)
    T[:3, 3] = pos
    T[:3, :3] = R.from_quat(quat_xyzw).as_matrix()
    return T

def load_panda_urdf(assets_dir):
    # Load URDF and replace package://Panda with absolute path
    urdf_path = os.path.join(assets_dir, 'Panda', 'panda.urdf')
    with open(urdf_path, 'r') as f:
        urdf_content = f.read()
    
    # Replace package://Panda with absolute path
    panda_assets_path = os.path.join(assets_dir, 'Panda')
    urdf_content = urdf_content.replace('package://Panda', panda_assets_path)
    
    # Create temp file
    temp_urdf = tempfile.NamedTemporaryFile(delete=False, suffix='.urdf', mode='wb')
    temp_urdf.write(urdf_content.encode('utf-8'))
    temp_urdf.close()
    return temp_urdf.name

def main():
    dataset_path = "/mnt/nas/datasets_tmp/LIBERO/libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5"
    assets_dir = "/home/yds/code/act_multi-dataset/assets"
    
    # Init PyBullet
    p.connect(p.DIRECT)
    panda_urdf = load_panda_urdf(assets_dir)
    panda_id = p.loadURDF(panda_urdf, useFixedBase=True)
    
    # Identify EE link
    # Usually panda_hand or panda_link8.
    # Let's search for "panda_hand"
    ee_idx = -1
    for i in range(p.getNumJoints(panda_id)):
        info = p.getJointInfo(panda_id, i)
        name = info[12].decode('utf-8')
        if "hand" in name: # panda_hand
            ee_idx = i
            print(f"Found EE link: {name} at index {i}")
            break
            
    if ee_idx == -1:
        # Fallback to last link?
        ee_idx = p.getNumJoints(panda_id) - 1
        print(f"Using last link as EE: {ee_idx}")

    with h5py.File(dataset_path, 'r') as f:
        demo_key = list(f['data'].keys())[0]
        print(f"Checking episode: {demo_key}")
        
        obs = f[f'data/{demo_key}/obs']
        joint_states = obs['joint_states'][()]
        
        # In Libero, 'ee_states' is (N, 6) -> pos(3), rotvec(3).
        if 'ee_states' in obs:
            ee_states = obs['ee_states'][()]
        else:
            print("ee_states not found")
            return

        print(f"ee_states shape: {ee_states.shape}")
        
    # sample 5 points
    for t in [0, 50, 100, 150, 200]:
        q = joint_states[t] # 7 values
        
        # Reset PyBullet
        # PyBullet panda joints usually 0-6 (7 DoF) + fingers
        for j in range(7):
            p.resetJointState(panda_id, j, q[j])
            
        # Get PyBullet EE Pose (Base is 0)
        ls = p.getLinkState(panda_id, ee_idx)
        pb_pos = np.array(ls[4])
        pb_ori = np.array(ls[5]) # xyzw
        
        # Convert pb_ori to matrix
        T_pb = get_transform(pb_pos, pb_ori)
        
        # Get Dataset EE Pose
        ds_pos = ee_states[t, :3]
        ds_ori_6d = ee_states[t, 3:]
        
        # Convert ds_ori to matrix. Assuming Axis Angle.
        if np.linalg.norm(ds_ori_6d) < 1e-6:
             R_ds = np.eye(3)
        else:
             R_ds = R.from_rotvec(ds_ori_6d).as_matrix()
             
        T_ds = np.eye(4)
        T_ds[:3, 3] = ds_pos
        T_ds[:3, :3] = R_ds
        
        # Compute T_root = T_ds * T_pb^-1
        # because T_world_ee = T_world_base * T_base_ee
        # so T_world_base = T_world_ee * T_base_ee^-1
        T_root = T_ds @ np.linalg.inv(T_pb)
        
        root_pos = T_root[:3, 3]
        root_quat = R.from_matrix(T_root[:3, :3]).as_quat() # xyzw
        
        print(f"T={t}")
        print(f"  Calc Root Pos: {root_pos}")
        print(f"  Calc Root Rot(deg): {R.from_quat(root_quat).as_euler('xyz', degrees=True)}")
        
    p.disconnect()

if __name__ == '__main__':
    main()
