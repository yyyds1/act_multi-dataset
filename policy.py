import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms

from detr.main import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer
import IPython
e = IPython.embed

class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
        print(f'KL Weight {self.kl_weight}')

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            return loss_dict
        else: # inference time
            a_hat, _, (_, _) = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


class CNNMLPPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model # decoder
        self.optimizer = optimizer

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None # TODO
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, 0]
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)
            loss_dict = dict()
            loss_dict['mse'] = mse
            loss_dict['loss'] = loss_dict['mse']
            return loss_dict
        else: # inference time
            a_hat = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


class ACTPolicy_MultiDataset(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
        print(f'KL Weight {self.kl_weight}')

    def __call__(self, obs, history_obs=None, actions=None, is_pad=None):
        # obs is a dict with: 'joint_pose', 'gripper_pose', 'images', 'eepose', 'ref_point'
        
        # 1. Construct qpos: [B, 9] (7 joint + 2 gripper)
        joint_pose = obs['joint_pose']
        gripper_pose = obs['gripper_pose']
        qpos = torch.cat([joint_pose, gripper_pose], dim=-1)

        # 1.5 Add history to qpos if available
        if history_obs is not None:
            # history_obs: dict of sequences [B, H, ...] 
            h_joint = history_obs['joint_pose']
            
            # Check if history is populated (list or tensor)
            has_history = False
            if isinstance(h_joint, list) and len(h_joint) > 0:
                has_history = True
            elif isinstance(h_joint, torch.Tensor) and h_joint.numel() > 0:
                has_history = True
                
            if has_history:
                # Note: LiberoDataset constructs history as [t-1, t-2, ...]. We typically want [t-H, ..., t-1].
                h_joint = history_obs['joint_pose'] # [B, H, 7]
                h_gripper = history_obs['gripper_pose'] # [B, H, 2]
                
                # Flip along time dimension (dim=1) to get chronological order
                h_joint = torch.flip(h_joint, dims=[1])
                h_gripper = torch.flip(h_gripper, dims=[1])
                
                h_qpos = torch.cat([h_joint, h_gripper], dim=-1) # [B, H, 9]
                B, H, D = h_qpos.shape
                h_qpos = h_qpos.reshape(B, H * D)
                
                qpos = torch.cat([h_qpos, qpos], dim=-1) # [B, H*9 + 9]

        # 2. Construct image: [B, NumCam, 3, H, W]
        # obs['images'] is dict of [B, H, W, 3] (based on LiberoDataset)
        images_dict = obs['images']
        
        # Sort keys for deterministic order
        image_tensors = []
        for cam_name in sorted(images_dict.keys()):
            img = images_dict[cam_name] # [B, H, W, 3]
            image_tensors.append(img)
            
        image = torch.stack(image_tensors, dim=1) # [B, NumCam, H, W, 3]
        
        # Permute to [B, NumCam, 3, H, W]
        image = image.permute(0, 1, 4, 2, 3) 
        
        # Normalize
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        
        # Normalize expects (B, C, H, W), reshape momentarily
        B, NC, C, H, W = image.shape
        image = image.reshape(B * NC, C, H, W)
        image = normalize(image)
        image = image.reshape(B, NC, C, H, W)

        if actions is not None: # training time
            actions = actions[:, :self.model.num_queries]
            
            # Handle is_pad if not provided
            if is_pad is None:
                is_pad = torch.zeros(actions.shape[0], actions.shape[1], dtype=torch.bool, device=actions.device)
            else:
                is_pad = is_pad[:, :self.model.num_queries]

            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            return loss_dict
        else: # inference time
            a_hat, _, (_, _) = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld
