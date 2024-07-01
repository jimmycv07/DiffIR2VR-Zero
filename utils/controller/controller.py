import gc

import torch
import torch.nn.functional as F

from einops import repeat, rearrange
from utils.flow_utils import flow_warp, coords_grid

# AdaIn


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


class AttentionControl():

    def __init__(self,
                 warp_period=(0.0, 0.0),
                 merge_ratio=(0.3, 0.3),
                 ToMe_period=(0.0, 1.0),
                 mask_period=(0.0, 0.0),
                 loose_cfatnn=False,
                ):
        
        self.cur_frame_idx = 0

        self.step_store = self.get_empty_store()
        self.cur_step = 0
        self.total_step = 0
        self.cur_index = 0
        self.init_store = False
        self.restore = False
        self.update = False
        self.restorex0 = True
        self.updatex0 = False
        self.warp_period = warp_period
        self.ToMe_period = ToMe_period
        self.merge_ratio = merge_ratio
        self.keyframe_idx = 0
        self.distances = {}
        self.flow_correspondence = {}
        self.non_pad_ratio = (1.0, 1.0)
        self.up_resolution = 1280 if loose_cfatnn else 1281

    @staticmethod
    def get_empty_store():
        return {
            'first': [],
            'previous': [],
            'x0_previous': [],
            'first_ada': [],
            'pre_x0': [],
            "pre_keyframe_lq": None,
            "flows": None,
            "occ_masks": None,
            "flow_confids": None,
        }

    def update_x0(self, x0, cur_frame=0):
        if self.restorex0:
            if self.cur_step >= self.total_step * self.warp_period[
                    0] and self.cur_step < int(self.total_step * self.warp_period[1]):
                
                key_idx = x0.shape[0] // 2
                if len(self.step_store["pre_x0"]) == int(self.total_step * self.warp_period[1]):
                    # print(f"[INFO] keyframe latent warping @ step {self.cur_step}...")
                    x0[key_idx] = (1 - self.step_store["occ_masks"][key_idx]) * x0[key_idx] + \
                        flow_warp(self.step_store["pre_x0"][self.cur_step][None], self.step_store["flows"][key_idx], mode='nearest')[0] * self.step_store["occ_masks"][key_idx] 
                    
                # print(f"[INFO] local latent warping @ step {self.cur_step}...")
                for i in range(x0.shape[0]):
                    if i == key_idx:
                        continue
                    x0[i] = (1 - self.step_store["occ_masks"][i]) * x0[i] + \
                        flow_warp(x0[key_idx][None], self.step_store["flows"][i], mode='nearest')[0] * self.step_store["occ_masks"][i] 
                
                if len(self.step_store["pre_x0"]) < int(self.total_step * self.warp_period[1]):
                    self.step_store['pre_x0'].append(x0[key_idx])
                else:
                    self.step_store['pre_x0'][self.cur_step] = x0[key_idx]

        return x0

    def set_distance(self, B, H, W, radius, device):
        y, x = torch.meshgrid(torch.arange(H), torch.arange(W))
        coords = torch.stack((y, x), dim=-1).float().to(device)
        coords = rearrange(coords, 'h w c -> (h w) c')

        # Calculate the Euclidean distance between all pixels
        distances = torch.cdist(coords, coords)
        radius = 1 if radius == 0 else radius
        # print(f"[INFO]  W: {W} Radius: {radius} ")
        distances //= radius
        distances = torch.exp(-distances)
        distances = repeat(distances, 'h a -> 1 (b h) a', b=B)
        self.distances[H] = distances
    
    def set_flow_correspondence(self, B, H, W, key_idx, flow_confid, flow):

        if len(flow) != B - 1:
                flow_confid = flow_confid[:key_idx] + flow_confid[key_idx+1:]
                flow = flow[:key_idx] + flow[key_idx+1:]

        flow_confid = torch.cat(flow_confid, dim=0) 
        flow = torch.cat(flow, dim=0) 
        flow_confid = rearrange(flow_confid, 'b h w -> 1 (b h w)')
        
        edge_idx = flow_confid.argsort(dim=-1, descending=True)[..., None]

        src_idx = edge_idx[..., :, :]  # Merged Tokens 

        A = H * W
        src_idx_tensor = src_idx[0, : ,0]
        f = src_idx_tensor // A
        id = src_idx_tensor % A
        x = id % W
        y = id // W

        # Stack the results into a 2D tensor
        src_fxy = torch.stack((f, x, y), dim=1)
        
        grid = coords_grid(B-1, H, W).to(flow.device) + flow  # [F-1, 2, H, W]

        x = grid[src_fxy[:, 0], 0, src_fxy[:, 2], src_fxy[:, 1]].clamp(0, W-1).long()
        y = grid[src_fxy[:, 0], 1, src_fxy[:, 2], src_fxy[:, 1]].clamp(0, H-1).long()
        tar_xy = torch.stack((x, y), dim=1)
        tar_idx = y * W + x
        tar_idx = rearrange(tar_idx, ' d -> 1 d 1')

        self.flow_correspondence[H] = (src_idx, tar_idx)

    def set_warp(self, flows, masks, flow_confids=None):
        self.step_store["flows"] = flows
        self.step_store["occ_masks"] = masks
        if flow_confids is not None:
            self.step_store["flow_confids"] = flow_confids

    def set_pre_keyframe_lq(self, pre_keyframe_lq):
        self.step_store["pre_keyframe_lq"] = pre_keyframe_lq

    def __call__(self, context, is_cross: bool, place_in_unet: str):
        context = self.forward(context, is_cross, place_in_unet)
        return context

    def set_cur_frame_idx(self, frame_idx):
        self.cur_frame_idx = frame_idx

    def set_step(self, step):
        self.cur_step = step

    def set_total_step(self, total_step):
        self.total_step = total_step
        self.cur_index = 0

    def clear_store(self):
        del self.step_store
        torch.cuda.empty_cache()
        gc.collect()
        self.step_store = self.get_empty_store()

    def set_task(self, task, restore_step=1.0):
        self.init_store = False
        self.restore = False
        self.update = False
        self.cur_index = 0
        self.restore_step = restore_step
        self.updatex0 = False
        self.restorex0 = False
        if 'initfirst' in task:
            self.init_store = True
            self.clear_store()
        if 'updatestyle' in task:
            self.update = True
        if 'keepstyle' in task:
            self.restore = True
        if 'updatex0' in task: 
            self.updatex0 = True
        if 'keepx0' in task:
            self.restorex0 = True
