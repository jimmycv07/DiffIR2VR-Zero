from typing import overload, Tuple, Optional

import os
import cv2
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T

import numpy as np
from glob import glob
from PIL import Image
from einops import rearrange

from model.cldm import ControlLDM
from model.gaussian_diffusion import Diffusion
from model.bsrnet import RRDBNet
from model.swinir import SwinIR
from model.scunet import SCUNet
from utils.sampler import SpacedSampler
from utils.cond_fn import Guidance
from utils.video_visualizer import VideoVisualizer
from utils.common import wavelet_decomposition, wavelet_reconstruction, count_vram_usage

from utils import vidtome
from GMFlow.gmflow.gmflow import GMFlow
from utils.flow_utils import get_warped_and_mask

def save_video(input_folder, out_path, output_name, fps=25):
    video_visualizer = VideoVisualizer(path=os.path.join(out_path, output_name),
                                    frame_size=None,
                                    fps=fps)
    input_folder = os.path.join(out_path, input_folder)
    imgs = sorted([filename for filename in os.listdir(input_folder) if filename.endswith(('.png', '.jpg'))], key=lambda x: int(x.split('.')[0]))
    for img in imgs:
        img_pth = os.path.join(input_folder, img)
        image = cv2.imread(img_pth)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        video_visualizer.add(image)
    video_visualizer.save()

def batch_bicubic_resize(img: np.ndarray, scale: float) -> np.ndarray:
    
    if scale != 1:
        for i in range(img.shape[0]):
            img[i] = bicubic_resize(img[i], scale)
    # pil = Image.fromarray(img)
    # res = pil.resize(tuple(int(x * scale) for x in pil.size), Image.BICUBIC)
    return img

def bicubic_resize(img: np.ndarray, scale: float) -> np.ndarray:
    
    if scale != 1:
        pil = Image.fromarray(img)
        res = pil.resize(tuple(int(x * scale) for x in pil.size), Image.BICUBIC)
    return np.array(res)


def resize_short_edge_to(imgs: torch.Tensor, size: int) -> torch.Tensor:
    _, _, h, w = imgs.size()
    if h == w:
        new_h, new_w = size, size
    elif h < w:
        new_h, new_w = size, int(w * (size / h))
    else:
        new_h, new_w = int(h * (size / w)), size
    return F.interpolate(imgs, size=(new_h, new_w), mode="bicubic", antialias=True)


def pad_to_multiples_of(imgs: torch.Tensor, multiple: int) -> torch.Tensor:
    _, _, h, w = imgs.size()
    if h % multiple == 0 and w % multiple == 0:
        return imgs.clone()
    # get_pad = lambda x: (x // multiple + 1) * multiple - x
    get_pad = lambda x: (x // multiple + int(x % multiple != 0)) * multiple - x
    ph, pw = get_pad(h), get_pad(w)
    return F.pad(imgs, pad=(0, pw, 0, ph), mode="constant", value=0)


class Pipeline:

    def __init__(self, stage1_model: nn.Module, cldm: ControlLDM, diffusion: Diffusion, cond_fn: Optional[Guidance], device: str) -> None:
        self.stage1_model = stage1_model
        self.cldm = cldm
        self.diffusion = diffusion
        self.cond_fn = cond_fn
        self.device = device
        self.final_size: Tuple[int] = None

    def set_final_size(self, lq: torch.Tensor) -> None:
        h, w = lq.shape[2:]
        self.final_size = (h, w)

    @overload
    def run_stage1(self, lq: torch.Tensor) -> torch.Tensor:
        ...

    @count_vram_usage
    def run_stage2(
        self,
        clean: torch.Tensor,
        steps: int,
        strength: float,
        tiled: bool,
        tile_size: int,
        tile_stride: int,
        pos_prompt: str,
        neg_prompt: str,
        cfg_scale: float,
        better_start: float,
        index: int = 0,
        input: str = None
    ) -> torch.Tensor:
        ### preprocess
        bs, _, ori_h, ori_w = clean.shape
        # pad: ensure that height & width are multiples of 64
        pad_clean = pad_to_multiples_of(clean, multiple=64)
        h, w = pad_clean.shape[2:]
        if self.cldm.controller is not None:
            self.cldm.controller.cldm = self.cldm
            self.cldm.controller.non_pad_ratio = (ori_h / h, ori_w / w)
            self.cldm.vae.decoder.controller = self.cldm.controller
        # prepare conditon
        if not tiled:
            cond = self.cldm.prepare_condition(pad_clean, [pos_prompt] * bs)
            uncond = self.cldm.prepare_condition(pad_clean, [neg_prompt] * bs)
        else:
            cond = self.cldm.prepare_condition_tiled(pad_clean, [pos_prompt] * bs, tile_size, tile_stride)
            uncond = self.cldm.prepare_condition_tiled(pad_clean, [neg_prompt] * bs, tile_size, tile_stride)
        if self.cond_fn:
            self.cond_fn.load_target(pad_clean * 2 - 1)
        old_control_scales = self.cldm.control_scales
        self.cldm.control_scales = [strength] * 13
        if better_start:
            # using noised low frequency part of condition as a better start point of 
            # reverse sampling, which can prevent our model from generating noise in 
            # image background.
            _, low_freq = wavelet_decomposition(pad_clean)
            # low_freq = pad_clean
            if not tiled:
                if ori_w > 1500:
                    x_0 = self.cldm.vae_encode(low_freq, batch_size=2)
                else:
                    x_0 = self.cldm.vae_encode(low_freq, batch_size=5)
                # x_0 = self.cldm.vae_encode(low_freq)
            else:
                x_0 = self.cldm.vae_encode_tiled(low_freq, tile_size, tile_stride)
            x_T = self.diffusion.q_sample(
                x_0,
                torch.full((bs, ), self.diffusion.num_timesteps - 1, dtype=torch.long, device=self.device),
                torch.randn(x_0.shape, dtype=torch.float32, device=self.device)
            )
            # print(f"diffusion sqrt_alphas_cumprod: {self.diffusion.sqrt_alphas_cumprod[-1]}")
        else:
            x_T = torch.randn((bs, 4, h // 8, w // 8), dtype=torch.float32, device=self.device)

        ### run sampler
        sampler = SpacedSampler(self.diffusion.betas)
        z = sampler.sample(
            model=self.cldm, device=self.device, steps=steps, batch_size=bs, x_size=(4, h // 8, w // 8),
            cond=cond, uncond=uncond, cfg_scale=cfg_scale, x_T=x_T, progress=True,
            progress_leave=True, cond_fn=self.cond_fn, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride,
            non_pad_ratio=(ori_h / h, ori_w / w)
        )
        if not tiled:
            if ori_w > 1500:
                x = self.cldm.vae_decode(z, batch_size=2)
            else:
                x = self.cldm.vae_decode(z, batch_size=5)
        else:
            x = self.cldm.vae_decode_tiled(z, tile_size // 8, tile_stride // 8)
        ### postprocess
        self.cldm.control_scales = old_control_scales
        sample = x[:, :, :ori_h, :ori_w]
        return sample

    @torch.no_grad()
    def run(
        self,
        lq: np.ndarray,
        steps: int,
        strength: float,
        tiled: bool,
        tile_size: int,
        tile_stride: int,
        pos_prompt: str,
        neg_prompt: str,
        cfg_scale: float,
        better_start: bool,
        index: int = 0,
        input: str = None,
        final_size: Tuple[int] = None,
        flow_model: GMFlow = None,
    ) -> np.ndarray:
        # image to tensor
        lq = torch.tensor((lq / 255.).clip(0, 1), dtype=torch.float32, device=self.device)
        lq = rearrange(lq, "n h w c -> n c h w").contiguous()
        # set pipeline output size
        if final_size is None:
            self.set_final_size(lq)
        else:
            self.final_size = final_size
        
        clean = self.run_stage1(lq)
        # print(f"[INFO] {clean.shape}")
        
        
        ''' flow & occlusion mask '''
        pre_keyframe_lq = None

        if self.cldm.controller is not None and \
            self.cldm.controller.step_store["pre_keyframe_lq"] is not None:
            
            pre_keyframe_lq = self.cldm.controller.step_store["pre_keyframe_lq"]
            pre_keyframe_lq = torch.tensor((pre_keyframe_lq / 255.).clip(0, 1), dtype=torch.float32, device=self.device)
            pre_keyframe_lq = rearrange(pre_keyframe_lq, "n h w c -> n c h w").contiguous()
            pre_keyframe_clean = self.run_stage1(pre_keyframe_lq)[0]

        flows, masks, confids = [], [], []
        flows2, confids2 = [], []
        key_idx = lq.shape[0] // 2 
        for k in range(lq.shape[0]):
            if k == key_idx:
                if pre_keyframe_lq is not None:
                    tar_img = (torch.clamp(clean[key_idx], 0 ,1) * 255).float().to(self.device)
                    src_img = (torch.clamp(pre_keyframe_clean, 0 ,1) * 255).float().to(self.device)
                else:
                    flows.append(None)
                    masks.append(None)
                    confids.append(None)
                    continue
            else:
                tar_img = (torch.clamp(clean[k], 0 ,1) * 255).float().to(self.device)
                src_img = (torch.clamp(clean[key_idx], 0 ,1) * 255).float().to(self.device)
            # tar_img = stage1_x[0].float().to(args.device)
            _, bwd_occ, bwd_flow, bwd_confid = get_warped_and_mask(
                flow_model, src_img, tar_img, image3=None, pixel_consistency=False, return_confidence=True)
            blend_mask = T.GaussianBlur(kernel_size=(9, 9), sigma=(18, 18))(
                F.max_pool2d(bwd_occ, kernel_size=9, stride=1, padding=4))
            blend_mask = torch.clamp(blend_mask + bwd_occ, 0, 1)
            blend_mask = 1 - F.max_pool2d(blend_mask, kernel_size=8)
            
            blend_mask = 1 - F.max_pool2d(bwd_occ, kernel_size=8)

            bwd_confid2 = F.max_pool2d(bwd_confid, kernel_size=16)
            bwd_flow2 = F.interpolate(bwd_flow / 16.0, scale_factor=1. / 16, mode='bilinear')


            bwd_confid = F.max_pool2d(bwd_confid, kernel_size=8)
            bwd_flow = F.interpolate(bwd_flow / 8.0, scale_factor=1. / 8, mode='bilinear')

            blend_mask = pad_to_multiples_of(blend_mask[None], 8)[0]
            flows.append(bwd_flow)
            masks.append(blend_mask)
            confids.append(bwd_confid)

            flows2.append(bwd_flow2)
            confids2.append(bwd_confid2)
            
        if self.cldm.controller is not None:
            self.cldm.controller.set_warp(flows, masks, flow_confids=confids)
            
            _, H, W = confids[0].shape
            self.cldm.controller.set_flow_correspondence(lq.shape[0], H, W, key_idx, confids, flows)
            _, H, W = confids2[0].shape
            self.cldm.controller.set_flow_correspondence(lq.shape[0], H, W, key_idx, confids2, flows2)
            for j, flow in enumerate(self.cldm.controller.step_store["flows"]): 
                if flow is not None:
                    self.cldm.controller.step_store["flows"][j] = pad_to_multiples_of(self.cldm.controller.step_store["flows"][j], 8)
        ''' flow & occlusion mask ended '''


        sample = self.run_stage2(
            clean, steps, strength, tiled, tile_size, tile_stride,
            pos_prompt, neg_prompt, cfg_scale, better_start,
            index=index, input=input
        )

        # colorfix (borrowed from StableSR, thanks for their work)
        sample = (sample + 1) / 2
        sample = wavelet_reconstruction(sample, clean)
        # resize to desired output size
        sample = F.interpolate(sample, size=self.final_size, mode="bicubic", antialias=True)
        clean = F.interpolate(clean, size=self.final_size, mode="bilinear", antialias=True)
        # tensor to image
        sample = rearrange(sample * 255., "n c h w -> n h w c")
        sample = sample.contiguous().clamp(0, 255).to(torch.uint8).cpu().numpy()
        clean = rearrange(clean * 255., "n c h w -> n h w c")
        clean = clean.contiguous().clamp(0, 255).to(torch.uint8).cpu().numpy()
        return sample, clean


class BSRNetPipeline(Pipeline):

    def __init__(self, bsrnet: RRDBNet, cldm: ControlLDM, diffusion: Diffusion, cond_fn: Optional[Guidance], device: str, upscale: float) -> None:
        super().__init__(bsrnet, cldm, diffusion, cond_fn, device)
        self.upscale = upscale

    def set_final_size(self, lq: torch.Tensor) -> None:
        h, w = lq.shape[2:]
        self.final_size = (int(h * self.upscale), int(w * self.upscale))

    @count_vram_usage
    def run_stage1(self, lq: torch.Tensor) -> torch.Tensor:
        # NOTE: upscale is always set to 4 in our experiments
        if lq.shape[-2] > 1000:
                clean = []
                for i in range(lq.shape[0]):
                    torch.cuda.empty_cache()
                    clean.append(self.stage1_model(lq[i:i+1]))
                clean = torch.cat(clean, dim=0)
        else:
            clean = self.stage1_model(lq)
        # if self.final_size[0] < 512 and self.final_size[1] < 512:
        if min(self.final_size) < 512:
            clean = resize_short_edge_to(clean, size=512)
        else:
            clean = F.interpolate(clean, size=self.final_size, mode="bicubic", antialias=True)
        return clean


class SwinIRPipeline(Pipeline):

    def __init__(self, swinir: SwinIR, cldm: ControlLDM, diffusion: Diffusion, cond_fn: Optional[Guidance], device: str) -> None:
        super().__init__(swinir, cldm, diffusion, cond_fn, device)

    @count_vram_usage
    def run_stage1(self, lq: torch.Tensor) -> torch.Tensor:
        # NOTE: lq size is always equal to 512 in our experiments
        # resize: ensure the input lq size is as least 512, since SwinIR is trained on 512 resolution
        if min(lq.shape[2:]) < 512:
            lq = resize_short_edge_to(lq, size=512)
        ori_h, ori_w = lq.shape[2:]
        # pad: ensure that height & width are multiples of 64
        pad_lq = pad_to_multiples_of(lq, multiple=64)
        # run
        clean = self.stage1_model(pad_lq)
        # remove padding
        clean = clean[:, :, :ori_h, :ori_w]
        return clean


class SCUNetPipeline(Pipeline):

    def __init__(self, scunet: SCUNet, cldm: ControlLDM, diffusion: Diffusion, cond_fn: Optional[Guidance], device: str) -> None:
        super().__init__(scunet, cldm, diffusion, cond_fn, device)

    @count_vram_usage
    def run_stage1(self, lq: torch.Tensor) -> torch.Tensor:
        if lq.shape[-1] > 1500:
                clean = []
                for i in range(lq.shape[0]):
                    torch.cuda.empty_cache()
                    clean.append(self.stage1_model(lq[i:i+1]))
                clean = torch.cat(clean, dim=0)
        else:
            clean = self.stage1_model(lq)
        if min(clean.shape[2:]) < 512:
            clean = resize_short_edge_to(clean, size=512)
        
        return clean