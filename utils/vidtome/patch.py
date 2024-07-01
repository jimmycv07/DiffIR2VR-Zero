import os 
import math
import time
from typing import Type, Dict, Any, Tuple, Callable

import numpy as np
from einops import rearrange, repeat
import torch
import torch.nn.functional as F

from . import merge
from .utils import isinstance_str, init_generator, join_frame, split_frame, func_warper, join_warper, split_warper


def compute_merge(module: torch.nn.Module, x: torch.Tensor, tome_info: Dict[str, Any]) -> Tuple[Callable, ...]:
    H, original_w = tome_info["size"]
    downsample = tome_info["args"]["downsample"]

    args = tome_info["args"]
    # generator = module.generator

    # Frame Number and Token Number
    fsize = x.shape[0] // args["batch_size"]
    tsize = x.shape[1]
    # Merge tokens in high resolution layers
    # print(f"[INFO] {args['current_step']} downsample {downsample} time")

    label = args["label"].split('_')
    

    if downsample <= args["max_downsample"] and downsample > args["min_downsample"]:
        # print(f"[INFO] downsample: {args['min_downsample']} < {downsample} <= {args['max_downsample']} token shape: {x.shape} H: {H}")
        if args["generator"] is None:
            args["generator"] = init_generator(x.device)
            # module.generator = module.generator.manual_seed(123)
        elif args["generator"].device != x.device:
            args["generator"] = init_generator(x.device, fallback=args["generator"])

        # Local Token Merging!

        local_tokens = join_frame(x, fsize)
        m_ls = [join_warper(fsize)]
        u_ls = [split_warper(fsize)]
        unm = 0
        curF = fsize
        
        # Recursive merge multi-frame tokens into one set. Such as 4->1 for 4 frames and 8->2->1 for 8 frames when target stride is 4.
        while curF > 1:

            controller, total_step = args["controller"], args["controller"].total_step

            if controller is not None and label[0] == "unet" and label[1] == "down":
                # print(f"[INFO] flow merge @ {label[0]} {label[1]} {downsample}")
                m, u, ret_dict = merge.bipartite_soft_matching_randframe(
                    local_tokens, curF, args["local_merge_ratio"], unm, generator=args["generator"],
                    target_stride=x.shape[0],
                    H=H,
                    flow_merge=True, 
                    controller=controller,
                    )
            else:
                m, u, ret_dict = merge.bipartite_soft_matching_randframe(
                    local_tokens, curF, args["local_merge_ratio"], unm, generator=args["generator"],
                    target_stride=x.shape[0],
                    H=H,
                    flow_merge=False, 
                    controller=controller,
                    )
            
            unm += ret_dict["unm_num"]
            m_ls.append(m)
            u_ls.append(u)
            local_tokens = m(local_tokens)

            # assert (x.shape[1] - unm) % tsize == 0
            # Total token number = current frame number * per-frame token number + unmerged token number
            curF = (local_tokens.shape[1] - unm) // tsize

        merged_tokens = local_tokens
        
        # Global Token Merging!
        if args["merge_global"]:
            if hasattr(module, "global_tokens") and module.global_tokens is not None:
                # Merge local tokens with global tokens. Randomly determine merging destination.
                if torch.rand(1, generator=args["generator"], device=args["generator"].device) > args["global_rand"]:
                    src_len = local_tokens.shape[1]
                    tokens = torch.cat(
                        [local_tokens, module.global_tokens.to(local_tokens)], dim=1)
                    local_chunk = 0
                else:
                    src_len = module.global_tokens.shape[1]
                    tokens = torch.cat(
                        [module.global_tokens.to(local_tokens), local_tokens], dim=1)
                    local_chunk = 1
                m, u, _ = merge.bipartite_soft_matching_2s(
                    tokens, src_len, args["global_merge_ratio"], unmerge_chunk=local_chunk)
                merged_tokens = m(tokens)
                # print(f"[INFO] global merging {local_tokens.shape} {tokens.shape} {merged_tokens.shape}")
                
                m_ls.append(m)
                u_ls.append(u)

                # Update global tokens with unmerged local tokens. There should be a better way to do this.
                module.global_tokens = u(merged_tokens).detach().clone().cpu()
            else:
                module.global_tokens = local_tokens.detach().clone().cpu()

        m = func_warper(m_ls)
        u = func_warper(u_ls[::-1])
    else:
        m, u = (merge.do_nothing, merge.do_nothing)
        merged_tokens = x

    # Return merge op, unmerge op, and merged tokens.
    return m, u, merged_tokens

def make_tome_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    """
    Make a patched class on the fly so we don't have to import any specific modules.
    This patch applies ToMe to the forward function of the block.
    """

    class ToMeBlock(block_class):
        # Save for unpatching later
        _parent = block_class

        def _forward(self, x: torch.Tensor, context: torch.Tensor = None, label: str = None) -> torch.Tensor:
            # print(f"[INFO] ~~~ ToMeblock ~~~ {label} ~~~")

            B, A, C = x.shape
            original_h, original_w = self._tome_info["size"]
            original_tokens = original_h * original_w
            downsample = int(math.ceil(math.sqrt(original_tokens // A)))
            # print(f"[INFO] downsample {downsample} time A {A} original_h {original_h} original_w {original_w}")
            self._tome_info["args"]["downsample"] = downsample
            H, W = original_h // downsample, original_w // downsample
            
            non_pad_ratio_h, non_pad_ratio_w = self._tome_info["args"]["controller"].non_pad_ratio

            padding_size_w = W - int(W * non_pad_ratio_w)
            padding_size_h = H - int(H * non_pad_ratio_h)
            padding_mask = torch.zeros((H, W), device=x.device, dtype=torch.bool)
            if padding_size_w:
                padding_mask[:, -padding_size_w:] = 1
            if padding_size_h:
                padding_mask[-padding_size_h:, :] = 1
            padding_mask = rearrange(padding_mask, 'h w -> (h w)')
            
            idx_buffer = torch.arange(A, device=x.device, dtype=torch.int64)
            non_pad_idx = idx_buffer[None, ~padding_mask, None]
            # pad_idx = idx_buffer[None, padding_mask, None]
            del idx_buffer, padding_mask
            x_non_pad = torch.gather(x, dim=1, index=non_pad_idx.expand(B, -1, C))
            self._tome_info["args"]["label"] = label
            self._tome_info["size"] = (int(H * non_pad_ratio_h), int(W * non_pad_ratio_w))
            # print(f"[INFO] original shape {x.shape} removed padding shape {x_non_pad.shape}")
            m_a, u_a, merged_tokens = compute_merge(
                self, self.norm1(x_non_pad), self._tome_info)
    
            x_non_pad = u_a(self.attn1(m_a(self.norm1(x_non_pad)),
                    context=context if self.disable_self_attn else None)) + x_non_pad
            x_non_pad = self.attn2(self.norm2(x_non_pad), context=context) + x_non_pad
            x_non_pad = self.ff(self.norm3(x_non_pad)) + x_non_pad
            x.scatter_(dim=1, index=non_pad_idx.expand(B, -1, C), src=x_non_pad)
            del x_non_pad
            self._tome_info["size"] = (original_h, original_w)
            torch.cuda.empty_cache()

            return x

    return ToMeBlock


def make_diffusers_tome_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    """
    Make a patched class for a diffusers model.
    This patch applies ToMe to the forward function of the block.
    """
    class ToMeBlock(block_class):
        # Save for unpatching later
        _parent = block_class

        def forward(
            self,
            hidden_states,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            timestep=None,
            cross_attention_kwargs=None,
            class_labels=None,
        ) -> torch.Tensor:

            if self.use_ada_layer_norm:
                norm_hidden_states = self.norm1(hidden_states, timestep)
            elif self.use_ada_layer_norm_zero:
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                    hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
                )
            else:
                norm_hidden_states = self.norm1(hidden_states)

            # Merge input tokens
                
            m_a, u_a, merged_tokens = compute_merge(
                self, norm_hidden_states, self._tome_info)
            norm_hidden_states = merged_tokens

            # 1. Self-Attention
            cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
            # tt = time.time()
            attn_output = self.attn1(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )
            # print(time.time() - tt)
            if self.use_ada_layer_norm_zero:
                attn_output = gate_msa.unsqueeze(1) * attn_output

            # Unmerge output tokens
            attn_output = u_a(attn_output)
            hidden_states = attn_output + hidden_states

            if self.attn2 is not None:
                norm_hidden_states = (
                    self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(
                        hidden_states)
                )

                # 2. Cross-Attention
                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    **cross_attention_kwargs,
                )

                hidden_states = attn_output + hidden_states

            # 3. Feed-forward
            norm_hidden_states = self.norm3(hidden_states)

            if self.use_ada_layer_norm_zero:
                norm_hidden_states = norm_hidden_states * \
                    (1 + scale_mlp[:, None]) + shift_mlp[:, None]

            ff_output = self.ff(norm_hidden_states)

            if self.use_ada_layer_norm_zero:
                ff_output = gate_mlp.unsqueeze(1) * ff_output
            
            hidden_states = ff_output + hidden_states
            
            return hidden_states

    return ToMeBlock


def hook_tome_model(model: torch.nn.Module):
    """ Adds a forward pre hook to get the image size. This hook can be removed with remove_patch. """
    def hook(module, args):
        # print(args[0].shape)
        module._tome_info["size"] = (args[0].shape[2], args[0].shape[3])
        return None
    model._tome_info["hooks"].append(model.register_forward_pre_hook(hook))


def hook_tome_module(module: torch.nn.Module):
    """ Adds a forward pre hook to initialize random number generator.
        All modules share the same generator state to keep their randomness in VidToMe consistent in one pass.
        This hook can be removed with remove_patch. """
    def hook(module, args):
        
        if not hasattr(module, "generator"):
            module.generator = init_generator(args[0].device)
        elif module.generator.device != args[0].device:
            module.generator = init_generator(
                args[0].device, fallback=module.generator)
        else:
            return None

        # module.generator = module.generator.manual_seed(module._tome_info["args"]["seed"])
        return None

    module._tome_info["hooks"].append(module.register_forward_pre_hook(hook))


def apply_patch(
        model: torch.nn.Module,
        local_merge_ratio: float = 0.9,
        merge_global: bool = False,
        global_merge_ratio = 0.8,
        max_downsample: int = 2,
        min_downsample: int = 0,
        seed: int = 123,
        batch_size: int = 2,
        include_control: bool = False,
        target_stride: int = 4,
        global_rand=0.5):
    """
    Patches a stable diffusion model with VidToMe.
    Apply this to the highest level stable diffusion object (i.e., it should have a .model.diffusion_model).

    Important Args:
     - model: A top level Stable Diffusion module to patch in place. Should have a ".model.diffusion_model"
     - local_merge_ratio: The ratio of tokens to merge locally. I.e., 0.9 would merge 90% src tokens.
              If there are 4 frames in a chunk (3 src, 1 dst), the compression ratio will be 1.3 / 4.0.
              And the largest compression ratio is 0.25 (when local_merge_ratio = 1.0).
              Higher values result in more consistency, but with more visual quality loss.
     - merge_global: Whether or not to include global token merging.
     - global_merge_ratio: The ratio of tokens to merge locally. I.e., 0.8 would merge 80% src tokens.
                           When find significant degradation in video quality. Try to lower the value.

    Args to tinker with if you want:
     - max_downsample [1, 2, 4, or 8]: Apply VidToMe to layers with at most this amount of downsampling.
                                       E.g., 1 only applies to layers with no downsampling (4/15) while
                                       8 applies to all layers (15/15). I recommend a value of 1 or 2.
     - seed: Manual random seed. 
     - batch_size: Video batch size. Number of video chunks in one pass. When processing one video, it 
                   should be 2 (cond + uncond) or 3 (when using PnP, source + cond + uncond).
     - include_control: Whether or not to patch ControlNet model.
     - target_stride: Stride between target frames. I.e., when target_stride = 4, there is 1 target frame
                      in any 4 consecutive frames. 
     - global_rand: Probability in global token merging src/dst split. Global tokens are always src when
                    global_rand = 1.0 and always dst when global_rand = 0.0 .
    """

    # Make sure the module is not currently patched
    remove_patch(model)

    is_diffusers = isinstance_str(
        model, "DiffusionPipeline") or isinstance_str(model, "ModelMixin")
    
    if not is_diffusers:
        if (not hasattr(model, "model") or not hasattr(model.model, "diffusion_model")) \
            and not hasattr(model, "unet"):
            # Provided model not supported
            raise RuntimeError(
                "Provided model was not a Stable Diffusion / Latent Diffusion model, as expected.")
        else:
            diffusion_model = model.unet if hasattr(model, "unet") else model.model.diffusion_model
    else:
        # Supports "pipe.unet" and "unet"
        diffusion_model = model.unet if hasattr(model, "unet") else model

    if isinstance_str(model, "StableDiffusionControlNetPipeline") and include_control:
        diffusion_models = [diffusion_model, model.controlnet]
    else:
        diffusion_models = [diffusion_model]

    if not is_diffusers and hasattr(model, "controlnet"):
        diffusion_models = [diffusion_model, model.controlnet]

    for diffusion_model in diffusion_models:
        diffusion_model._tome_info = {
            "size": None,
            "hooks": [],
            "args": {
                "max_downsample": max_downsample,
                "min_downsample": min_downsample,
                "generator": None,
                "seed": seed,
                "batch_size": batch_size,
                "merge_global": merge_global,
                "global_merge_ratio": global_merge_ratio,
                "local_merge_ratio": local_merge_ratio,
                "global_rand": global_rand,
                "target_stride": target_stride,
                "label": "",
                "downsample": 1,
                "controller": None,
            }
        }
        hook_tome_model(diffusion_model)

        for name, module in diffusion_model.named_modules():
            # If for some reason this has a different name, create an issue and I'll fix it
            # if isinstance_str(module, "BasicTransformerBlock") and "down_blocks" not in name:
            if isinstance_str(module, "BasicTransformerBlock"):
                make_tome_block_fn = make_diffusers_tome_block if is_diffusers else make_tome_block
                module.__class__ = make_tome_block_fn(module.__class__)
                module._tome_info = diffusion_model._tome_info
                hook_tome_module(module)

                # Something introduced in SD 2.0 (LDM only)
                if not hasattr(module, "disable_self_attn") and not is_diffusers:
                    module.disable_self_attn = False

                # Something needed for older versions of diffusers
                if not hasattr(module, "use_ada_layer_norm_zero") and is_diffusers:
                    module.use_ada_layer_norm = False
                    module.use_ada_layer_norm_zero = False
        
    return model


def remove_patch(model: torch.nn.Module):
    """ Removes a patch from a ToMe Diffusion module if it was already patched. """
    # For diffusers
    modelu = model.unet if hasattr(model, "unet") else model
    model_ls = [modelu]
    if hasattr(model, "controlnet"):
        model_ls.append(model.controlnet)
    for model in model_ls:
        for _, module in model.named_modules():
            if hasattr(module, "_tome_info"):
                for hook in module._tome_info["hooks"]:
                    hook.remove()
                module._tome_info["hooks"].clear()

            if module.__class__.__name__ == "ToMeBlock":
                module.__class__ = module._parent

    return model


def update_patch(model: torch.nn.Module, **kwargs):
    """ Update arguments in patched modules """
    # For diffusers
    model0 = model.unet if hasattr(model, "unet") else model
    model_ls = [model0]
    if hasattr(model, "controlnet"):
        model_ls.append(model.controlnet)
    for model in model_ls:
        for _, module in model.named_modules():
            if hasattr(module, "_tome_info"):
                for k, v in kwargs.items():
                    # setattr(module, k, v)
                    if k in module._tome_info["args"]:
                        module._tome_info["args"][k] = v
    return model


def collect_from_patch(model: torch.nn.Module, attr="tome"):
    """ Collect attributes in patched modules """
    # For diffusers
    model0 = model.unet if hasattr(model, "unet") else model
    model_ls = [model0]
    if hasattr(model, "controlnet"):
        model_ls.append(model.controlnet)
    ret_dict = dict()
    for model in model_ls:
        for name, module in model.named_modules():
            if hasattr(module, attr):
                res = getattr(module, attr)
                ret_dict[name] = res

    return ret_dict
