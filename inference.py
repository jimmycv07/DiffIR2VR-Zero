from argparse import ArgumentParser, Namespace, ArgumentTypeError

import ast
import torch

from accelerate.utils import set_seed
# from utils.inference import (
#     V1InferenceLoop,
#     BSRInferenceLoop, BFRInferenceLoop, BIDInferenceLoop, UnAlignedBFRInferenceLoop
# )
from utils.batch_inference import (
    V1InferenceLoop,
    BSRInferenceLoop, BFRInferenceLoop, BIDInferenceLoop, UnAlignedBFRInferenceLoop
)


def check_device(device: str) -> str:
    if device[:4] == "cuda":
        if not torch.cuda.is_available():
            print("CUDA not available because the current PyTorch install was not "
                  "built with CUDA enabled.")
            device = "cpu"
    else:
        if device == "mps":
            if not torch.backends.mps.is_available():
                if not torch.backends.mps.is_built():
                    print("MPS not available because the current PyTorch install was not "
                          "built with MPS enabled.")
                    device = "cpu"
                else:
                    print("MPS not available because the current MacOS version is not 12.3+ "
                          "and/or you do not have an MPS-enabled device on this machine.")
                    device = "cpu"
    print(f"using device {device}")
    return device

def tuple_type(s):
    try:
        value = ast.literal_eval(s)
        if isinstance(value, tuple):
            return value
    except ValueError:
        pass
    raise ArgumentTypeError(f"{s} is not a valid tuple")

def parse_args() -> Namespace:
    parser = ArgumentParser()
    ### model parameters
    parser.add_argument("--task", type=str, required=True, choices=["sr", "dn", "fr", "fr_bg"])
    parser.add_argument("--upscale", type=float, required=True)
    parser.add_argument("--version", type=str, default="v2", choices=["v1", "v2"])
    ### sampling parameters
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--better_start", action="store_true")
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument("--tile_stride", type=int, default=256)
    parser.add_argument("--pos_prompt", type=str, default="")
    parser.add_argument("--neg_prompt", type=str, default="low quality, blurry, low-resolution, noisy, unsharp, weird textures")
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    ### input parameters
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--final_size", type=tuple_type, default=(480, 854))
    parser.add_argument("--config", type=str, default="/home/yehhh/DiffBIR/configs/inference/my_cldm.yaml")
    ### guidance parameters
    parser.add_argument("--guidance", action="store_true")
    parser.add_argument("--g_loss", type=str, default="w_mse", choices=["mse", "w_mse"])
    parser.add_argument("--g_scale", type=float, default=0.0)
    parser.add_argument("--g_start", type=int, default=1001)
    parser.add_argument("--g_stop", type=int, default=-1)
    parser.add_argument("--g_space", type=str, default="latent")
    parser.add_argument("--g_repeat", type=int, default=1)
    ### output parameters
    parser.add_argument("--output", type=str, required=True)
    ### common parameters
    parser.add_argument("--seed", type=int, default=231)
    parser.add_argument("--device", type=str, default="cuda")
    # parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda", "mps", "cuda:0", "cuda:1"])

    ### saving parameters
    parser.add_argument("--save_img_only", action="store_true")

    ### config parameters
    parser.add_argument("--warp_period", type=tuple_type, default=None)
    parser.add_argument("--ToMe_period", type=tuple_type, default=None)
    parser.add_argument("--merge_ratio", type=tuple_type, default=None)
    
    return parser.parse_args()


def main():
    args = parse_args()
    args.device = check_device(args.device)
    set_seed(args.seed)
    print(f"[INFO] {args.task} {args.input} -> {args.output}")
    if args.version == "v1":
        V1InferenceLoop(args).run()
    else:
        supported_tasks = {
            "sr": BSRInferenceLoop,
            "dn": BIDInferenceLoop,
            "fr": BFRInferenceLoop,
            "fr_bg": UnAlignedBFRInferenceLoop
        }
        supported_tasks[args.task](args).run()
        print("done!")
    

if __name__ == "__main__":
    main()
