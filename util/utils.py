import os
import math
import tqdm
import logging
import argparse
import itertools
import PIL.Image
import numpy as np
from PIL import Image
import safetensors.torch
from datetime import datetime
from typing import Union, List
from spandrel import ModelLoader

import torch
import torch.nn.functional as F
from diffusers.utils import export_to_video

logger = logging.getLogger(__file__)
def get_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script for ID.")

    # ID information
    parser.add_argument("--train_type", choices=['t2v', 'i2v'], help="t2v or i2v")
    parser.add_argument("--is_train_face", action='store_true')
    parser.add_argument("--is_diff_lr", action='store_true')
    parser.add_argument("--is_train_lora", action='store_true')
    parser.add_argument("--is_kps", action='store_true')
    parser.add_argument("--is_shuffle_data", action='store_true')
    parser.add_argument("--enable_mask_loss", action='store_true')
    parser.add_argument("--is_single_face", action='store_true')
    parser.add_argument("--is_cross_face", action='store_true')
    parser.add_argument("--is_align_face", action='store_true')
    parser.add_argument("--is_reserve_face", action='store_true')
    parser.add_argument("--is_accelerator_state_dict", action='store_true')
    parser.add_argument("--is_validation", action='store_true')
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--mask_path", type=str, default=None)
    parser.add_argument("--pretrained_weight", type=str, default=None)
    parser.add_argument("--sample_stride", type=int, default=3, help=".")
    parser.add_argument("--skip_frames_start_percent", type=float, default=0.0, help=".")
    parser.add_argument("--skip_frames_end_percent", type=float, default=1.0, help=".")
    parser.add_argument("--miss_tolerance", type=int, default=6)
    parser.add_argument("--min_distance", type=int, default=3)
    parser.add_argument("--min_frames", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=5)
    parser.add_argument("--LFE_num_tokens", type=int, default=32)
    parser.add_argument("--LFE_output_dim", type=int, default=768)
    parser.add_argument("--LFE_heads", type=int, default=12)
    parser.add_argument("--cross_attn_interval", type=int, default=1)

    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )

    # Model information
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )

    # Dataset information
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) containing the training data of instance images (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--instance_data_root",
        type=str,
        default=None,
        help=("A folder containing the training data."),
    )
    parser.add_argument(
        "--video_column",
        type=str,
        default="video",
        help="The column of the dataset containing videos. Or, the name of the file in `--instance_data_root` folder containing the line-separated path to video data.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing the instance prompt for each video. Or, the name of the file in `--instance_data_root` folder containing the line-separated instance prompts.",
    )
    parser.add_argument(
        "--id_token", type=str, default=None, help="Identifier token appended to the start of each prompt if provided."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )

    # Validation
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="One or more prompt(s) that is used during validation to verify that the model is learning. Multiple validation prompts should be separated by the '--validation_prompt_seperator' string.",
    )
    parser.add_argument(
        "--validation_images",
        type=str,
        default=None,
        help="One or more image path(s) that is used during validation to verify that the model is learning. Multiple validation paths should be separated by the '--validation_prompt_seperator' string. These should correspond to the order of the validation prompts.",
    )
    parser.add_argument(
        "--validation_prompt_separator",
        type=str,
        default=":::",
        help="String that separates multiple validation prompts",
    )
    parser.add_argument(
        "--num_validation_videos",
        type=int,
        default=1,
        help="Number of videos that should be generated during validation per `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=50,
        help=(
            "Run validation every X epochs. Validation consists of running the prompt `args.validation_prompt` multiple times: `args.num_validation_videos`."
        ),
    )
    parser.add_argument(
        "--low_vram", action="store_true", help="Whether enable low_vram mode."
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=6,
        help="The guidance scale to use while sampling validation videos.",
    )
    parser.add_argument(
        "--use_dynamic_cfg",
        action="store_true",
        default=False,
        help="Whether or not to use the default cosine dynamic guidance schedule when sampling validation videos.",
    )

    # Training information
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--rank",
        type=int,
        default=128,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=128,
        help=("The scaling factor to scale LoRA weight update. The actual scaling factor is `lora_alpha / rank`"),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="cogvideox-i2v-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="All input videos are resized to this height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=720,
        help="All input videos are resized to this width.",
    )
    parser.add_argument("--fps", type=int, default=8, help="All input videos will be used at this FPS.")
    parser.add_argument(
        "--max_num_frames", type=int, default=49, help="All input videos will be truncated to these many frames."
    )
    parser.add_argument(
        "--skip_frames_start",
        type=int,
        default=0,
        help="Number of frames to skip from the beginning of each input video. Useful if training data contains intro sequences.",
    )
    parser.add_argument(
        "--skip_frames_end",
        type=int,
        default=0,
        help="Number of frames to skip from the end of each input video. Useful if training data contains outro sequences.",
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip videos horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides `--num_train_epochs`.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine_with_restarts",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--enable_slicing",
        action="store_true",
        default=False,
        help="Whether or not to use VAE slicing for saving memory.",
    )
    parser.add_argument(
        "--enable_tiling",
        action="store_true",
        default=False,
        help="Whether or not to use VAE tiling for saving memory.",
    )
    parser.add_argument(
        "--noised_image_dropout",
        type=float,
        default=0.05,
        help="Image condition dropout probability.",
    )

    # Optimizer
    parser.add_argument(
        "--optimizer",
        type=lambda s: s.lower(),
        default="adam",
        choices=["adam", "adamw", "prodigy"],
        help=("The optimizer type to use."),
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )
    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.95, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="Coefficients for computing the Prodigy optimizer's stepsize using running averages. If set to None, uses the value of square root of beta2.",
    )
    parser.add_argument("--prodigy_decouple", action="store_true", help="Use AdamW style decoupled weight decay")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use for unet params")
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--prodigy_use_bias_correction", action="store_true", help="Turn on Adam's bias correction.")
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        action="store_true",
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage.",
    )

    # Other information
    parser.add_argument("--tracker_name", type=str, default=None, help="Project tracker name")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="Directory where logs are stored.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default=None,
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        '--trainable_modules', 
        nargs='+', 
        help='Enter a list of trainable modules'
    )
    parser.add_argument("--nccl_timeout", type=int, default=600, help="NCCL backend timeout in seconds.")

    return parser.parse_args()

def resize_mask(mask, latent, process_first_frame_only=True):
    latent_size = latent.size()

    if process_first_frame_only:
        target_size = list(latent_size[2:])
        target_size[0] = 1
        first_frame_resized = F.interpolate(
            mask[:, :, 0:1, :, :],
            size=target_size,
            mode='trilinear',
            align_corners=False
        )
        
        target_size = list(latent_size[2:])
        target_size[0] = target_size[0] - 1
        if target_size[0] != 0:
            remaining_frames_resized = F.interpolate(
                mask[:, :, 1:, :, :],
                size=target_size,
                mode='trilinear',
                align_corners=False
            )
            resized_mask = torch.cat([first_frame_resized, remaining_frames_resized], dim=2)
        else:
            resized_mask = first_frame_resized
    else:
        target_size = list(latent_size[2:])
        resized_mask = F.interpolate(
            mask,
            size=target_size,
            mode='trilinear',
            align_corners=False
        )
    return resized_mask

def save_tensor_as_image(tensor, file_path):
    """
    Saves a PyTorch tensor as an image file.

    Args:
        tensor (torch.Tensor): The image tensor to save.
        file_path (str): Path to save the image file.
    """
    # Ensure the tensor is in CPU memory and detach it from the computation graph
    tensor = tensor.cpu().detach()
    
    # Convert from PyTorch to NumPy format, and handle the scaling from [0, 1] to [0, 255]
    tensor = tensor.squeeze()  # Remove unnecessary dimensions if any
    tensor = tensor.permute(1, 2, 0)  # Change from (C, H, W) to (H, W, C)
    tensor = tensor.numpy() * 255  # Scale from [0, 1] to [0, 255]
    tensor = tensor.astype(np.uint8)  # Convert to uint8
    
    # Convert the NumPy array to a PIL Image and save it
    image = Image.fromarray(tensor)
    image.save(file_path)

def pixel_values_to_pil(pixel_values, frame_index=0):
    if pixel_values.is_cuda:
        pixel_values = pixel_values.clone().cpu()
    pixel_values = (pixel_values + 1.0) / 2.0 * 255.0
    pixel_values = pixel_values.clamp(0, 255).byte()
    frame = pixel_values[frame_index]  # [C, H, W]
    frame = frame.permute(1, 2, 0)  # [H, W, C]
    frame_np = frame.numpy()
    image = Image.fromarray(frame_np)
    return image

def load_torch_file(ckpt, device=None, dtype=torch.float16):
    if device is None:
        device = torch.device("cpu")
    if ckpt.lower().endswith(".safetensors") or ckpt.lower().endswith(".sft"):
        sd = safetensors.torch.load_file(ckpt, device=device.type)
    else:
        if not "weights_only" in torch.load.__code__.co_varnames:
            logger.warning(
                "Warning torch.load doesn't support weights_only on this pytorch version, loading unsafely."
            )

        pl_sd = torch.load(ckpt, map_location=device, weights_only=True)
        if "global_step" in pl_sd:
            logger.debug(f"Global Step: {pl_sd['global_step']}")
        if "state_dict" in pl_sd:
            sd = pl_sd["state_dict"]
        elif "params_ema" in pl_sd:
            sd = pl_sd["params_ema"]
        else:
            sd = pl_sd

    sd = {k: v.to(dtype) for k, v in sd.items()}
    return sd


def state_dict_prefix_replace(state_dict, replace_prefix, filter_keys=False):
    if filter_keys:
        out = {}
    else:
        out = state_dict
    for rp in replace_prefix:
        replace = list(
            map(
                lambda a: (a, "{}{}".format(replace_prefix[rp], a[len(rp) :])),
                filter(lambda a: a.startswith(rp), state_dict.keys()),
            )
        )
        for x in replace:
            w = state_dict.pop(x[0])
            out[x[1]] = w
    return out


def module_size(module):
    module_mem = 0
    sd = module.state_dict()
    for k in sd:
        t = sd[k]
        module_mem += t.nelement() * t.element_size()
    return module_mem


def get_tiled_scale_steps(width, height, tile_x, tile_y, overlap):
    return math.ceil((height / (tile_y - overlap))) * math.ceil((width / (tile_x - overlap)))


@torch.inference_mode()
def tiled_scale_multidim(
    samples, function, tile=(64, 64), overlap=8, upscale_amount=4, out_channels=3, output_device="cpu", pbar=None
):
    dims = len(tile)
    print(f"samples dtype:{samples.dtype}")
    output = torch.empty(
        [samples.shape[0], out_channels] + list(map(lambda a: round(a * upscale_amount), samples.shape[2:])),
        device=output_device,
    )

    for b in range(samples.shape[0]):
        s = samples[b : b + 1]
        out = torch.zeros(
            [s.shape[0], out_channels] + list(map(lambda a: round(a * upscale_amount), s.shape[2:])),
            device=output_device,
        )
        out_div = torch.zeros(
            [s.shape[0], out_channels] + list(map(lambda a: round(a * upscale_amount), s.shape[2:])),
            device=output_device,
        )

        for it in itertools.product(*map(lambda a: range(0, a[0], a[1] - overlap), zip(s.shape[2:], tile))):
            s_in = s
            upscaled = []

            for d in range(dims):
                pos = max(0, min(s.shape[d + 2] - overlap, it[d]))
                l = min(tile[d], s.shape[d + 2] - pos)
                s_in = s_in.narrow(d + 2, pos, l)
                upscaled.append(round(pos * upscale_amount))

            ps = function(s_in).to(output_device)
            mask = torch.ones_like(ps)
            feather = round(overlap * upscale_amount)
            for t in range(feather):
                for d in range(2, dims + 2):
                    m = mask.narrow(d, t, 1)
                    m *= (1.0 / feather) * (t + 1)
                    m = mask.narrow(d, mask.shape[d] - 1 - t, 1)
                    m *= (1.0 / feather) * (t + 1)

            o = out
            o_d = out_div
            for d in range(dims):
                o = o.narrow(d + 2, upscaled[d], mask.shape[d + 2])
                o_d = o_d.narrow(d + 2, upscaled[d], mask.shape[d + 2])

            o += ps * mask
            o_d += mask

            if pbar is not None:
                pbar.update(1)

        output[b : b + 1] = out / out_div
    return output


def tiled_scale(
    samples,
    function,
    tile_x=64,
    tile_y=64,
    overlap=8,
    upscale_amount=4,
    out_channels=3,
    output_device="cpu",
    pbar=None,
):
    return tiled_scale_multidim(
        samples, function, (tile_y, tile_x), overlap, upscale_amount, out_channels, output_device, pbar
    )


def load_sd_upscale(ckpt, inf_device):
    sd = load_torch_file(ckpt, device=inf_device)
    if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
        sd = state_dict_prefix_replace(sd, {"module.": ""})
    out = ModelLoader().load_from_state_dict(sd).half()
    return out


def upscale(upscale_model, tensor: torch.Tensor, inf_device, output_device="cpu") -> torch.Tensor:
    memory_required = module_size(upscale_model.model)
    memory_required += (
        (512 * 512 * 3) * tensor.element_size() * max(upscale_model.scale, 1.0) * 384.0
    )  # The 384.0 is an estimate of how much some of these models take, TODO: make it more accurate
    memory_required += tensor.nelement() * tensor.element_size()
    print(f"UPScaleMemory required: {memory_required / 1024 / 1024 / 1024} GB")

    upscale_model.to(inf_device)
    tile = 512
    overlap = 32

    steps = tensor.shape[0] * get_tiled_scale_steps(
        tensor.shape[3], tensor.shape[2], tile_x=tile, tile_y=tile, overlap=overlap
    )

    pbar = ProgressBar(steps, desc="Tiling and Upscaling")

    s = tiled_scale(
        samples=tensor.to(torch.float16),
        function=lambda a: upscale_model(a),
        tile_x=tile,
        tile_y=tile,
        overlap=overlap,
        upscale_amount=upscale_model.scale,
        pbar=pbar,
    )

    upscale_model.to(output_device)
    return s


def upscale_batch_and_concatenate(upscale_model, latents, inf_device, output_device="cpu") -> torch.Tensor:
    upscaled_latents = []
    for i in range(latents.size(0)):
        latent = latents[i]
        upscaled_latent = upscale(upscale_model, latent, inf_device, output_device)
        upscaled_latents.append(upscaled_latent)
    return torch.stack(upscaled_latents)


def save_video(tensor: Union[List[np.ndarray], List[PIL.Image.Image]], fps: int = 8):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = f"./output/{timestamp}.mp4"
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    export_to_video(tensor, video_path, fps=fps)
    return video_path


class ProgressBar:
    def __init__(self, total, desc=None):
        self.total = total
        self.current = 0
        self.b_unit = tqdm.tqdm(total=total, desc="ProgressBar context index: 0" if desc is None else desc)

    def update(self, value):
        if value > self.total:
            value = self.total
        self.current = value
        if self.b_unit is not None:
            self.b_unit.set_description("ProgressBar context index: {}".format(self.current))
            self.b_unit.refresh()

            self.b_unit.update(self.current)