import os
import random
import argparse
import numpy as np
# Copyright Alibaba Inc. All Rights Reserved.
from PIL import Image, ImageOps

import torch
from diffusers import CogVideoXDPMScheduler

import insightface
from insightface.app import FaceAnalysis
from facexlib.parsing import init_parsing_model
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from diffusers.training_utils import free_memory
from diffusers.utils import export_to_video, load_image, load_video

from models.utils import process_face_embeddings
from models.transformer_id import IDTransformer3DModel
from models.pipeline_id import IDPipeline
from models.eva_clip import create_model_and_transforms
from models.eva_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from models.eva_clip.utils_qformer import resize_numpy_image_long

import open3d as o3d
import cv2
def get_random_seed():
    return random.randint(0, 2**32 - 1)

def generate_video(
    prompt: str,
    model_path: str,
    transformer_dir: str,
    pcd_path: str,
    output_path: str = "./output/",
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    num_videos_per_prompt: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 42,
    img_file_path: str = None,
):
    """
    Generates a video based on the given prompt and saves it to the specified path.

    Parameters:
    - prompt (str): The description of the video to be generated.
    - model_path (str): The path of the pre-trained model to be used.
    - output_path (str): The path where the generated video will be saved.
    - num_inference_steps (int): Number of steps for the inference process. More steps can result in better quality.
    - guidance_scale (float): The scale for classifier-free guidance. Higher values can lead to better alignment with the prompt.
    - num_videos_per_prompt (int): Number of videos to generate per prompt.
    - dtype (torch.dtype): The data type for computation (default is torch.bfloat16).
    - seed (int): The seed for reproducibility.
    """
    device = "cuda"
    # 0. load main models
    if not os.path.exists(output_path): 
        os.makedirs(output_path, exist_ok=True)
    
    if os.path.exists(os.path.join(model_path, "transformer_ema")):
        subfolder = "transformer_ema"
    else:
        subfolder = "transformer"
        
    transformer = IDTransformer3DModel.from_pretrained_cus(transformer_dir, subfolder=subfolder)
    scheduler = CogVideoXDPMScheduler.from_pretrained(model_path, subfolder="scheduler")

    try:
        is_kps = transformer.config.is_kps
    except:
        is_kps = False
        
    # 1. load face helper models
    face_helper = FaceRestoreHelper(
        upscale_factor=1,
        face_size=512,
        crop_ratio=(1, 1),
        det_model='retinaface_resnet50',
        save_ext='png',
        device=device,
        model_rootpath=os.path.join(model_path, "face_encoder")
    )
    face_helper.face_parse = None
    face_helper.face_parse = init_parsing_model(model_name='bisenet', device=device, model_rootpath=os.path.join(model_path, "face_encoder"))
    face_helper.face_det.eval()
    face_helper.face_parse.eval()

    model, _, _ = create_model_and_transforms('EVA02-CLIP-L-14-336', os.path.join(model_path, "face_encoder", "EVA02_CLIP_L_336_psz14_s6B.pt"), force_custom_clip=True)
    face_clip_model = model.visual
    face_clip_model.eval()

    eva_transform_mean = getattr(face_clip_model, 'image_mean', OPENAI_DATASET_MEAN)
    eva_transform_std = getattr(face_clip_model, 'image_std', OPENAI_DATASET_STD)
    if not isinstance(eva_transform_mean, (list, tuple)):
        eva_transform_mean = (eva_transform_mean,) * 3
    if not isinstance(eva_transform_std, (list, tuple)):
        eva_transform_std = (eva_transform_std,) * 3
    eva_transform_mean = eva_transform_mean
    eva_transform_std = eva_transform_std

    face_main_model = FaceAnalysis(name='antelopev2', root=os.path.join(model_path, "face_encoder"), providers=['CUDAExecutionProvider'])
    handler_ante = insightface.model_zoo.get_model(f'{model_path}/face_encoder/models/antelopev2/glintr100.onnx', providers=['CUDAExecutionProvider'])
    face_main_model.prepare(ctx_id=0, det_size=(640, 640))
    handler_ante.prepare(ctx_id=0)
        
    face_clip_model.to(device, dtype=dtype)
    face_helper.face_det.to(device)
    face_helper.face_parse.to(device)
    transformer.to(device, dtype=dtype)
    free_memory()
    
    pipe = IDPipeline.from_pretrained(model_path, transformer=transformer, scheduler=scheduler, torch_dtype=dtype)

    # 2. Set Scheduler.
    scheduler_args = {}
    if "variance_type" in pipe.scheduler.config:
        variance_type = pipe.scheduler.config.variance_type
        if variance_type in ["learned", "learned_range"]:
            variance_type = "fixed_small"
        scheduler_args["variance_type"] = variance_type

    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, **scheduler_args)

    # 3. Enable CPU offload for the model.
    pipe.to(device)

    # turn on if you don't have multiple GPUs or enough GPU memory(such as H100) and it will cost more time in inference, it may also reduce the quality
    # pipe.enable_model_cpu_offload()
    # pipe.enable_sequential_cpu_offload()
    # pipe.vae.enable_slicing()
    # pipe.vae.enable_tiling()

    # process face data
    id_image = np.array(load_image(image=img_file_path).convert("RGB"))
    id_image = resize_numpy_image_long(id_image, 1024)
    face_helper.clean_all()
    face_helper.read_image(cv2.cvtColor(id_image,cv2.COLOR_RGB2BGR))
    face_helper.align_warp_face()    
    face_helper.get_face_landmarks_5(only_center_face=True)
    face_helper.align_warp_face()
    id_image = face_helper.cropped_faces[0] 
    cv2.imwrite("output/id_image_infer.png", id_image)
    id_cond, id_vit_hidden, align_crop_face_image, face_kps = process_face_embeddings(face_helper, face_clip_model, handler_ante, 
                                                                            eva_transform_mean, eva_transform_std, 
                                                                            face_main_model, device, dtype, id_image, 
                                                                            original_id_image=id_image, is_align_face=True, 
                                                                            cal_uncond=False)
    
    if is_kps:
        kps_cond = face_kps
    else:
        kps_cond = None

    tensor = align_crop_face_image.cpu().detach()
    tensor = tensor.squeeze()
    tensor = tensor.permute(1, 2, 0)
    tensor = tensor.numpy() * 255
    tensor = tensor.astype(np.uint8)
    image = ImageOps.exif_transpose(Image.fromarray(tensor))

    prompt = prompt.strip('"')
    
    generator = torch.Generator(device).manual_seed(seed) if seed else None
    
    point_cloud = o3d.io.read_point_cloud(pcd_path)

    vertices = np.asarray(point_cloud.points)
    vertices = torch.tensor(vertices)
    vertices = vertices.to(pipe.device,dtype=torch.float).unsqueeze(0)
    id_image = cv2.cvtColor(id_image, cv2.COLOR_BGR2RGB)
    video_generate = pipe(
        prompt=prompt,
        image=Image.fromarray(id_image),
        num_videos_per_prompt=num_videos_per_prompt,
        num_inference_steps=num_inference_steps,
        num_frames=49,
        use_dynamic_cfg=False,
        guidance_scale=guidance_scale,
        generator=generator,
        id_vit_hidden=id_vit_hidden,
        id_cond=id_cond,
        kps_cond=kps_cond,
        extra_face=vertices
    ).frames[0]

    # 5. Export the generated frames to a video file. fps must be 8 for original video.
    file_count = len([f for f in os.listdir(output_path) if os.path.isfile(os.path.join(output_path, f))])
    filename = f"{output_path}/{seed}_{file_count:04d}.mp4"
    export_to_video(video_generate, filename, fps=8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video from a text prompt using ID")
    
    # ckpt arguments
    parser.add_argument("--model_path", type=str, default="Fantasy-ID", help="The path of the pre-trained model to be used")
    parser.add_argument("--transformer_dir", type=str, default="Fantasy-ID", help="The path of the pre-trained model to be used")
    parser.add_argument("--pcd_path", type=str, default="assets/lyf.ply", help="The path of the pcd file")

    
    # input arguments
    parser.add_argument("--img_file_path", type=str, default="assets/anne.png")
    parser.add_argument("--prompt", type=str, default="A man is walking")
    
    # output arguments
    parser.add_argument("--output_path", type=str, default="./output", help="The path where the generated video will be saved")
    
    # generation arguments
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="The scale for classifier-free guidance")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of steps for the inference process")
    parser.add_argument("--num_videos_per_prompt", type=int, default=1, help="Number of videos to generate per prompt")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="The data type for computation (e.g., 'float16' or 'bfloat16')")
    parser.add_argument("--seed", type=int, default=42, help="The seed for reproducibility")
    
    args = parser.parse_args()

    generate_video(
        prompt=args.prompt,
        model_path=args.model_path,
        transformer_dir=args.transformer_dir,
        pcd_path=args.pcd_path,
        output_path=args.output_path,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        num_videos_per_prompt=args.num_videos_per_prompt,
        dtype=torch.float16 if args.dtype == "float16" else torch.bfloat16,
        seed=args.seed,
        img_file_path=args.img_file_path
    )
