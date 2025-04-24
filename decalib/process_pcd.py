import os
import argparse
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
import open3d as o3d
import pandas as pd

from decalib.deca import DECA
from decalib.utils.config import cfg as deca_cfg

def parse_args():
    parser = argparse.ArgumentParser(description='Process face image to point cloud')
    
    parser.add_argument(
        '--input_path',
        type=str,
        required=True,
        help='Path to input image file'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        required=True,
        help='Path to output point cloud file'
    )
    
    return parser.parse_args()

def process_face_to_point_cloud(image_path, output_pcd_path, deca):
    # Define transformation pipeline for input images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Adjust image size
        transforms.ToTensor(),          # Convert to tensor and scale to [0, 1]
    ])
    
    # Use the transformation to preprocess the input image
    image = Image.open(image_path).convert('RGB')
    crop_src_img = transform(image).to("cuda").unsqueeze(0)

    # Using DECA to encode & decode the image to get vertices
    with torch.no_grad():
        src_codedict = deca.encode(crop_src_img, use_detail=False)
        src_outputs = deca.decode(src_codedict, rendering=False, vis_lmk=False, return_vis=False, use_detail=False)
        src_verts = src_outputs["verts"].detach().cpu().numpy()[0]

    # Create an Open3D PointCloud object and assign the vertices
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(src_verts)

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_pcd_path), exist_ok=True)
    
    # Save the point cloud to a file
    o3d.io.write_point_cloud(output_pcd_path, pcd)
    return output_pcd_path

def main():
    # Parse arguments
    args = parse_args()
    
    # Initialize DECA
    deca = DECA(config=deca_cfg)
    deca = deca.to("cuda")
    deca.eval()
    
    # Process the image
    print(f"Processing {args.input_path}...")
    process_face_to_point_cloud(args.input_path, args.output_path, deca)
    print(f"Saved point cloud to {args.output_path}")

if __name__ == "__main__":
    main()
