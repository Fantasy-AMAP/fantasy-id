# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms
# in the LICENSE file included with this software distribution.
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

import os
import pickle
import sys
from time import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from skimage.io import imread

from .models.decoders import Generator

# from .utils.renderer import SRenderY, set_rasterizer
from .models.encoders import ResnetEncoder
from .models.FLAME import FLAME
from .utils import util

# from .datasets import datasets
from .utils.config import cfg
from .utils.rotation_converter import batch_euler2axis
from .utils.tensor_cropper import transform_points

torch.backends.cudnn.benchmark = True


class DECA(nn.Module):
    def __init__(self, config=None, device="cuda"):
        super(DECA, self).__init__()
        if config is None:
            self.cfg = cfg
        else:
            self.cfg = config
        self.device = device
        self.image_size = self.cfg.dataset.image_size
        self.uv_size = self.cfg.model.uv_size

        self._create_model(self.cfg.model)

    def _create_model(self, model_cfg):
        # set up parameters
        self.n_param = (
            model_cfg.n_shape
            + model_cfg.n_tex
            + model_cfg.n_exp
            + model_cfg.n_pose
            + model_cfg.n_cam
            + model_cfg.n_light
        )
        self.n_detail = model_cfg.n_detail
        self.n_cond = model_cfg.n_exp + 3  # exp + jaw pose
        self.num_list = [
            model_cfg.n_shape,
            model_cfg.n_tex,
            model_cfg.n_exp,
            model_cfg.n_pose,
            model_cfg.n_cam,
            model_cfg.n_light,
        ]
        self.param_dict = {i: model_cfg.get("n_" + i) for i in model_cfg.param_list}

        # encoders
        self.E_flame = ResnetEncoder(outsize=self.n_param).to(self.device)

        # decoders
        self.flame = FLAME(model_cfg)

        # resume model
        model_path = self.cfg.pretrained_modelpath
        if os.path.exists(model_path):
            print(f"trained model found. load {model_path}")
            checkpoint = torch.load(model_path, map_location="cpu")
            self.checkpoint = checkpoint
            util.copy_state_dict(self.E_flame.state_dict(), checkpoint["E_flame"])
        else:
            print(f"please check model path: {model_path}")
        # eval mode
        self.E_flame.eval()

    def decompose_code(self, code, num_dict):
        """Convert a flattened parameter vector to a dictionary of parameters
        code_dict.keys() = ['shape', 'tex', 'exp', 'pose', 'cam', 'light']
        """
        code_dict = {}
        start = 0
        for key in num_dict:
            end = start + int(num_dict[key])
            code_dict[key] = code[:, start:end]
            start = end
            if key == "light":
                code_dict[key] = code_dict[key].reshape(code_dict[key].shape[0], 9, 3)
        return code_dict

    def displacement2normal(self, uv_z, coarse_verts, coarse_normals):
        """Convert displacement map into detail normal map"""
        batch_size = uv_z.shape[0]
        uv_coarse_vertices = self.render.world2uv(coarse_verts).detach()
        uv_coarse_normals = self.render.world2uv(coarse_normals).detach()

        uv_z = uv_z * self.uv_face_eye_mask
        uv_detail_vertices = (
            uv_coarse_vertices
            + uv_z * uv_coarse_normals
            + self.fixed_uv_dis[None, None, :, :] * uv_coarse_normals.detach()
        )
        dense_vertices = uv_detail_vertices.permute(0, 2, 3, 1).reshape(
            [batch_size, -1, 3]
        )
        uv_detail_normals = util.vertex_normals(
            dense_vertices, self.render.dense_faces.expand(batch_size, -1, -1)
        )
        uv_detail_normals = uv_detail_normals.reshape(
            [batch_size, uv_coarse_vertices.shape[2], uv_coarse_vertices.shape[3], 3]
        ).permute(0, 3, 1, 2)
        uv_detail_normals = (
            uv_detail_normals * self.uv_face_eye_mask
            + uv_coarse_normals * (1.0 - self.uv_face_eye_mask)
        )
        return uv_detail_normals

    def visofp(self, normals):
        """visibility of keypoints, based on the normal direction"""
        normals68 = self.flame.seletec_3d68(normals)
        vis68 = (normals68[:, :, 2:] < 0.1).float()
        return vis68

    # @torch.no_grad()
    def encode(self, images, use_detail=True):
        parameters = self.E_flame(images)
        codedict = self.decompose_code(parameters, self.param_dict)

        codedict["images"] = images

        if self.cfg.model.jaw_type == "euler":
            posecode = codedict["pose"]
            euler_jaw_pose = posecode[
                :, 3:
            ].clone()  # x for yaw (open mouth), y for pitch (left ang right), z for roll
            posecode[:, 3:] = batch_euler2axis(euler_jaw_pose)
            codedict["pose"] = posecode
            codedict["euler_jaw_pose"] = euler_jaw_pose
        return codedict

    # @torch.no_grad()
    def decode(
        self,
        codedict,
        rendering=False,
        iddict=None,
        vis_lmk=True,
        return_vis=True,
        use_detail=True,
        render_orig=False,
        original_image=None,
        tform=None,
    ):
        images = codedict["images"]  # (B, 3, 224, 224)
        batch_size = images.shape[0]  # (B, 3, H, W)

        # >>>>>>>>>>>>>>> Flame forward: weights -> verts >>>>>>>>>>>>>>>
        verts, landmarks2d, landmarks3d = self.flame(
            shape_params=codedict["shape"],
            expression_params=codedict["exp"],
            pose_params=codedict["pose"],
        )

        landmarks3d_world = landmarks3d.clone()

        # >>>>>>>>>>>>>>> project lmk >>>>>>>>>>>>>>>
        landmarks2d = util.batch_orth_proj(landmarks2d, codedict["cam"])[
            :, :, :2
        ]  # (B, 68, 2)
        landmarks2d[:, :, 1:] = -landmarks2d[:, :, 1:]

        landmarks3d = util.batch_orth_proj(landmarks3d, codedict["cam"])
        landmarks3d[:, :, 1:] = -landmarks3d[:, :, 1:]

        trans_verts = util.batch_orth_proj(verts, codedict["cam"])
        trans_verts[:, :, 1:] = -trans_verts[
            :, :, 1:
        ]  # !!! Rotate Y  [X, Y, Z] -> s * (X+tx, -(Y+ty), -Z)
        # the negative sign help make the rotation element of by default of 0, rather than pi, for the frontal face

        opdict = {
            "verts": verts,  # (B, 5023, 3), world space
            "trans_verts": trans_verts,  # (B, 5023, 3)  s * (X+tx, Y+ty, Z) in cropped images
            "landmarks2d": landmarks2d,  # (B, 68, 2)
            "landmarks3d": landmarks3d,  # (B, 68, 3)
            "landmarks3d_world": landmarks3d_world,
        }

        return opdict

    def visualize(self, visdict, size=224, dim=2):
        """
        image range should be [0,1]
        dim: 2 for horizontal. 1 for vertical
        """
        assert dim == 1 or dim == 2
        grids = {}
        for key in visdict:
            # 'inputs', 'landmarks2d', 'landmarks3d', 'shape_images', 'shape_detail_images'
            _, _, h, w = visdict[key].shape
            if dim == 2:
                new_h = size
                new_w = int(w * size / h)
            elif dim == 1:
                new_h = int(h * size / w)
                new_w = size
            grids[key] = torchvision.utils.make_grid(
                F.interpolate(visdict[key], [new_h, new_w]).detach().cpu()
            )
        grid = torch.cat(list(grids.values()), dim)
        grid_image = (grid.numpy().transpose(1, 2, 0).copy() * 255)[:, :, [2, 1, 0]]
        grid_image = np.minimum(np.maximum(grid_image, 0), 255).astype(np.uint8)
        return grid_image

    def save_obj(self, filename, opdict):
        """
        vertices: [nv, 3], tensor
        texture: [3, h, w], tensor
        """
        i = 0
        vertices = opdict["verts"][i].cpu().numpy()  # (B, 5023, 3)
        faces = self.render.faces[0].cpu().numpy()  # (B, 9976, 3)
        texture = util.tensor2image(opdict["uv_texture_gt"][i])
        uvcoords = self.render.raw_uvcoords[0].cpu().numpy()
        uvfaces = self.render.uvfaces[0].cpu().numpy()
        # save coarse mesh, with texture and normal map
        normal_map = util.tensor2image(opdict["uv_detail_normals"][i] * 0.5 + 0.5)
        util.write_obj(
            filename,
            vertices,
            faces,
            texture=texture,
            uvcoords=uvcoords,
            uvfaces=uvfaces,
            normal_map=normal_map,
        )

        # upsample mesh, save detailed mesh
        texture = texture[:, :, [2, 1, 0]]
        normals = opdict["normals"][i].cpu().numpy()
        displacement_map = opdict["displacement_map"][i].cpu().numpy().squeeze()
        dense_vertices, dense_colors, dense_faces = util.upsample_mesh(
            vertices, normals, faces, displacement_map, texture, self.dense_template
        )
        # dense_vertices, [B, 59315, 3]
        # dense_faces, [B, 117380, 3]
        util.write_obj(
            filename.replace(".obj", "_detail.obj"),
            dense_vertices,
            dense_faces,
            colors=dense_colors,
            inverse_face_order=True,
        )

    def run(self, imagepath, iscrop=True):
        """An api for running deca given an image path"""
        testdata = datasets.TestData(imagepath)
        images = testdata[0]["image"].to(self.device)[None, ...]
        codedict = self.encode(images)
        opdict, visdict = self.decode(codedict)
        return codedict, opdict, visdict

    def model_dict(self):
        return {"E_flame": self.E_flame.state_dict()}
