import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import torchvision
import openmesh as om
from sklearn.neighbors import KDTree
import numpy as np
from torch_scatter import scatter_add
import math
from timm.layers.mlp import Mlp


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Transformer implementation based on ViT
# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
BN_MOMENTUM = 0.1


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

from diffusers.models.attention import Attention

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., selfatt=True):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, z=None):
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # (B, N, inner_dim)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale   # (B, Nq, Nk)

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., selfatt=True):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head,
                                       dropout=dropout, selfatt=selfatt)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, z=None):
        for attn, ff in self.layers:
            x = attn(x, z=z) + x
            x = ff(x) + x
        return x


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Image Feature Backbone
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


"""
Linear Projector
"""

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


"""
CNN backbone
"""


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class StemNet(nn.Module):
    def __init__(self, emb_dim=128):
        super().__init__()
        self.inplanes = 64
        self.patch_size = [4, 4]
        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(Bottleneck, 64, 4)

        channels = 256
        patch_dim = channels * self.patch_size[0] * self.patch_size[1]
        self.patch_to_embedding = nn.Linear(patch_dim, emb_dim)

        self.pe_h = 16
        self.pe_w = 16
        
        self.pos_embedding = nn.Parameter(
            self._make_sine_position_embedding(emb_dim),
            requires_grad=False)
        

    def _make_sine_position_embedding(self, d_model, temperature=10000,
                                      scale=2 * math.pi):
        h, w = self.pe_h, self.pe_w

        area = torch.ones(1, h, w)  # [b, h, w]
        y_embed = area.cumsum(1, dtype=torch.float32)
        x_embed = area.cumsum(2, dtype=torch.float32)

        one_direction_feats = d_model // 2

        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

        dim_t = torch.arange(one_direction_feats, dtype=torch.float32)
        dim_t = temperature ** (2 * (dim_t // 2) / one_direction_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = pos.flatten(2).permute(0, 2, 1)
        print("==> Add Sine PositionEmbedding~")
        return pos



    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, img):
        # x, (B, 3, 256, 256)
        
        p = self.patch_size

        x = self.conv1(img)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p[0], p2 = p[1])

        x = self.patch_to_embedding(x)

        x = x + self.pos_embedding
        return x

#%%
class SpiralConv(nn.Module):
    def __init__(self, in_channels, out_channels, indices, dim=1):
        super(SpiralConv, self).__init__()
        self.dim = dim
        self.indices = indices
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.seq_length = indices.size(1)

        self.layer = nn.Linear(in_channels * self.seq_length, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.layer.weight)
        torch.nn.init.constant_(self.layer.bias, 0)

    def forward(self, x):
        # (B, V, 3)
        n_nodes, _ = self.indices.size()
        if x.dim() == 2:
            x = torch.index_select(x, 0, self.indices.view(-1).to(x.device))
            x = x.view(n_nodes, -1)
        elif x.dim() == 3:
            bs = x.size(0)
            x = torch.index_select(x, self.dim, self.indices.view(-1).to(x.device))
            x = x.view(bs, n_nodes, -1)
        else:
            raise RuntimeError(
                'x.dim() is expected to be 2 or 3, but received {}'.format(
                    x.dim()))
        x = self.layer(x)
        return x

    def __repr__(self):
        return '{}({}, {}, seq_length={})'.format(self.__class__.__name__,
                                                  self.in_channels,
                                                  self.out_channels,
                                                  self.seq_length)


def Pool(x, trans, dim=1):
    row, col = trans._indices()
    value = trans._values().unsqueeze(-1)
    out = torch.index_select(x, dim, col) * value
    out = scatter_add(out, row, dim, dim_size=trans.size(0))
    return out


class SpiralEnblock(nn.Module):
    def __init__(self, in_channels, out_channels, indices):
        super(SpiralEnblock, self).__init__()
        self.conv = SpiralConv(in_channels, out_channels, indices)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()
    def forward(self, x, down_transform):
        out = F.elu(self.conv(x))
        out = Pool(out, down_transform)
        return out


class SpiralDeblock(nn.Module):
    def __init__(self, in_channels, out_channels, indices):
        super(SpiralDeblock, self).__init__()
        self.conv = SpiralConv(in_channels, out_channels, indices)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, up_transform):
        out = Pool(x, up_transform)
        out = F.elu(self.conv(out))
        return out

class SpiralDecoder(nn.Module):
    def __init__(self, emb_dim, spiral_indices, up_transform):
        super(SpiralDecoder, self).__init__()

        self.spiral_indices = spiral_indices
        self.up_transform = up_transform 

        num_layer = len(up_transform)

        # decoder
        self.de_layers = nn.ModuleList()
        for idx in range(num_layer):
            self.de_layers.append(
                SpiralDeblock(emb_dim, emb_dim, self.spiral_indices[num_layer - idx - 1]))
        
        self.de_layers.append(
            SpiralConv(emb_dim, emb_dim, self.spiral_indices[0]))
    
    def forward(self, x):
        num_layers = len(self.de_layers) 
        num_features = num_layers - 1 

        for i, layer in enumerate(self.de_layers):
            if i != num_layers - 1:
                x = layer(x, self.up_transform[num_features - i - 1].to(x.device))
            else:
                x = layer(x)

        return x

def to_sparse(spmat):
    return torch.sparse.FloatTensor(
        torch.LongTensor([spmat.tocoo().row,
                          spmat.tocoo().col]),
        torch.FloatTensor(spmat.tocoo().data), torch.Size(spmat.tocoo().shape))

def _next_ring(mesh, last_ring, other):
    res = []

    def is_new_vertex(idx):
        return (idx not in last_ring and idx not in other and idx not in res)

    for vh1 in last_ring:
        vh1 = om.VertexHandle(vh1)
        after_last_ring = False
        for vh2 in mesh.vv(vh1):
            if after_last_ring:
                if is_new_vertex(vh2.idx()):
                    res.append(vh2.idx())
            if vh2.idx() in last_ring:
                after_last_ring = True
        for vh2 in mesh.vv(vh1):
            if vh2.idx() in last_ring:
                break
            if is_new_vertex(vh2.idx()):
                res.append(vh2.idx())
    return res

def preprocess_spiral(face, seq_length, vertices=None, dilation=1):
    assert face.shape[1] == 3
    if vertices is not None:
        mesh = om.TriMesh(np.array(vertices), np.array(face))
    else:
        n_vertices = face.max() + 1     # V
        mesh = om.TriMesh(np.ones([n_vertices, 3]), np.array(face))
    spirals = torch.tensor(
        extract_spirals(mesh, seq_length=seq_length, dilation=dilation))
    return spirals

def extract_spirals(mesh, seq_length, dilation=1):
    spirals = []
    for vh0 in mesh.vertices():
        reference_one_ring = []
        for vh1 in mesh.vv(vh0):
            reference_one_ring.append(vh1.idx())
        spiral = [vh0.idx()]
        one_ring = list(reference_one_ring)
        last_ring = one_ring
        next_ring = _next_ring(mesh, last_ring, spiral)
        spiral.extend(last_ring)
        while len(spiral) + len(next_ring) < seq_length * dilation:
            if len(next_ring) == 0:
                break
            last_ring = next_ring
            next_ring = _next_ring(mesh, last_ring, spiral)
            spiral.extend(last_ring)
        if len(next_ring) > 0:
            spiral.extend(next_ring)
        else:
            kdt = KDTree(mesh.points(), metric='euclidean')
            spiral = kdt.query(np.expand_dims(mesh.points()[spiral[0]],
                                              axis=0),
                               k=seq_length * dilation,
                               return_distance=False).tolist()
            spiral = [item for subspiral in spiral for item in subspiral]
        spirals.append(spiral[:seq_length * dilation][::dilation])
    return spirals

def get_coarse_mesh_decoder(emb_dim=32, transform_fp = "./assets/transform.pkl", down_degree=2):
    dilation = [1, 1, 1, 1]
    seq_length = [9, 9, 9, 9]
    with open(transform_fp, 'rb') as f:
        tmp = pickle.load(f, encoding='latin1')

    spiral_indices_list = [
        preprocess_spiral(tmp['face'][idx], seq_length[idx],tmp['vertices'][idx], dilation[idx])
        for idx in range(len(tmp['face']) - 1)
    ]
    down_transform_list = [to_sparse(down_transform) for down_transform in tmp['down_transform'] ]
    up_transform_list = [to_sparse(up_transform) for up_transform in tmp['up_transform']]

    spiral_indices_list = spiral_indices_list[: down_degree]
    down_transform_list = down_transform_list[ : down_degree]
    up_transform_list = up_transform_list[: down_degree]
    
    mesh_decoder = SpiralDecoder(emb_dim=emb_dim, spiral_indices=spiral_indices_list, up_transform=up_transform_list)
    return mesh_decoder, down_transform_list

def downsample_vertices(vertices, down_transform_list):
    coarse_vertices = vertices.clone()
    for transform in down_transform_list:
        coarse_vertices = Pool(coarse_vertices, transform.to(vertices.device)) 
    return coarse_vertices

class VertexSpiralNet(nn.Module):
    def __init__(self, EMB_DIM=128, num_v_coarse=314, transform_fp = "./assets/transform.pkl"):
        super(VertexSpiralNet, self).__init__()

        self.EMB_DIM=EMB_DIM
        self.transform_fp = transform_fp
        self.num_v_coarse = num_v_coarse
        self.vtx_query = nn.Parameter(torch.zeros(1, num_v_coarse, EMB_DIM))       # (1, Vc=314, 128)
        vtx_upsampler, down_transform_list = get_coarse_mesh_decoder(EMB_DIM, transform_fp)
        self.vtx_upsampler = vtx_upsampler
        self.down_transform_list = down_transform_list
        self.transformer = Transformer(dim=EMB_DIM, depth=6, heads=4, dim_head=EMB_DIM // 4, mlp_dim=EMB_DIM * 2)
        self.vtx_descriptor_head = nn.Sequential( 
            nn.Linear(EMB_DIM, EMB_DIM),
            # nn.Linear(EMB_DIM, 1024),
            # nn.Linear(1024,2048)
        )
        self.hw_down = nn.ModuleList([
            Mlp(in_features=2048, out_features=1024),
            Mlp(in_features=1024, out_features=512),
            Mlp(in_features=512, out_features=256),
            Mlp(in_features=256, out_features=128),
        ])
    def depth_sin_pos_encoding(self, depth, temperature=10000, scale = 2 * math.pi):
        depth = depth * scale 
        dim_t = torch.arange(self.EMB_DIM, dtype=torch.float32).to(depth.device)
        dim_t = temperature ** (2 * (dim_t // 2) / self.EMB_DIM)
        pos_depth  = depth / dim_t
        pos_depth = torch.stack([pos_depth[:, :, 0::2].sin(), pos_depth[:, :, 1::2].cos() ], dim=3).flatten(2)  
        
        return pos_depth

    def forward(self, vertices, hw=None):
        B = vertices.shape[0]
        vtx_query = repeat(self.vtx_query, '() n d -> b n d', b = B)    
        coarse_vertices = downsample_vertices(vertices, self.down_transform_list)     
        depth_emb = self.depth_sin_pos_encoding(0.5 * coarse_vertices[:, :, 2:] + 0.5)  

        vtx_query = vtx_query + depth_emb
        tokens = vtx_query
        if hw is not None:
            for block in self.hw_down:
                hw = block(hw)
            tokens = torch.cat([vtx_query, hw], dim=1)

        tokens = self.transformer(tokens)                    
        # feature head 
        vtx_feat = self.vtx_upsampler(tokens[:, :self.num_v_coarse]) 
        vtx_descriptor = self.vtx_descriptor_head(vtx_feat)       
        return vtx_descriptor

from typing import Optional, Tuple, Union
ACTIVATION_FUNCTIONS = {
    "swish": nn.SiLU(),
    "silu": nn.SiLU(),
    "mish": nn.Mish(),
    "gelu": nn.GELU(),
    "relu": nn.ReLU(),
}


def get_activation(act_fn: str) -> nn.Module:
    """Helper function to get activation function from string.

    Args:
        act_fn (str): Name of activation function.

    Returns:
        nn.Module: Activation function.
    """

    act_fn = act_fn.lower()
    if act_fn in ACTIVATION_FUNCTIONS:
        return ACTIVATION_FUNCTIONS[act_fn]
    else:
        raise ValueError(f"Unsupported activation function: {act_fn}")

def rearrange_dims(tensor: torch.Tensor) -> torch.Tensor:
    if len(tensor.shape) == 2:
        return tensor[:, :, None]
    if len(tensor.shape) == 3:
        return tensor[:, :, None, :]
    elif len(tensor.shape) == 4:
        return tensor[:, :, 0, :]
    else:
        raise ValueError(f"`len(tensor)`: {len(tensor)} has to be 2, 3 or 4.")
        
class Conv1dBlock(nn.Module):
    """
    Conv1d --> GroupNorm --> Mish

    Parameters:
        inp_channels (`int`): Number of input channels.
        out_channels (`int`): Number of output channels.
        kernel_size (`int` or `tuple`): Size of the convolving kernel.
        n_groups (`int`, default `8`): Number of groups to separate the channels into.
        activation (`str`, defaults to `mish`): Name of the activation function.
    """

    def __init__(
        self,
        inp_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: int = 1,
        padding: int = 0,
        n_groups: int = 8,
        activation: str = "mish",
    ):
        super().__init__()

        self.conv1d = nn.Conv1d(inp_channels, out_channels, kernel_size, stride=stride, padding=padding)
        # self.conv1d = nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.group_norm = nn.GroupNorm(n_groups, out_channels)
        self.mish = get_activation(activation)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        intermediate_repr = self.conv1d(inputs)
        intermediate_repr = rearrange_dims(intermediate_repr)
        intermediate_repr = self.group_norm(intermediate_repr)
        intermediate_repr = rearrange_dims(intermediate_repr)
        output = self.mish(intermediate_repr)
        return output

class ResidualBlock1D(nn.Module):
    """
    Residual 1D block with temporal convolutions that reduces token amount through stride.

    Parameters:
        inp_channels (`int`): Number of input channels.
        out_channels (`int`): Number of output channels.
        kernel_size (`int` or `tuple`): Size of the convolving kernel.
        stride (`int`): Stride of the convolution.
    """

    def __init__(
        self,
        inp_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 5,
        stride: int = 1,
        padding: int = 0
    ):
        super().__init__()
        # Apply stride to reduce time dimension
        self.conv_in = Conv1dBlock(inp_channels, out_channels, kernel_size, stride=stride, padding=padding)
        # Assuming stride=1 for the second conv layer to maintain reduced size
        self.conv_out = Conv1dBlock(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        # Residual connection needs to match stride if dimensions differ
        self.residual_conv = (
            nn.Conv1d(inp_channels, out_channels, kernel_size, stride=stride, padding=padding)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs : [ batch_size x inp_channels x horizon ]

        Returns:
            out : [ batch_size x out_channels x reduced_horizon ]
        """
        out = self.conv_in(inputs)
        out = self.conv_out(out)
        return out + self.residual_conv(inputs)

class Conv1DReducer4(nn.Module):
    def __init__(self):
        super(Conv1DReducer4, self).__init__()
        self.conv_block1 = ResidualBlock1D(128, 256, kernel_size=5, stride=4, padding=2)
        self.conv_block2 = ResidualBlock1D(256, 512, kernel_size=5, stride=4, padding=2)
        self.conv_block3 = ResidualBlock1D(512, 1024, kernel_size=3, stride=2, padding=1)
        self.conv_block4 = ResidualBlock1D(1024, 2048, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        return x

