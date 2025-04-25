# Copyright Alibaba Inc. All Rights Reserved.

import math

import torch
import torch.nn as nn
from einops import rearrange
from transformers.models.deformable_detr import DeformableDetrConfig

from models.projectors import CAbstractor


def FeedForward(dim, mult=4):
    # FFN
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


def reshape_tensor(x, heads):
    bs, length, width = x.shape
    # (bs, length, width) --> (bs, length, n_heads, dim_per_head)
    x = x.view(bs, length, heads, -1)
    # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
    x = x.transpose(1, 2)
    # (bs, n_heads, length, dim_per_head) --> (bs*n_heads, length, dim_per_head)
    x = x.reshape(bs, heads, length, -1)
    return x


class AvgPoolProjector(nn.Module):
    def __init__(
        self,
        layer_num: int = 2,
        query_num: int = 144,
        mm_hidden_size: int = 1024,
        llm_hidden_size: int = 1024,
    ):
        super().__init__()
        self.layer_num = layer_num
        self.query_num = query_num
        self.mm_hidden_size = mm_hidden_size
        self.llm_hidden_size = llm_hidden_size
        self.build_net()

    def build_net(self):
        hw = int(self.query_num**0.5)
        sampler = nn.AdaptiveAvgPool2d((hw, hw))
        self.sampler = sampler
        modules = [nn.Linear(self.mm_hidden_size, self.llm_hidden_size)]
        for _ in range(1, self.layer_num):
            modules.append(nn.GELU())
            modules.append(nn.Linear(self.llm_hidden_size, self.llm_hidden_size))
        self.mlp_projector = nn.Sequential(*modules)

    def forward(self, visual_feat: torch.Tensor, x) -> torch.Tensor:
        batch_size, seq_len, h_dim = visual_feat.shape  # 576
        hw = int(seq_len**0.5)  # 24
        # torch.Size([64, 1024, 24, 24])
        shaped_visual_feat = rearrange(visual_feat, "b (h w) d -> b d h w", h=hw, w=hw)
        # torch.Size([64, 1024, 12, 12])
        pooled_visual_feat = self.sampler(shaped_visual_feat)
        reshaped_visual_feat = rearrange(
            pooled_visual_feat, "b d h w -> b (h w) d"
        )  # [64, 144, 1024]
        output_feat = self.mlp_projector(reshaped_visual_feat)  # [64, 144, 4096])
        return output_feat


class FaceAbstractor(nn.Module):
    def __init__(
        self,
        dim=1024,
        depth=10,
        dim_head=64,
        heads=16,
        num_id_token=5,
        num_queries=144,
        output_dim=2048,
        ff_mult=4,
    ):
        """
        Initializes the FaceAbstractor class.

        Parameters:
        - dim (int): The dimensionality of latent features.
        - depth (int): Total number of CAbstractor and FeedForward layers.
        - dim_head (int): Dimensionality of each attention head.
        - heads (int): Number of attention heads.
        - num_id_token (int): Number of tokens used for identity features.
        - num_queries (int): Number of query tokens for the latent representation.
        - output_dim (int): Output dimension after projection.
        - ff_mult (int): Multiplier for the feed-forward network hidden dimension.
        """
        super().__init__()

        # Storing identity token and query information
        self.num_id_token = num_id_token
        self.dim = dim
        self.num_queries = num_queries
        assert depth % 5 == 0
        self.depth = depth // 5
        scale = dim**-0.5

        # Learnable latent query embeddings
        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) * scale)
        # Projection layer to map the latent output to the desired dimension
        self.proj_out = nn.Parameter(scale * torch.randn(dim, output_dim))

        # Attention and FeedForward layer stack
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        CAbstractor(None, None),
                        # FeedForward layer
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

        # Mappings for each of the 5 different ViT features
        for i in range(5):
            setattr(
                self,
                f"mapping_{i}",
                nn.Sequential(
                    nn.Linear(1024, 1024),
                    nn.LayerNorm(1024),
                    nn.LeakyReLU(),
                    nn.Linear(1024, 1024),
                    nn.LayerNorm(1024),
                    nn.LeakyReLU(),
                    nn.Linear(1024, dim),
                ),
            )

        # Mapping for identity embedding vectors
        self.id_embedding_mapping = nn.Sequential(
            nn.Linear(1280, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(),
            nn.Linear(1024, dim * num_id_token),
        )

    def forward(self, x, y):
        """
        Forward pass for FaceAbstractor.

        Parameters:
        - x (Tensor): The input identity embedding tensor of shape (batch_size, 1280).
        - y (list of Tensor): A list of 5 visual feature tensors each of shape (batch_size, 1024).

        Returns:
        - Tensor: The extracted latent features of shape (batch_size, num_queries, output_dim).
        """

        # Repeat latent queries for the batch size
        latents = self.latents.repeat(x.size(0), 1, 1)

        # Map the identity embedding to tokens
        x = self.id_embedding_mapping(x)
        x = x.reshape(-1, self.num_id_token, self.dim)

        # Concatenate identity tokens with the latent queries
        latents = latents
        # latents = torch.cat((latents, x), dim=1)

        # Process each of the 5 visual feature inputs
        for i in range(5):
            vit_feature = getattr(self, f"mapping_{i}")(y[i])
            # ctx_feature = torch.cat((x, vit_feature), dim=1)
            ctx_feature = vit_feature[:, 1:]

            # Pass through the PerceiverAttention and FeedForward layers
            for attn, ff in self.layers[i * self.depth : (i + 1) * self.depth]:
                x2 = attn(ctx_feature)["last_hidden_state"]
                x2 = ff(x2)

        # Retain only the query latents
        # latents = latents[:, :self.num_queries]
        # Project the latents to the output dimension
        # latents = latents @ self.proj_out
        return x2 @ self.proj_out


class LayerAttention(nn.Module):
    """

    Args:
        dim (int): Dimension of the input latent and output. Default is 3072.
        dim_head (int): Dimension of each attention head. Default is 128.
        heads (int): Number of attention heads. Default is 16.
        kv_dim (int): Dimension of the key/value input, allowing flexible cross-attention. Default is 2048.

    Attributes:
        scale (float): Scaling factor used in dot-product attention for numerical stability.
        norm1 (nn.LayerNorm): Layer normalization applied to the input image features.
        norm2 (nn.LayerNorm): Layer normalization applied to the latent features.
        to_q (nn.Linear): Linear layer for projecting the latent features into queries.
        to_kv (nn.Linear): Linear layer for projecting the input features into keys and values.
        to_out (nn.Linear): Linear layer for outputting the final result after attention.

    """

    def __init__(self, *, dim=3072, dim_head=128, heads=16, kv_dim=2048):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        # Layer normalization to stabilize training
        self.norm1 = nn.LayerNorm(dim if kv_dim is None else kv_dim)
        self.norm2 = nn.LayerNorm(dim)

        # Linear transformations to produce queries, keys, and values
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(
            dim if kv_dim is None else kv_dim, inner_dim * 2, bias=False
        )
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        """

        Args:
            x (torch.Tensor): Input image features with shape (batch_size, n1, D), where:
                - batch_size (b): Number of samples in the batch.
                - n1: Sequence length (e.g., number of patches or tokens).
                - D: Feature dimension.

            latents (torch.Tensor): Latent feature representations with shape (batch_size, n2, D), where:
                - n2: Number of latent elements.

        Returns:
            torch.Tensor: Attention-modulated features with shape (batch_size, n2, D).

        """
        # Apply layer normalization to the input image and latent features
        x = self.norm1(x)
        latents = self.norm2(latents)

        b, seq_len, _ = latents.shape

        # Compute queries, keys, and values
        q = self.to_q(latents)
        k, v = self.to_kv(x).chunk(2, dim=-1)

        # Reshape tensors to split into attention heads
        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # Compute attention weights
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        # More stable scaling than post-division
        weight = (q * scale) @ (k * scale).transpose(-2, -1)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)

        # Compute the output via weighted combination of values
        out = weight @ v

        # Reshape and permute to prepare for final linear transformation
        out = out.permute(0, 2, 1, 3).reshape(b, seq_len, -1)  # (B, 17550, 2048)

        return self.to_out(out)  # (B, 17550, 2048) -> #(B, 17550, 3072)
