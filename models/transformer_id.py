# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob
import json
import os
import sys
from typing import Any, Dict, Optional, Tuple, Union

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.attention_processor import (
    AttentionProcessor,
    CogVideoXAttnProcessor2_0,
    FusedCogVideoXAttnProcessor2_0,
)
from diffusers.models.embeddings import (
    CogVideoXPatchEmbed,
    TimestepEmbedding,
    Timesteps,
)
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNorm, CogVideoXLayerNormZero
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_version,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import maybe_allow_in_graph
from einops import rearrange, reduce
from torch import nn

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path)]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

import numpy as np
import open3d as o3d
from face_abstractor import FaceAbstractor, LayerAttention

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@maybe_allow_in_graph
class CogVideoXBlock(nn.Module):
    r"""
    Transformer block used in [CogVideoX](https://github.com/THUDM/CogVideo) model.

    Parameters:
        dim (`int`):
            The number of channels in the input and output.
        num_attention_heads (`int`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`):
            The number of channels in each head.
        time_embed_dim (`int`):
            The number of channels in timestep embedding.
        dropout (`float`, defaults to `0.0`):
            The dropout probability to use.
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to be used in feed-forward.
        attention_bias (`bool`, defaults to `False`):
            Whether or not to use bias in attention projection layers.
        qk_norm (`bool`, defaults to `True`):
            Whether or not to use normalization after query and key projections in Attention.
        norm_elementwise_affine (`bool`, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        norm_eps (`float`, defaults to `1e-5`):
            Epsilon value for normalization layers.
        final_dropout (`bool` defaults to `False`):
            Whether to apply a final dropout after the last feed-forward layer.
        ff_inner_dim (`int`, *optional*, defaults to `None`):
            Custom hidden dimension of Feed-forward layer. If not provided, `4 * dim` is used.
        ff_bias (`bool`, defaults to `True`):
            Whether or not to use bias in Feed-forward layer.
        attention_out_bias (`bool`, defaults to `True`):
            Whether or not to use bias in Attention output projection layer.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        time_embed_dim: int,
        dropout: float = 0.0,
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = False,
        qk_norm: bool = True,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        final_dropout: bool = True,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
    ):
        super().__init__()

        # 1. Self Attention
        self.norm1 = CogVideoXLayerNormZero(
            time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True
        )

        self.attn1 = Attention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            qk_norm="layer_norm" if qk_norm else None,
            eps=1e-6,
            bias=attention_bias,
            out_bias=attention_out_bias,
            processor=CogVideoXAttnProcessor2_0(),
        )

        # 2. Feed Forward
        self.norm2 = CogVideoXLayerNormZero(
            time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True
        )

        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)

        # norm & modulate
        (
            norm_hidden_states,
            norm_encoder_hidden_states,
            gate_msa,
            enc_gate_msa,
        ) = self.norm1(hidden_states, encoder_hidden_states, temb)

        # insert here
        # pass
        def hook_function(name, output_dir, detach=True):
            def forward_hook(module, input, output):
                print("hook here")
                print(input.shape)
                print(output.shape)
                if hasattr(module.processor, "attn_map_all"):
                    timestep = module.processor.timestep
                    timestep_dir = os.path.join(output_dir, str(timestep))
                    layer_dir = os.path.join(timestep_dir, str(name))
                    os.makedirs(layer_dir, exist_ok=True)

                    if detach:
                        attn_map_all = module.processor.attn_map_all.cpu()
                        attn_map_t2v = module.processor.attn_map_t2v.cpu()
                        attn_map_t2t = module.processor.attn_map_t2t.cpu()
                        attn_map_v2v = module.processor.attn_map_v2v.cpu()
                    else:
                        attn_map_all = module.processor.attn_map_all
                        attn_map_t2v = module.processor.attn_map_t2v
                        attn_map_t2t = module.processor.attn_map_t2t
                        attn_map_v2v = module.processor.attn_map_v2v

                    torch.save(
                        attn_map_all,
                        os.path.join(layer_dir, f"{name}_attn_map_all.pth"),
                    )
                    torch.save(
                        attn_map_t2t,
                        os.path.join(layer_dir, f"{name}_attn_map_t2t.pth"),
                    )
                    torch.save(
                        attn_map_v2v,
                        os.path.join(layer_dir, f"{name}_attn_map_v2v.pth"),
                    )
                    torch.save(
                        attn_map_t2v,
                        os.path.join(layer_dir, f"{name}_attn_map_t2v.pth"),
                    )

                    del module.processor.attn_map_all
                    del module.processor.attn_map_t2v
                    del module.processor.attn_map_t2t
                    del module.processor.attn_map_v2v

            return forward_hook

        def register_cross_attention_hook(
            model, hook_function, target_name, output_dir
        ):
            for name, module in model.named_modules():
                if not name.endswith(target_name):
                    continue

                elif isinstance(module.processor, CogVideoXAttnProcessor2_0):
                    module.processor.store_attn_map = True

                hook = module.register_forward_hook(hook_function(name, output_dir))

            return model

        # register_cross_attention_hook(self,hook_function,'attn1', 't')
        # attention

        attn_hidden_states, attn_encoder_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        hidden_states = hidden_states + gate_msa * attn_hidden_states
        encoder_hidden_states = (
            encoder_hidden_states + enc_gate_msa * attn_encoder_hidden_states
        )

        # norm & modulate
        (
            norm_hidden_states,
            norm_encoder_hidden_states,
            gate_ff,
            enc_gate_ff,
        ) = self.norm2(hidden_states, encoder_hidden_states, temb)

        # feed-forward
        norm_hidden_states = torch.cat(
            [norm_encoder_hidden_states, norm_hidden_states], dim=1
        )
        ff_output = self.ff(norm_hidden_states)

        hidden_states = hidden_states + gate_ff * ff_output[:, text_seq_length:]
        encoder_hidden_states = (
            encoder_hidden_states + enc_gate_ff * ff_output[:, :text_seq_length]
        )

        return hidden_states, encoder_hidden_states


class IDTransformer3DModel(ModelMixin, ConfigMixin, PeftAdapterMixin):
    """
    A Transformer model for video-like data in [CogVideoX](https://github.com/THUDM/CogVideo).

    Parameters:
        num_attention_heads (`int`, defaults to `30`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`, defaults to `64`):
            The number of channels in each head.
        in_channels (`int`, defaults to `16`):
            The number of channels in the input.
        out_channels (`int`, *optional*, defaults to `16`):
            The number of channels in the output.
        flip_sin_to_cos (`bool`, defaults to `True`):
            Whether to flip the sin to cos in the time embedding.
        time_embed_dim (`int`, defaults to `512`):
            Output dimension of timestep embeddings.
        text_embed_dim (`int`, defaults to `4096`):
            Input dimension of text embeddings from the text encoder.
        num_layers (`int`, defaults to `30`):
            The number of layers of Transformer blocks to use.
        dropout (`float`, defaults to `0.0`):
            The dropout probability to use.
        attention_bias (`bool`, defaults to `True`):
            Whether or not to use bias in the attention projection layers.
        sample_width (`int`, defaults to `90`):
            The width of the input latents.
        sample_height (`int`, defaults to `60`):
            The height of the input latents.
        sample_frames (`int`, defaults to `49`):
            The number of frames in the input latents. Note that this parameter was incorrectly initialized to 49
            instead of 13 because CogVideoX processed 13 latent frames at once in its default and recommended settings,
            but cannot be changed to the correct value to ensure backwards compatibility. To create a transformer with
            K latent frames, the correct value to pass here would be: ((K - 1) * temporal_compression_ratio + 1).
        patch_size (`int`, defaults to `2`):
            The size of the patches to use in the patch embedding layer.
        temporal_compression_ratio (`int`, defaults to `4`):
            The compression ratio across the temporal dimension. See documentation for `sample_frames`.
        max_text_seq_length (`int`, defaults to `226`):
            The maximum sequence length of the input text embeddings.
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to use in feed-forward.
        timestep_activation_fn (`str`, defaults to `"silu"`):
            Activation function to use when generating the timestep embeddings.
        norm_elementwise_affine (`bool`, defaults to `True`):
            Whether or not to use elementwise affine in normalization layers.
        norm_eps (`float`, defaults to `1e-5`):
            The epsilon value to use in normalization layers.
        spatial_interpolation_scale (`float`, defaults to `1.875`):
            Scaling factor to apply in 3D positional embeddings across spatial dimensions.
        temporal_interpolation_scale (`float`, defaults to `1.0`):
            Scaling factor to apply in 3D positional embeddings across temporal dimensions.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 30,
        attention_head_dim: int = 64,
        in_channels: int = 16,
        out_channels: Optional[int] = 16,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        time_embed_dim: int = 512,
        text_embed_dim: int = 4096,
        num_layers: int = 30,
        dropout: float = 0.0,
        attention_bias: bool = True,
        sample_width: int = 90,
        sample_height: int = 60,
        sample_frames: int = 49,
        patch_size: int = 2,
        temporal_compression_ratio: int = 4,
        max_text_seq_length: int = 226,
        activation_fn: str = "gelu-approximate",
        timestep_activation_fn: str = "silu",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        spatial_interpolation_scale: float = 1.875,
        temporal_interpolation_scale: float = 1.0,
        use_rotary_positional_embeddings: bool = False,
        use_learned_positional_embeddings: bool = False,
        is_train_face: bool = False,
        is_kps: bool = False,
        cross_attn_interval: int = 1,
        LFE_num_tokens: int = 32,
        LFE_output_dim: int = 768,
        LFE_heads: int = 12,
        local_face_scale: float = 1.0,
    ):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim

        if not use_rotary_positional_embeddings and use_learned_positional_embeddings:
            raise ValueError(
                "There are no CogVideoX checkpoints available with disable rotary embeddings and learned positional "
                "embeddings. If you're using a custom model and/or believe this should be supported, please open an "
                "issue at https://github.com/huggingface/diffusers/issues."
            )

        # 1. Patch embedding
        self.patch_embed = CogVideoXPatchEmbed(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=inner_dim,
            text_embed_dim=text_embed_dim,
            bias=True,
            sample_width=sample_width,
            sample_height=sample_height,
            sample_frames=sample_frames,
            temporal_compression_ratio=temporal_compression_ratio,
            max_text_seq_length=max_text_seq_length,
            spatial_interpolation_scale=spatial_interpolation_scale,
            temporal_interpolation_scale=temporal_interpolation_scale,
            use_positional_embeddings=not use_rotary_positional_embeddings,
            use_learned_positional_embeddings=use_learned_positional_embeddings,
        )
        self.embedding_dropout = nn.Dropout(dropout)

        # 2. Time embeddings
        self.time_proj = Timesteps(inner_dim, flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(
            inner_dim, time_embed_dim, timestep_activation_fn
        )

        # 3. Define spatio-temporal transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                CogVideoXBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    time_embed_dim=time_embed_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                )
                for _ in range(num_layers)
            ]
        )

        from timm.models.layers import Mlp

        self.norm_final = nn.LayerNorm(inner_dim, norm_eps, norm_elementwise_affine)

        # 4. Output blocks
        self.norm_out = AdaLayerNorm(
            embedding_dim=time_embed_dim,
            output_dim=2 * inner_dim,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            chunk_dim=1,
        )
        self.proj_out = nn.Linear(inner_dim, patch_size * patch_size * out_channels)

        self.gradient_checkpointing = False

        self.is_train_face = is_train_face
        self.is_kps = is_kps

        if is_train_face:
            self.inner_dim = inner_dim
            self.cross_attn_interval = cross_attn_interval
            self.num_ca = num_layers // cross_attn_interval
            self.LFE_num_tokens = LFE_num_tokens
            self.LFE_output_dim = LFE_output_dim
            self.LFE_heads = LFE_heads
            self.LFE_final_output_dim = int(self.inner_dim / 3 * 2)
            self.local_face_scale = local_face_scale
            self._init_face_inputs()

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    def _init_face_inputs(self):
        device = self.device
        weight_dtype = next(self.transformer_blocks.parameters()).dtype
        self.face_abstractor = FaceAbstractor()

        from models.vertex_encode import Conv1DReducer4, VertexSpiralNet

        self.vertex_spiral_net = VertexSpiralNet()
        self.vertex_spiral_net.to(device)
        self.vertex_reducer = Conv1DReducer4()

        self.face_abstractor.to(device, dtype=weight_dtype)
        self.layer_attention = nn.ModuleList(
            [
                LayerAttention(
                    dim=self.inner_dim,
                    dim_head=128,
                    heads=16,
                    kv_dim=self.LFE_final_output_dim,
                ).to(device, dtype=weight_dtype)
                for _ in range(self.num_ca)
            ]
        )

    def save_face_modules(self, path: str):
        save_dict = {
            "face_abstractor": self.face_abstractor.state_dict(),
            "layer_attention": [ca.state_dict() for ca in self.layer_attention],
        }
        torch.save(save_dict, path)

    def load_face_modules(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.face_abstractor.load_state_dict(checkpoint["face_abstractor"])
        for ca, state_dict in zip(self.layer_attention, checkpoint["layer_attention"]):
            ca.load_state_dict(state_dict)

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(
            name: str,
            module: torch.nn.Module,
            processors: Dict[str, AttentionProcessor],
        ):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(
        self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]
    ):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.fuse_qkv_projections with FusedAttnProcessor2_0->FusedCogVideoXAttnProcessor2_0
    def fuse_qkv_projections(self):
        """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query, key, value)
        are fused. For cross-attention modules, key and value projection matrices are fused.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>
        """
        self.original_attn_processors = None

        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError(
                    "`fuse_qkv_projections()` is not supported for models having added KV projections."
                )

        self.original_attn_processors = self.attn_processors

        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)

        self.set_attn_processor(FusedCogVideoXAttnProcessor2_0())

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.unfuse_qkv_projections
    def unfuse_qkv_projections(self):
        """Disables the fused QKV projection if enabled.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>

        """
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor],
        timestep_cond: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        id_cond: Optional[torch.Tensor] = None,
        id_vit_hidden: Optional[torch.Tensor] = None,
        extra_face=None,
        return_dict: bool = True,
    ):
        # fuse clip and insightface
        if self.is_train_face:
            assert id_cond is not None and id_vit_hidden is not None
            valid_face_emb = self.face_abstractor(
                id_cond, id_vit_hidden
            )  # torch.Size([1, 1280]), list[5](torch.Size([1, 577, 1024]))  ->  torch.Size([1, 32, 2048])

            vertices = extra_face
            vertices = vertices.to(hidden_states.device, dtype=torch.float)
            # vertices = vertices.to(hidden_states.device, dtype=hidden_states.dtype).unsqueeze(0)
            self.vertex_spiral_net.to(dtype=torch.float)
            valid_vertex_emb = self.vertex_spiral_net(
                vertices, valid_face_emb.to(dtype=vertices.dtype)
            )
            valid_vertex_emb = valid_vertex_emb.to(dtype=hidden_states.dtype)

            valid_vertex_emb = rearrange(valid_vertex_emb, "b l c -> b c l")
            valid_vertex_emb = self.vertex_reducer(valid_vertex_emb)
            valid_vertex_emb = rearrange(valid_vertex_emb, "b c l -> b l c")

        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        batch_size, num_frames, channels, height, width = hidden_states.shape

        # 1. Time embedding
        timesteps = timestep
        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        # 2. Patch embedding
        # torch.Size([1, 226, 4096])   torch.Size([1, 13, 32, 60, 90])
        hidden_states = self.patch_embed(
            encoder_hidden_states, hidden_states
        )  # torch.Size([1, 17776, 3072])
        hidden_states = self.embedding_dropout(
            hidden_states
        )  # torch.Size([1, 17776, 3072])

        text_seq_length = encoder_hidden_states.shape[1]
        encoder_hidden_states = hidden_states[
            :, :text_seq_length
        ]  # torch.Size([1, 226, 3072])
        hidden_states = hidden_states[
            :, text_seq_length:
        ]  # torch.Size([1, 17550, 3072])

        # 3. Transformer blocks
        ca_idx = 0
        for i, block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = (
                    {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                )
                (
                    hidden_states,
                    encoder_hidden_states,
                ) = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    emb,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )
            else:
                hidden_states, encoder_hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=emb,
                    image_rotary_emb=image_rotary_emb,
                )

            if self.is_train_face:
                if i % self.cross_attn_interval == 0 and valid_face_emb is not None:
                    # for validation
                    if extra_face is None:
                        logger.info("extra face is None at the stage of infering")
                        mv_face_emb = valid_face_emb
                    else:
                        mv_face_emb = valid_face_emb
                        # mv_face_emb = torch.cat([valid_face_emb, valid_face_emb2], dim=1)
                    # mv_face_emb = torch.randn([1,144,2048]).to(hidden_states.device, hidden_states.dtype)
                    mixed23 = torch.cat([mv_face_emb, valid_vertex_emb], dim=1)
                    hidden_states = hidden_states + self.layer_attention[ca_idx](
                        mixed23, hidden_states
                    )

                    ca_idx += 1

        if not self.config.use_rotary_positional_embeddings:
            # CogVideoX-2B
            hidden_states = self.norm_final(hidden_states)
        else:
            # CogVideoX-5B
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
            hidden_states = self.norm_final(hidden_states)
            hidden_states = hidden_states[:, text_seq_length:]

        # 4. Final block
        hidden_states = self.norm_out(hidden_states, temb=emb)
        hidden_states = self.proj_out(hidden_states)

        # 5. Unpatchify
        # Note: we use `-1` instead of `channels`:
        #   - It is okay to `channels` use for CogVideoX-2b and CogVideoX-5b (number of input channels is equal to output channels)
        #   - However, for CogVideoX-5b-I2V also takes concatenated input image latents (number of input channels is twice the output channels)
        p = self.config.patch_size
        output = hidden_states.reshape(
            batch_size, num_frames, height // p, width // p, -1, p, p
        )
        output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

    @classmethod
    def from_pretrained_cus(
        cls,
        pretrained_model_path,
        subfolder=None,
        config_path=None,
        transformer_additional_kwargs={},
    ):
        if subfolder:
            config_path = config_path or pretrained_model_path
            config_file = os.path.join(config_path, subfolder, "config.json")
            pretrained_model_path = os.path.join(pretrained_model_path, subfolder)
        else:
            config_file = os.path.join(
                config_path or pretrained_model_path, "config.json"
            )

        print(
            f"Loading 3D transformer's pretrained weights from {pretrained_model_path} ..."
        )

        # Check if config file exists
        if not os.path.isfile(config_file):
            raise RuntimeError(f"Configuration file '{config_file}' does not exist")

        # Load the configuration
        with open(config_file, "r") as f:
            config = json.load(f)

        from diffusers.utils import WEIGHTS_NAME

        model = cls.from_config(config, **transformer_additional_kwargs)
        model_file = os.path.join(pretrained_model_path, WEIGHTS_NAME)
        model_file_safetensors = model_file.replace(".bin", ".safetensors")
        if os.path.exists(model_file):
            state_dict = torch.load(model_file, map_location="cpu")
        elif os.path.exists(model_file_safetensors):
            from safetensors.torch import load_file

            state_dict = load_file(model_file_safetensors)
        else:
            from safetensors.torch import load_file

            model_files_safetensors = glob.glob(
                os.path.join(pretrained_model_path, "*.safetensors")
            )
            state_dict = {}
            for model_file_safetensors in model_files_safetensors:
                _state_dict = load_file(model_file_safetensors)
                for key in _state_dict:
                    state_dict[key] = _state_dict[key]

        if (
            model.state_dict()["patch_embed.proj.weight"].size()
            != state_dict["patch_embed.proj.weight"].size()
        ):
            new_shape = model.state_dict()["patch_embed.proj.weight"].size()
            if len(new_shape) == 5:
                state_dict["patch_embed.proj.weight"] = (
                    state_dict["patch_embed.proj.weight"]
                    .unsqueeze(2)
                    .expand(new_shape)
                    .clone()
                )
                state_dict["patch_embed.proj.weight"][:, :, :-1] = 0
            else:
                if (
                    model.state_dict()["patch_embed.proj.weight"].size()[1]
                    > state_dict["patch_embed.proj.weight"].size()[1]
                ):
                    model.state_dict()["patch_embed.proj.weight"][
                        :, : state_dict["patch_embed.proj.weight"].size()[1], :, :
                    ] = state_dict["patch_embed.proj.weight"]
                    model.state_dict()["patch_embed.proj.weight"][
                        :, state_dict["patch_embed.proj.weight"].size()[1] :, :, :
                    ] = 0
                    state_dict["patch_embed.proj.weight"] = model.state_dict()[
                        "patch_embed.proj.weight"
                    ]
                else:
                    model.state_dict()["patch_embed.proj.weight"][
                        :, :, :, :
                    ] = state_dict["patch_embed.proj.weight"][
                        :,
                        : model.state_dict()["patch_embed.proj.weight"].size()[1],
                        :,
                        :,
                    ]
                    state_dict["patch_embed.proj.weight"] = model.state_dict()[
                        "patch_embed.proj.weight"
                    ]

        tmp_state_dict = {}
        for key in state_dict:
            if (
                key in model.state_dict().keys()
                and model.state_dict()[key].size() == state_dict[key].size()
            ):
                tmp_state_dict[key] = state_dict[key]
            else:
                print(key, "Size don't match, skip")
        state_dict = tmp_state_dict

        m, u = model.load_state_dict(state_dict, strict=False)
        print(f"### missing keys: {len(m)}; \n### unexpected keys: {len(u)};")
        print(m)

        params = [p.numel() if "mamba" in n else 0 for n, p in model.named_parameters()]
        print(f"### Mamba Parameters: {sum(params) / 1e6} M")

        params = [
            p.numel() if "attn1." in n else 0 for n, p in model.named_parameters()
        ]
        print(f"### attn1 Parameters: {sum(params) / 1e6} M")

        return model
