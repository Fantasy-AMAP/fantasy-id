"""Custom Honeybee projectors based on Conv and MLP, including C-Abstractor.
"""
from functools import partial

import torch
import torch.nn as nn
from einops import rearrange
from timm.layers import LayerNorm, LayerNorm2d
from timm.models.regnet import RegStage
from transformers.modeling_outputs import BaseModelOutput

"""Honeybee configuration"""
import timm
from transformers import AutoConfig, CLIPVisionConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.models.deformable_detr import DeformableDetrConfig
from transformers.utils import logging
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

def check_local_file(model_name_or_path):
    cache_dir = get_cache_dir()
    file_name = os.path.join(
        cache_dir, f"models--{model_name_or_path.replace('/', '--')}"
    )
    local_files_only = os.path.exists(file_name)
    file_name = file_name if local_files_only else model_name_or_path
    return local_files_only, file_name


logger = logging.get_logger(__name__)


class HoneybeeVisionConfig(PretrainedConfig):
    def __init__(
        self,
        pretrained_vision_name_or_path: str = "openai/clip-vit-large-patch14",
        image_size: int = 224,
        image_mean = OPENAI_CLIP_MEAN,
        image_std = OPENAI_CLIP_STD,
        hidden_size: int = None,
        encoder_type: str = "openai.clip",
        **kwargs,
    ):
        assert hidden_size is not None, "hidden_size is required for HoneybeeVisionConfig"
        super().__init__(**kwargs)
        self.pretrained_vision_name_or_path = pretrained_vision_name_or_path
        self.image_size = image_size
        self.image_mean = image_mean
        self.image_std = image_std
        self.hidden_size = hidden_size
        self.encoder_type = encoder_type

    @staticmethod
    def from_exp_config(vision_config: dict):
        """Build MLLVisionConfig from exp config (hydra conifg)
        """
        pretrained_vision_name_or_path = vision_config.get("pretrained_vision_name_or_path")
        if pretrained_vision_name_or_path is None:
            raise ValueError("pretrained_vision_name_or_path is required for vision config.")

        vm_local_files_only, vm_file_name = check_local_file(pretrained_vision_name_or_path)
        encoder_type = vision_config["encoder_type"]
        if encoder_type == "openai.clip":
            v_enc_config = CLIPVisionConfig.from_pretrained(
                vm_file_name,
                local_files_only=vm_local_files_only,
            )
            v_enc_config = v_enc_config.to_dict()
            if "encoder_type" not in v_enc_config:  # for eval on previously trained models
                v_enc_config["encoder_type"] = encoder_type
        else:
            raise NotImplementedError()

        v_enc_config.update(vision_config)
        v_enc_config = HoneybeeVisionConfig(**v_enc_config)

        return v_enc_config


class HoneybeeVisualProjectorConfig(PretrainedConfig):
    def __init__(
        self,
        projector_type: str = "c-abs",
        num_eos_tokens: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.projector_type = projector_type
        self.num_eos_tokens = num_eos_tokens

    @staticmethod
    def from_exp_config(
        projector_config: dict,
        vision_hidden_size: int,
        lm_hidden_size: int,
    ):
        if projector_config["projector_type"] == "d-abs":
            projector_config = DeformableDetrConfig(**projector_config).to_dict()

        # projector has three inter-module configs:
        # 1) encoder_hidden_size (hidden size of vision model)
        # 2) output_hidden_size (hidden size of LLM)
        # the number of query tokens  (total num_visual_tokens = num_query_tokens + num_eos_tokens)
        inter_module_configs = {
            "encoder_hidden_size": vision_hidden_size,
            "output_hidden_size": lm_hidden_size,
        }

        projector_config = HoneybeeVisualProjectorConfig(
            **(projector_config | inter_module_configs),
        )

        return projector_config


class HoneybeeLanguageConfig(PretrainedConfig):
    def __init__(
        self,
        pretrained_lm_name_or_path: str = "llama-2-7b",
        pretrained_tokenizer_name_or_path: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pretrained_lm_name_or_path = pretrained_lm_name_or_path
        self.pretrained_tokenizer_name_or_path = (
            pretrained_tokenizer_name_or_path or pretrained_lm_name_or_path
        )


class HoneybeeConfig(PretrainedConfig):
    is_composition = True

    def __init__(
        self,
        vision_config: dict,
        projector_config: dict,
        lm_config: dict,
        **kwargs,
    ):
        """Honeybee model config.

        This init function is called with two different scenario:
        - in PT, explicitly called in train.py, with **hydra exp config**.
        - in FT, implicitly called in from_pretrained, with **hf model config**.

        Thus, we need to address both cases.
        """
        super().__init__(**kwargs)

        # Note) three inter-module configs (vision -> projector or lm -> projector):
        # 1) vision_config.hidden_size -> projector_config.encoder_hidden_size
        # 2) text_config.hidden_size -> projector_config.output_hidden_size
        # the number of query tokens (total num_visual_tokens = num_query_tokens + num_eos_tokens)

        # Vision config
        self.vision_config = HoneybeeVisionConfig.from_exp_config(vision_config)

        # LM config (from exp config)
        self.lm_config = HoneybeeLanguageConfig(**lm_config)
        lm_local_files_only, lm_file_name = check_local_file(
            self.lm_config.pretrained_lm_name_or_path
        )
        self.text_config = AutoConfig.from_pretrained(
            lm_file_name,
            local_files_only=lm_local_files_only,
        )

        # Projector config
        self.projector_config = HoneybeeVisualProjectorConfig.from_exp_config(
            projector_config,
            vision_hidden_size=self.vision_config.hidden_size,
            lm_hidden_size=self.text_config.hidden_size,
        )

    @property
    def num_visual_tokens(self):
        return self.projector_config.num_query_tokens + self.projector_config.num_eos_tokens

    @property
    def hidden_size(self):
        # hidden_size is required for deepspeed auto config
        return self.text_config.hidden_size

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = super().to_dict()
        for k, v in output.items():
            if isinstance(v, PretrainedConfig):
                output[k] = v.to_dict()

        return output
    
    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        # update old config
        if "hidden_size" in config_dict:
            config_dict.pop("hidden_size")

        if "visual_projector_config" in config_dict:
            config_dict["projector_config"] = config_dict.pop("visual_projector_config")
            config_dict["projector_config"].pop("encoder_hidden_size")
            config_dict["projector_config"]["num_query_tokens"] = config_dict.pop("num_query_tokens")

        return super().from_dict(config_dict, **kwargs)

def build_pos_embeds(
    config: HoneybeeVisualProjectorConfig, num_input_tokens: int, vision_hidden_size: int
):
    # pos emb
    if config.pos_emb:
        pos_emb = torch.nn.Parameter(torch.zeros(1, num_input_tokens, vision_hidden_size))
        nn.init.trunc_normal_(pos_emb, mean=0.0, std=0.02)
    else:
        pos_emb = None

    return pos_emb


def build_eos_tokens(config: HoneybeeVisualProjectorConfig, output_hidden_size: int):
    # think tokens
    num_eos_tokens = config.num_eos_tokens
    if num_eos_tokens:
        eos_tokens = torch.nn.Parameter(torch.randn(1, num_eos_tokens, output_hidden_size))
        nn.init.trunc_normal_(eos_tokens, mean=0.0, std=config.initializer_range)
    else:
        eos_tokens = None

    return eos_tokens


def build_prenorm(config: HoneybeeVisualProjectorConfig):
    if getattr(config, "prenorm", False):
        prenorm = LayerNorm(config.encoder_hidden_size)
    else:
        prenorm = None
    return prenorm


def build_mlp(depth: int, hidden_size: int, output_hidden_size: int):
    layers = [nn.Linear(hidden_size, output_hidden_size)]
    for _ in range(1, depth):
        layers.append(nn.SiLU())
        layers.append(nn.Linear(output_hidden_size, output_hidden_size))
    return nn.Sequential(*layers)


class Projector(nn.Module):
    """Base projector class"""

    def __init__(
        self,
        config: HoneybeeVisualProjectorConfig,
        num_input_tokens: int,
    ):
        super().__init__()
        num_input_tokens=576
        cabs_config = {
            "projector_type": "d-abs",
            "num_eos_tokens": 0,
            "initializer_range": 0.02,  # initialization std for eos tokens
            "disable_custom_kernels": False,  # use custom cuda kernel or pytorch implementation
            # below is for ablation
            "num_feature_levels": 1,
            "feature_layer_index": -1,  # vision feature layer index; -1: last layer
            "pos_emb": True,
            "pooled_v_target": "query",
            "num_query_tokens": 36,
            "encoder_hidden_size": 1024,
            "output_hidden_size": 1024,
            "depth": 3,
            "mlp_depth": 2,
        }
        config = DeformableDetrConfig()
        config.update(cabs_config)
        self.config = config
        self.num_input_tokens = num_input_tokens

        # think tokens
        self.eos_tokens = build_eos_tokens(config, config.output_hidden_size)

        # pos emb
        self.pos_emb = build_pos_embeds(config, num_input_tokens, config.encoder_hidden_size)

        self.prenorm = build_prenorm(config)

        self.build_net()

    def build_net(self):
        raise NotImplementedError()

    def _forward(self, x):
        raise NotImplementedError()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, encoder_hidden_size) tensor from the visual backbone (CLIP visual encoder),
                including cls token.
        """
        if self.prenorm is not None:
            x = self.prenorm(x)

        if self.pos_emb is not None:
            x += self.pos_emb

        x = self._forward(x)  # (B, L, output_hidden_size)

        B = x.size(0)
        if self.eos_tokens is not None:
            x = torch.cat([x, self.eos_tokens.expand(B, -1, -1)], dim=1)

        output = BaseModelOutput(last_hidden_state=x)
        return output
    
    # def _load_from_state_dict(self, state_dict, *args, **kwargs):
    #     # update old ckpt compatible with current code
    #     pos_emb = state_dict["abstractor.pos_emb"]
    #     if pos_emb.size(1) == self.pos_emb.size(1) + 1:
    #         # remove obsolete first pos emb (for cls token originally)
    #         state_dict["abstractor.pos_emb"] = pos_emb[:, 1:]

    #     super()._load_from_state_dict(state_dict, *args, **kwargs)


class MLPProjector(Projector):
    def build_net(self):
        encoder_hidden_size = self.config.encoder_hidden_size
        output_hidden_size = self.config.output_hidden_size
        depth = self.config.depth

        self.net = build_mlp(depth, encoder_hidden_size, output_hidden_size)

    def _forward(self, x):
        return self.net(x)


class ConvProjector(Projector):
    def _forward(self, x):
        # x: [B, L, dim]
        hw = int(x.size(1) ** 0.5)
        x = rearrange(x, "b (h w) d -> b d h w", h=hw, w=hw)
        x = self.net(x)
        x = rearrange(x, "b d h w -> b (h w) d")
        x = self.readout(x)

        return x


class CAbstractor(ConvProjector):
    """C-Abstractor based on RegBlock"""
    def build_net(self):
        encoder_hidden_size = self.config.encoder_hidden_size
        hidden_size = self.config.hidden_size
        output_hidden_size = self.config.output_hidden_size
        depth = self.config.depth
        mlp_depth = self.config.mlp_depth

        n_queries = self.config.num_query_tokens
        assert (n_queries ** 0.5).is_integer(), "n_queries must be square number"
        hw = int(n_queries ** 0.5)

        RegBlock = partial(
            RegStage,
            stride=1,
            dilation=1,
            act_layer=nn.SiLU,
            norm_layer=LayerNorm2d,
        )

        s1 = RegBlock(
            depth,
            encoder_hidden_size,
            hidden_size,
        )
        sampler = nn.AdaptiveAvgPool2d((hw, hw))
        s2 = RegBlock(
            depth,
            hidden_size,
            hidden_size,
        )

        if depth:
            self.net = nn.Sequential(s1, sampler, s2)
            self.readout = build_mlp(mlp_depth, hidden_size, output_hidden_size)
        else:
            self.net = sampler
            self.readout = build_mlp(mlp_depth, encoder_hidden_size, output_hidden_size)