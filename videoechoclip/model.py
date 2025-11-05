from typing import Optional, Tuple
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from transformers import VivitImageProcessor, VideoMAEImageProcessor, VivitConfig, VivitModel, AutoConfig, AutoModel
from open_clip.model import get_cast_dtype, CLIPTextCfg, _build_text_tower
from omegaconf import DictConfig


OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)


def create_model_and_transforms(
    args: DictConfig = None,
    output_dict: bool = True,
):
    device = torch.device(args.device)

    preprocess_train = preprocess_val = VivitImageProcessor(
        do_resize=True,
        size={"shortest_edge": 224},
        do_center_crop=True,
        crop_size={"height": 224, "width": 224},
        do_rescale=True,
        rescale_factor=1 / 255,  # [0, 255] -> [0, 1]
        offset=False,  # NOTE if true, [0, 1] -> [-1, 1]
        do_normalize=True,
        image_mean=args.image_mean or OPENAI_DATASET_MEAN,
        image_std=args.image_std or OPENAI_DATASET_STD,
    )

    vision_cfg = CLIPVisionCfg(**args.model.vision)
    text_cfg = CLIPTextCfg(**args.model.text)

    model_kwargs = {}
    if args.siglip:
        model_kwargs["init_logit_scale"] = np.log(10)  # different from CLIP
        model_kwargs["init_logit_bias"] = -10

    cast_dtype = get_cast_dtype(args.precision)
    model = CustomTextCLIP(
        args.model.embed_dim,
        vision_cfg,
        text_cfg,
        cast_dtype=cast_dtype,
        output_dict=output_dict,
        **model_kwargs,
    ).to(device)

    return model, preprocess_train, preprocess_val


@dataclass
class CLIPVisionCfg:
    name: str = None
    image_size: int = None
    pretrained: str = None
    num_frames: int = None
    tubelet_size: Tuple[int, int, int] = None
    embed_dim: int = None


def _build_vision_tower(embed_dim: int, vision_cfg: CLIPVisionCfg):
    if vision_cfg.name == "vivit":
        if vision_cfg.pretrained is None:
            hf_config = VivitConfig(
                image_size=vision_cfg.image_size,
                num_frames=vision_cfg.num_frames,
                tubelet_size=vision_cfg.tubelet_size,
                hidden_size=vision_cfg.embed_dim,
            )
            hf_model = VivitModel(hf_config)
        else:
            hf_config = VivitConfig(
                image_size=vision_cfg.image_size,
                num_frames=vision_cfg.num_frames,
                tubelet_size=vision_cfg.tubelet_size,
                hidden_size=vision_cfg.embed_dim,
            )
            hf_model = VivitModel.from_pretrained(vision_cfg.pretrained, config=hf_config)

        hf_model.pooler.dense = nn.Linear(vision_cfg.embed_dim, embed_dim)
        hf_model.pooler.activation = nn.Identity()

        visual = HFVivitWrapper(hf_model)

    elif vision_cfg.name == "videomae":
        if vision_cfg.pretrained is None:
            hf_config = AutoConfig.from_pretrained("OpenGVLab/VideoMAEv2-Base", trust_remote_code=True)
            hf_model = AutoModel.from_config(hf_config, trust_remote_code=True)
        else:
            hf_config = AutoConfig.from_pretrained(vision_cfg.pretrained, trust_remote_code=True)
            hf_model = AutoModel.from_pretrained(vision_cfg.pretrained, config=hf_config, trust_remote_code=True)

        hf_model.model.head = nn.Linear(hf_model.model.fc_norm.weight.shape[0], embed_dim)

        visual = HFVideoMAEWrapper(hf_model)

    else:
        raise ValueError(f"Unsupported model name: {vision_cfg.name}")

    return visual


class CustomTextCLIP(nn.Module):
    output_dict: torch.jit.Final[bool]

    def __init__(
        self,
        embed_dim: int,
        vision_cfg: CLIPVisionCfg,
        text_cfg: CLIPTextCfg,
        quick_gelu: bool = False,
        init_logit_scale: float = np.log(1 / 0.07),
        init_logit_bias: Optional[float] = None,
        cast_dtype: Optional[torch.dtype] = None,
        output_dict: bool = False,
    ):
        super().__init__()
        self.output_dict = output_dict
        self.visual = _build_vision_tower(embed_dim, vision_cfg)
        self.text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        self.context_length = self.text.context_length
        self.vocab_size = self.text.vocab_size
        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones([]) * init_logit_bias)
        else:
            self.logit_bias = None

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        raise NotImplementedError
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    def lock_text_tower(self, unlocked_layers: int = 0, freeze_layer_norm: bool = True):
        self.text.lock(unlocked_layers, freeze_layer_norm)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        raise NotImplementedError
        self.visual.set_grad_checkpointing(enable)
        self.text.set_grad_checkpointing(enable)

    def encode_image(self, image, normalize: bool = False):
        features = self.visual(image)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        features = self.text(text)
        return F.normalize(features, dim=-1) if normalize else features

    def get_logits(self, image, text):
        image_features = self.encode_image(image, normalize=True)
        text_features = self.encode_text(text, normalize=True)
        image_logits = self.logit_scale.exp() * image_features @ text_features.T
        if self.logit_bias is not None:
            image_logits += self.logit_bias
        text_logits = image_logits.T
        return image_logits, text_logits

    def forward(
        self,
        image: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
    ):
        image_features = self.encode_image(image, normalize=True) if image is not None else None
        text_features = self.encode_text(text, normalize=True) if text is not None else None

        if self.output_dict:
            out_dict = {"image_features": image_features, "text_features": text_features, "logit_scale": self.logit_scale.exp()}
            if self.logit_bias is not None:
                out_dict["logit_bias"] = self.logit_bias
            return out_dict

        if self.logit_bias is not None:
            return image_features, text_features, self.logit_scale.exp(), self.logit_bias
        return image_features, text_features, self.logit_scale.exp()


class HFVivitWrapper(nn.Module):
    def __init__(self, hf_model):
        super().__init__()

        self.model = hf_model

    def forward(self, image):
        out = self.model(image)

        return out.pooler_output


class HFVideoMAEWrapper(nn.Module):
    def __init__(self, hf_model):
        super().__init__()

        self.model = hf_model

    def forward(self, image):
        image = image.permute(0, 2, 1, 3, 4)  # NOTE VideoMAEのみ (B, 3, N, H', W')
        out = self.model(image)

        return out
