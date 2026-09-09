import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchvision import transforms  # 保留以兼容原依赖
from math import sqrt, log
from typing import Dict
import logging

# ====== 你的原始 import，保持不变 ======
import mast3r.utils.path_to_dust3r  # noqa
from dust3r.heads.postprocess import reg_dense_depth, reg_dense_conf  # noqa
from dust3r.heads.dpt_head import PixelwiseTaskWithDPT  # noqa
import dust3r.utils.path_to_croco  # noqa
from models.blocks import Mlp  # noqa

# ====== 新增：本地 DINOv3 适配器（支持 .pth 或 HF 本地目录） ======
# 依赖：pip install timm>=1.0.20 transformers>=4.41 torchvision>=0.17
from transformers import AutoModel
import timm

DINOV3_LOCAL_DIR = os.environ.get(
    "DINOV3_LOCAL_DIR",
    os.path.abspath(os.path.join(os.getcwd(), "models", "dinov3-vitl16-pretrain-lvd1689m")),
)
DINOV3_LOCAL_PTH = os.path.join(DINOV3_LOCAL_DIR, "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")
USE_PTH = True  # True: 用 .pth + timm；False: 用 HF 本地目录 from_pretrained


def _strip_prefix_if_present(state_dict: Dict[str, torch.Tensor], prefix="module."):
    keys = list(state_dict.keys())
    if len(keys) > 0 and all(k.startswith(prefix) for k in keys):
        return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict


def _resize_pos_embed_for_timm(state_dict, model, img_size: int, patch_size: int):
    """
    将 pos_embed 从预训练分辨率插值到 (img_size//patch_size)^2 网格。
    仅当 state_dict 中含 'pos_embed' 且网格不匹配时才做插值。
    """
    if "pos_embed" not in state_dict:
        return state_dict
    pos = state_dict["pos_embed"]  # [1, 1+N, C]
    if pos.ndim != 3 or pos.shape[1] < 2:
        return state_dict
    cls_pos, patch_pos = pos[:, :1], pos[:, 1:]  # [1,1,C], [1,N,C]
    C = patch_pos.shape[-1]
    N = patch_pos.shape[1]
    src_hw = int(round(N ** 0.5))
    if src_hw * src_hw != N:
        return state_dict  # 不是方阵网格，跳过

    patch_pos = patch_pos.reshape(1, src_hw, src_hw, C).permute(0, 3, 1, 2).contiguous()  # [1,C,src,src]
    dst_hw = img_size // patch_size
    if dst_hw != src_hw:
        patch_pos = F.interpolate(patch_pos, size=(dst_hw, dst_hw), mode="bicubic", align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, dst_hw * dst_hw, C).contiguous()
        state_dict["pos_embed"] = torch.cat([cls_pos, patch_pos], dim=1)
    return state_dict


class DINOv3Adapter(nn.Module):
    """
    让本地 DINOv3 看起来像你原先使用的 DINOv2：
      - 暴露 .embed_dim, .patch_size
      - forward_features(x_bchw:[B,3,H,W] in [0,1]) -> {'x_norm_patchtokens':[B,N,C]}
    支持两种加载：
      1) .pth（state_dict）：timm 构建 ViT-L/16，再 load_state_dict
      2) HF 本地目录（含 config.json）：AutoModel.from_pretrained(local_dir)
    """
    def __init__(self, device=None, img_size=256, patch_size=16,
                 use_pth=True, local_dir=DINOV3_LOCAL_DIR, local_pth=DINOV3_LOCAL_PTH):
        super().__init__()
        self.device = device or torch.device("cpu")
        self.img_size = img_size
        self.patch_size = patch_size

        if use_pth:
            # 路线 A：仅 .pth（state_dict）
            assert os.path.isfile(local_pth), f"Local .pth not found: {local_pth}"
            self.model = timm.create_model(
                "vit_large_patch16_224",  # ViT-L/16
                img_size=img_size, pretrained=False, num_classes=0
            ).to(self.device)
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad_(False)

            sd = torch.load(local_pth, map_location="cpu")
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            sd = _strip_prefix_if_present(sd, "module.")
            sd = _strip_prefix_if_present(sd, "model.")
            sd = _resize_pos_embed_for_timm(sd, self.model, img_size=img_size, patch_size=patch_size)

            missing, unexpected = self.model.load_state_dict(sd, strict=False)
            if missing or unexpected:
                print(f"[DINOv3Adapter] load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}")

            self.embed_dim = self.model.num_features
            self._use_timm = True
        else:
            # 路线 B：HF 本地目录
            assert os.path.isdir(local_dir), f"Local HF dir not found: {local_dir}"
            self.model = AutoModel.from_pretrained(local_dir, local_files_only=True).to(self.device)
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad_(False)
            self.embed_dim = self.model.config.hidden_size
            self._use_timm = False

        # 预处理放到 forward 中在 GPU 批处理，避免 CPU 往返
        self.register_buffer("_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    @torch.inference_mode()
    def forward_features(self, x_bchw: torch.Tensor):
        """
        x_bchw: [B,3,H,W] in [0,1], Tensor (位于当前 device 上)
        return: {'x_norm_patchtokens': [B, N, C]}
        """
        dev = x_bchw.device
        # GPU 批处理 resize + 归一化（与 torchvision 标准等价）
        x = F.interpolate(x_bchw, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        x = (x.float() - self._mean.to(dev)) / self._std.to(dev)

        if self._use_timm:
            out = self.model.forward_features(x)  # dict 或 Tensor
            tokens = out["x"] if isinstance(out, dict) else out  # [B, 1+N, C]
            tokens = tokens[:, 1:, :]
        else:
            out = self.model(pixel_values=x, output_hidden_states=True)
            tokens = out.last_hidden_state[:, 1:, :]  # [B, N, C]

        return {"x_norm_patchtokens": tokens}


# --- 辅助函数部分（保持不变） ---
def reg_desc(desc, mode):
    if 'norm' in mode:
        desc = desc / desc.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    else:
        raise ValueError(f"Unknown desc mode {mode}")
    return desc


def postprocess(out, depth_mode, conf_mode, desc_dim=None, desc_mode='norm', two_confs=False, desc_conf_mode=None):
    if desc_conf_mode is None:
        desc_conf_mode = conf_mode
    fmap = out.permute(0, 2, 3, 1)  # B,H,W,D
    res = dict(pts3d=reg_dense_depth(fmap[..., 0:3], mode=depth_mode))

    current_idx = 3
    if conf_mode is not None and fmap.shape[-1] >= current_idx + 1:
        res['conf'] = reg_dense_conf(fmap[..., current_idx], mode=conf_mode)
        current_idx += 1

    if desc_dim is not None and fmap.shape[-1] >= current_idx + desc_dim:
        res['desc'] = reg_desc(fmap[..., current_idx:current_idx + desc_dim], mode=desc_mode)
        current_idx += desc_dim
        if two_confs and fmap.shape[-1] >= current_idx + 1:
            res['desc_conf'] = reg_dense_conf(fmap[..., current_idx], mode=desc_conf_mode)
        elif 'conf' in res:
            res['desc_conf'] = res['conf'].clone()
    return res


def reg_dense_offsets(xyz, shift=6.0):
    d = xyz.norm(dim=-1, keepdim=True)
    xyz = xyz / d.clip(min=1e-8)
    return xyz * (torch.exp(d - shift) - torch.exp(torch.zeros_like(d) - shift))


def reg_dense_scales(scales):
    return scales.exp()


def reg_dense_rotation(rotations, eps=1e-8):
    return rotations / (rotations.norm(dim=-1, keepdim=True) + eps)


def reg_dense_sh(sh):
    return rearrange(sh, '... (xyz d_sh) -> ... xyz d_sh', xyz=3)


def reg_dense_opacities(opacities):
    return opacities.sigmoid()


def gaussian_postprocess(out, depth_mode, conf_mode, desc_dim=None, desc_mode='norm', two_confs=False, desc_conf_mode=None, use_offsets=False, sh_degree=1):
    if desc_conf_mode is None:
        desc_conf_mode = conf_mode

    fmap = out.permute(0, 2, 3, 1)  # B,H,W,D

    ch_pts3d = 3
    ch_conf_pts3d = 1 if conf_mode is not None else 0
    ch_desc = desc_dim if desc_dim is not None else 0
    ch_conf_desc = 1 if two_confs and ch_desc > 0 else 0

    ch_offset = 3
    ch_scales = 3
    ch_rotations = 4
    # 使用正确的 SH 通道数（每个 xyz 各有 sh_degree 通道）
    sh_channels = 3 * sh_degree
    ch_opacities = 1

    expected_splits = [ch for ch in [
        ch_pts3d, ch_conf_pts3d, ch_desc, ch_conf_desc, ch_offset,
        ch_scales, ch_rotations, sh_channels, ch_opacities
    ] if ch > 0]

    if sum(expected_splits) != fmap.shape[-1]:
        raise ValueError(f"[gaussian_postprocess] Channel mismatch. Expected sum of splits {sum(expected_splits)}, but got tensor with {fmap.shape[-1]} channels.")

    fmap_parts = torch.split(fmap, expected_splits, dim=-1)

    res = {}
    current_part_idx = 0

    def get_part():
        nonlocal current_part_idx
        part = fmap_parts[current_part_idx]
        current_part_idx += 1
        return part

    res['pts3d'] = reg_dense_depth(get_part(), mode=depth_mode)
    if ch_conf_pts3d:
        res['conf'] = reg_dense_conf(get_part().squeeze(-1), mode=conf_mode)
    if ch_desc:
        res['desc'] = reg_desc(get_part(), mode=desc_mode)
    if ch_conf_desc:
        res['desc_conf'] = reg_dense_conf(get_part().squeeze(-1), mode=desc_conf_mode)
    elif ch_desc and not two_confs and ch_conf_pts3d:
        res['desc_conf'] = res['conf'].clone()

    offset_raw = get_part()
    scales_raw = get_part()  # 原始 logits
    rotations_raw = get_part()
    sh_raw = get_part()
    opacities_raw = get_part()

    # 同时返回 logits 与激活后的值
    res['scale_logits'] = scales_raw
    res['scales'] = reg_dense_scales(scales_raw)

    res['offset_pred'] = reg_dense_offsets(offset_raw)
    res['rotations'] = reg_dense_rotation(rotations_raw)
    res['sh'] = reg_dense_sh(sh_raw)
    res['opacities'] = reg_dense_opacities(opacities_raw)

    # 注意：这里保持与原始代码一致：means 用 pts3d.detach()
    res['means'] = res['pts3d'].detach() + res['offset_pred'] if use_offsets else res['pts3d'].detach()
    return res


# --- DINOv2 增强路径的独立预测头（保持不变的命名与用法） ---
class DinoFusionHead(nn.Module):
    def __init__(self, fusion_input_dim, fusion_output_dim, hidden_dim_factor=2.0):
        super().__init__()
        hidden_dim = int(fusion_output_dim / hidden_dim_factor)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, fusion_output_dim)
        )
        print(f"✅ [Init] DinoFusionHead created: Input({fusion_input_dim}) -> Hidden({hidden_dim}) -> Output({fusion_output_dim})")

    def forward(self, fusion_features):
        return self.fusion_mlp(fusion_features)


class GaussianHead(PixelwiseTaskWithDPT):

    def __init__(
        self,
        net,
        has_conf=False,
        local_feat_dim=16,
        hidden_dim_factor=4.0,
        hooks_idx=None,
        dim_tokens=None,
        num_channels=1,
        postprocess=None,
        feature_dim=256,
        last_dim=32,
        depth_mode=None,
        conf_mode=None,
        head_type="regression",
        use_offsets=False,
        sh_degree=1,
        use_dino=True,
        **kwargs,
    ):
        super().__init__(
            num_channels=num_channels, feature_dim=feature_dim, last_dim=last_dim,
            hooks_idx=hooks_idx, dim_tokens=dim_tokens, depth_mode=depth_mode,
            postprocess=postprocess, conf_mode=conf_mode, head_type=head_type,
        )

        self.local_feat_dim = local_feat_dim
        patch_size = net.patch_embed.patch_size
        if isinstance(patch_size, tuple):
            patch_size = patch_size[0]
        self.patch_size = patch_size
        self.desc_mode = net.desc_mode
        self.two_confs = net.two_confs
        self.desc_conf_mode = net.desc_conf_mode
        self.use_offsets = use_offsets
        self.sh_degree = sh_degree

        def DBG(tag, *msgs):
            if (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
                print(f"[{tag} GaussianHead]", *msgs)
        self.DBG = DBG

        # ====== 替换：用本地 DINOv3 ======
        self.use_dino = use_dino
        self.dino_model = None
        self.actual_dino_dim_for_mlp = 0
        if self.use_dino:
            try:
                dev = next(net.parameters()).device
                self.dino_model = DINOv3Adapter(
                    device=dev, img_size=256, patch_size=16,
                    use_pth=USE_PTH, local_dir=DINOV3_LOCAL_DIR, local_pth=DINOV3_LOCAL_PTH
                ).to(dev)

                if kwargs.get("dino_freeze_weights", True):
                    self.dino_model.eval()
                    for p in self.dino_model.parameters():
                        p.requires_grad_(False)

                self.actual_dino_dim_for_mlp = self.dino_model.embed_dim
                self.dino_internal_patch_size = 16
                self.dino_preprocess_transform = None  # 预处理在适配器里做

            except Exception as e:
                self.DBG("Init", f"ERROR: DINOv3 setup failed. Disabling. Error: {e}")
                self.use_dino = False
                self.actual_dino_dim_for_mlp = 0

        # 原始 splatt3r 路径的头（保持不变）
        splatt3r_feature_dim = net.enc_embed_dim + net.dec_embed_dim
        mlp_output_channels = self.local_feat_dim
        if self.two_confs:
            mlp_output_channels += 1

        self.head_local_features = Mlp(
            in_features=splatt3r_feature_dim,
            hidden_features=int(hidden_dim_factor * splatt3r_feature_dim),
            out_features=mlp_output_channels * self.patch_size**2,
        )
        self.DBG("Init", f"✅ Created baseline splatt3r head: Input({splatt3r_feature_dim})")

        # DINO 增强路径的头（保持不变）
        self.dino_fusion_head = None
        if self.use_dino:
            fusion_input_dim = splatt3r_feature_dim + self.actual_dino_dim_for_mlp
            fusion_output_dim = mlp_output_channels * self.patch_size**2
            self.dino_fusion_head = DinoFusionHead(
                fusion_input_dim=fusion_input_dim,
                fusion_output_dim=fusion_output_dim,
                hidden_dim_factor=2.0
            )

        # SH 通道数与初始化（保持你的修正）
        sh_channels = sh_degree
        gaussian_num_channels = 3 + 3 + 4 + 3 * sh_channels + 1
        self.gaussian_dpt = PixelwiseTaskWithDPT(
            num_channels=gaussian_num_channels, feature_dim=feature_dim, last_dim=last_dim,
            hooks_idx=hooks_idx, dim_tokens=dim_tokens, depth_mode=depth_mode,
            postprocess=None, conf_mode=None, head_type=head_type,
        )

        final_conv_gauss = self.gaussian_dpt.dpt.head[-1]
        splits_init_gauss = [
            (3, 1e-3, 1e-3),           # offsets
            (3, 3e-5, -7.0),           # scales
            (4, 1.0, 0.0),             # rotations
            (sh_channels, 1.0, 0.0),   # SH
            (1, 1.0, -2.0),            # opacity
        ]
        ptr = 0
        for ch, gain, bias_init_val in splits_init_gauss:
            if ch > 0 and ptr + ch <= final_conv_gauss.weight.shape[0]:
                nn.init.xavier_uniform_(final_conv_gauss.weight[ptr: ptr + ch], gain)
                nn.init.constant_(final_conv_gauss.bias[ptr: ptr + ch], bias_init_val)
                ptr += ch

    def forward(self, decout, img_shape, img_rgb=None):
        B = decout[0].shape[0]
        # DPT 主干：保持原分辨率输出（AMP 由上层 Lightning 控制）
        pts3d_block = self.dpt(decout, image_size=img_shape)
        enc_out, dec_out = decout[0], decout[-1]

        # 双路径前向（保持不变）
        splatt3r_features_seq = torch.cat([enc_out, dec_out], dim=-1)  # [B,HW,enc+dec]
        baseline_local_features_flat = self.head_local_features(splatt3r_features_seq)

        dino_seq_aligned = None
        if self.use_dino and self.dino_model and (img_rgb is not None):
            # 确保 DINOv3 adapter 与主干同设备
            dev_enc = enc_out.device
            if hasattr(self.dino_model, 'device') and self.dino_model.device != dev_enc:
                self.dino_model.to(dev_enc)
                self.dino_model.device = dev_enc  # 让 adapter 内部也记住新 device

            with torch.no_grad():
                # 直接把 [B,3,H,W] 喂给适配器，预处理在适配器里（GPU 批处理）
                dino_all_features = self.dino_model.forward_features(img_rgb)
                dino_patch_tokens = dino_all_features.get('x_norm_patchtokens')  # [B,N,C]
                if dino_patch_tokens is not None:
                    # 还原为 feature map 后对齐到 splatt3r 的 patch 网格
                    h_dino_grid = (256 // self.dino_internal_patch_size)
                    w_dino_grid = (256 // self.dino_internal_patch_size)
                    dino_feat_map = dino_patch_tokens.permute(0, 2, 1).reshape(
                        B, dino_patch_tokens.shape[-1], h_dino_grid, w_dino_grid
                    )

                    splatter_feat_grid_h = img_shape[0] // self.patch_size
                    splatter_feat_grid_w = img_shape[1] // self.patch_size
                    if (dino_feat_map.shape[2] != splatter_feat_grid_h) or (dino_feat_map.shape[3] != splatter_feat_grid_w):
                        dino_feat_map_aligned = F.interpolate(
                            dino_feat_map, size=(splatter_feat_grid_h, splatter_feat_grid_w),
                            mode="bilinear", align_corners=False
                        )
                    else:
                        dino_feat_map_aligned = dino_feat_map

                    dino_seq_aligned = dino_feat_map_aligned.flatten(2).transpose(1, 2).contiguous()  # [B,HW,C]
                    # 与主干序列长度严格一致
                    if dino_seq_aligned.shape[1] != enc_out.shape[1]:
                        self.DBG("FWD DINO",
                                 f"ERROR: DINO seq len ({dino_seq_aligned.shape[1]}) != splatt3r seq len ({enc_out.shape[1]}). Disabling fusion.")
                        dino_seq_aligned = None
                    else:
                        # 对齐 dtype：与 enc/dec 的 dtype 一致（AMP 下为半精度）
                        dino_seq_aligned = dino_seq_aligned.to(dtype=enc_out.dtype)

        if self.dino_fusion_head and dino_seq_aligned is not None:
            fusion_input_features = torch.cat([splatt3r_features_seq, dino_seq_aligned], dim=-1)
            residual_local_features_flat = self.dino_fusion_head(fusion_input_features)
            local_features_flat = baseline_local_features_flat + residual_local_features_flat
        else:
            local_features_flat = baseline_local_features_flat

        H, W = img_shape
        H_grid, W_grid = H // self.patch_size, W // self.patch_size
        # transpose 后才能 pixel_shuffle；这里保留 contiguous 以满足后续 view 的内存布局要求
        local_features_presuffle = local_features_flat.transpose(-1, -2).contiguous().view(B, -1, H_grid, W_grid)
        local_features_map = nn.functional.pixel_shuffle(local_features_presuffle, self.patch_size)

        gaussian_params_map = self.gaussian_dpt.dpt(decout, image_size=(H, W))

        out_concatenated = torch.cat([pts3d_block, local_features_map, gaussian_params_map], dim=1)

        final_output_dict = gaussian_postprocess(
            out_concatenated, depth_mode=self.depth_mode, conf_mode=self.conf_mode,
            desc_dim=self.local_feat_dim, desc_mode=self.desc_mode,
            two_confs=self.two_confs, desc_conf_mode=self.desc_conf_mode,
            use_offsets=self.use_offsets, sh_degree=self.sh_degree,
        )

        # 训练期对 scale logits 的裁剪（保持不变）
        if self.training and 'scale_logits' in final_output_dict:
            max_scale_limit = 1.0
            max_logit_limit = log(max_scale_limit)  # log(1.0) = 0.0
            clamped_logits = final_output_dict['scale_logits'].clamp(max=max_logit_limit)
            final_output_dict['scales'] = torch.exp(clamped_logits)

        return final_output_dict


def mast3r_head_factory(head_type, output_mode, net, has_conf=False, use_offsets=False, sh_degree=1):
    feature_dim_dpt = 256
    last_dim_dpt = feature_dim_dpt // 2
    l2 = net.dec_depth
    hooks_idx_dpt = [0, l2 * 2 // 4, l2 * 3 // 4, l2]
    dim_tokens_dpt = [net.enc_embed_dim, net.dec_embed_dim, net.dec_embed_dim, net.dec_embed_dim]

    if head_type == 'gaussian_head' and output_mode.startswith('pts3d+gaussian+desc'):
        local_feat_dim_val = int(output_mode.split('desc')[1])
        num_channels_gauss_main_dpt = 3 + int(has_conf)
        gaussian_head_kwargs = {"use_dino": True}
        return GaussianHead(
            net, local_feat_dim=local_feat_dim_val, has_conf=has_conf,
            num_channels=num_channels_gauss_main_dpt, feature_dim=feature_dim_dpt,
            last_dim=last_dim_dpt, hooks_idx=hooks_idx_dpt,
            dim_tokens=dim_tokens_dpt, postprocess=postprocess,
            depth_mode=net.depth_mode, conf_mode=net.conf_mode,
            head_type='regression', use_offsets=use_offsets,
            sh_degree=sh_degree, **gaussian_head_kwargs
        )
    else:
        raise NotImplementedError(f"unexpected {head_type=} and {output_mode=}")
