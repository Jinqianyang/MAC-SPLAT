# data/data.py

import logging
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import PIL
import torch
import torchvision

from src.mast3r_src.dust3r.dust3r.datasets.utils.transforms import ImgNorm
from src.mast3r_src.dust3r.dust3r.utils.geometry import (
    depthmap_to_absolute_camera_coordinates,
)
import src.mast3r_src.dust3r.dust3r.datasets.utils.cropping as cropping

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
LOG = logging.getLogger(__name__)
if not LOG.handlers:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


# -----------------------------------------------------------------------------
# Exception: mark a sample to be skipped (NOT a fatal error)
# -----------------------------------------------------------------------------
class SkipSample(Exception):
    """Raised to indicate the current sample should be skipped (non-fatal)."""


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _to_pil(image):
    """Ensure an input image is PIL.Image for downstream utils."""
    if isinstance(image, PIL.Image.Image):
        return image
    if isinstance(image, torch.Tensor):
        # expect CHW or HWC float [0,1] / uint8
        if image.ndim == 3 and image.shape[0] in (1, 3):
            image = torchvision.transforms.ToPILImage()(image)
        elif image.ndim == 3 and image.shape[-1] in (1, 3):
            image = torchvision.transforms.ToPILImage()(image.permute(2, 0, 1))
        else:
            image = torchvision.transforms.ToPILImage()(image)
        return image
    if isinstance(image, np.ndarray):
        return PIL.Image.fromarray(image)
    # fallback
    return PIL.Image.fromarray(np.array(image))


def crop_resize_if_necessary(
    image, depthmap, intrinsics, resolution: Tuple[int, int]
):
    """
    Adapted from DUST3R's Co3D dataset implementation.
    Downscale-and-crop so that (image.size == resolution), and update intrinsics.
    """
    image = _to_pil(image)

    # Downscale with lanczos interpolation; window centered on principal point
    W, H = image.size
    cx, cy = intrinsics[:2, 2].round().astype(int)
    min_margin_x = min(cx, W - cx)
    min_margin_y = min(cy, H - cy)
    # 若主点过于靠边，认为无效，交给上游跳过
    if not (min_margin_x > W / 5 and min_margin_y > H / 5):
        raise SkipSample("Principal point too close to image boundary.")

    l, t = cx - min_margin_x, cy - min_margin_y
    r, b = cx + min_margin_x, cy + min_margin_y
    crop_bbox = (int(l), int(t), int(r), int(b))
    image, depthmap, intrinsics = cropping.crop_image_depthmap(
        image, depthmap, intrinsics, crop_bbox
    )

    # High-quality Lanczos down-scaling
    target_resolution = np.array(resolution)
    image, depthmap, intrinsics = cropping.rescale_image_depthmap(
        image, depthmap, intrinsics, target_resolution
    )

    # Actual cropping (if necessary) with bilinear interpolation
    intrinsics2 = cropping.camera_matrix_of_crop(
        intrinsics, image.size, resolution, offset_factor=0.5
    )
    crop_bbox = cropping.bbox_from_intrinsics_in_out(intrinsics, intrinsics2, resolution)
    image, depthmap, intrinsics2 = cropping.crop_image_depthmap(
        image, depthmap, intrinsics, crop_bbox
    )

    return image, depthmap, intrinsics2


def _maybe_align_view_to_resolution(view: Dict[str, Any], resolution: Tuple[int, int]):
    """
    If depth/image/K are not aligned to the desired resolution,
    try to align them once. If anything goes wrong, we let the caller decide.
    """
    Ht, Wt = resolution[1], resolution[0]  # PIL uses (W, H)
    img_pil = _to_pil(view["original_img"])
    W, H = img_pil.size

    if (W, H) == (resolution[0], resolution[1]):
        # resolution already matches
        return view

    K = view.get("intrinsics") or view.get("K") or view.get("camera_intrinsics")
    depth = view.get("depthmap", None)

    # If we lack intrinsics or depth, can't align geometrically here; leave as-is
    if K is None or depth is None:
        view["original_img"] = img_pil.resize((resolution[0], resolution[1]), PIL.Image.BILINEAR)
        return view

    try:
        img2, depth2, K2 = crop_resize_if_necessary(img_pil, depth, np.array(K), resolution)
        view["original_img"] = img2
        view["depthmap"] = depth2
        # write back intrinsics using a common key 'intrinsics'
        view["intrinsics"] = np.array(K2, dtype=np.float32)
    except SkipSample:
        raise
    except Exception as e:
        # If alignment fails, we will let downstream compute mask; not fatal
        LOG.warning(f"[align] failed to align view to {resolution}: {e}")
        view["original_img"] = img_pil.resize((resolution[0], resolution[1]), PIL.Image.BILINEAR)

    return view


def _maybe_add_geometry(view: Dict[str, Any]) -> Dict[str, Any]:
    """
    Try to build pts3d and valid_mask if depth/intrinsics exist.
    If not possible, leave them as None; caller may decide to skip.
    """
    try:
        pts3d, valid_mask = depthmap_to_absolute_camera_coordinates(**view)
        view["pts3d"] = pts3d
        # Ensure finite points only
        view["valid_mask"] = valid_mask & np.isfinite(pts3d).all(axis=-1)
    except Exception as e:
        LOG.debug(f"[geom] geometry unavailable for view (non-fatal): {e}")
        view["pts3d"] = None
        view["valid_mask"] = None
    return view


def _build_img_tensors(view: Dict[str, Any], img_norm_callable, to_tensor):
    """
    Produce normalized network input and keep a clean float tensor copy of original image.
    """
    img_pil = _to_pil(view["original_img"])
    view["img"] = img_norm_callable(img_pil)  # network input (DUST3R style)
    view["original_img"] = to_tensor(img_pil)  # CHW float [0,1]
    return view


def _safe_imgnorm():
    """
    Instantiate ImgNorm if it's a class; otherwise return the callable as-is.
    """
    try:
        return ImgNorm()
    except TypeError:
        return ImgNorm


# -----------------------------------------------------------------------------
# Training Dataset
# -----------------------------------------------------------------------------
class DUST3RSplattingDataset(torch.utils.data.Dataset):
    """
    Training dataset that samples 2 context + N target views per sequence,
    robust to corrupted or geometry-missing samples.
    """

    def __init__(
        self,
        data,
        coverage,
        resolution: Tuple[int, int],
        num_epochs_per_epoch: int = 1,
        alpha: float = 0.3,
        beta: float = 0.3,
        max_retries_per_item: int = 8,
        require_nonempty_mask: bool = True,
    ):
        super().__init__()
        self.data = data
        self.coverage = coverage

        self.num_context_views = 2
        self.num_target_views = 3

        self.resolution = resolution
        self.transform = _safe_imgnorm()
        self.org_transform = torchvision.transforms.ToTensor()
        self.num_epochs_per_epoch = num_epochs_per_epoch

        self.alpha = alpha
        self.beta = beta

        self.max_retries_per_item = max_retries_per_item
        self.require_nonempty_mask = require_nonempty_mask

    def __len__(self):
        return len(self.data.sequences) * self.num_epochs_per_epoch

    # ------------------------------- sampling -------------------------------
    def sample(
        self,
        sequence: str,
        num_target_views: int,
        context_overlap_threshold: float = 0.5,
        target_overlap_threshold: float = 0.6,
    ):
        first_context_view = random.randint(0, len(self.data.color_paths[sequence]) - 1)

        # choose second context
        valid_second = [
            f
            for f in range(len(self.data.color_paths[sequence]))
            if f != first_context_view
            and self.coverage[sequence][first_context_view][f] > context_overlap_threshold
        ]
        if len(valid_second) > 0:
            second_context_view = random.choice(valid_second)
        else:
            # best fallback
            best_view, best_overlap = None, None
            for f in range(len(self.data.color_paths[sequence])):
                if f == first_context_view:
                    continue
                ov = self.coverage[sequence][first_context_view][f]
                if best_view is None or ov > best_overlap:
                    best_view, best_overlap = f, ov
            second_context_view = best_view

        # choose targets
        valid_targets = []
        for f in range(len(self.data.color_paths[sequence])):
            if f in (first_context_view, second_context_view):
                continue
            ov_max = max(
                self.coverage[sequence][first_context_view][f],
                self.coverage[sequence][second_context_view][f],
            )
            if ov_max > target_overlap_threshold:
                valid_targets.append(f)

        if len(valid_targets) >= num_target_views:
            target_views = random.sample(valid_targets, num_target_views)
        else:
            # top-k fallback
            cand = []
            for f in range(len(self.data.color_paths[sequence])):
                if f in (first_context_view, second_context_view):
                    continue
                ov = max(
                    self.coverage[sequence][first_context_view][f],
                    self.coverage[sequence][second_context_view][f],
                )
                cand.append((f, ov))
            cand.sort(key=lambda x: x[1], reverse=True)
            target_views = [f for f, _ in cand[:num_target_views]]

        return [first_context_view, second_context_view], target_views

    # ------------------------------- getters --------------------------------
    def _fetch_view(self, sequence: str, view_idx: int) -> Dict[str, Any]:
        seq_len = len(self.data.color_paths[sequence])
        if view_idx >= seq_len:
            raise SkipSample(
                f"view index out of range: seq={sequence} idx={view_idx} len={seq_len}"
            )
        view = self.data.get_view(sequence, view_idx, self.resolution)

        # (optional) align image/depth/K to target resolution
        try:
            view = _maybe_align_view_to_resolution(view, self.resolution)
        except SkipSample as e:
            raise e

        # Build tensors
        view = _build_img_tensors(view, self.transform, self.org_transform)

        # Geometry (if available)
        view = _maybe_add_geometry(view)

        if self.require_nonempty_mask and (view.get("valid_mask", None) is not None):
            if not np.asarray(view["valid_mask"]).any():
                raise SkipSample(
                    f"no valid mask after geometry: seq={sequence} idx={view_idx}"
                )

        return view

    # ------------------------------ __getitem__ ------------------------------
    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        tries = 0
        n_seq = len(self.data.sequences)
        # map global idx -> sequence
        sequence = self.data.sequences[idx // self.num_epochs_per_epoch]

        while tries < self.max_retries_per_item:
            tries += 1
            try:
                ctx_views, tgt_views = self.sample(
                    sequence, self.num_target_views, self.alpha, self.beta
                )

                views = {"context": [], "target": [], "scene": sequence}

                # fetch context
                for c_idx in ctx_views:
                    v = self._fetch_view(sequence, c_idx)
                    views["context"].append(v)

                # fetch targets
                for t_idx in tgt_views:
                    v = self.data.get_view(sequence, t_idx, self.resolution)
                    # (optional) align only image to resolution for targets
                    v = _maybe_align_view_to_resolution(v, self.resolution)
                    v["original_img"] = self.org_transform(_to_pil(v["original_img"]))
                    views["target"].append(v)

                return views

            except SkipSample as e:
                LOG.warning(f"[skip] seq={sequence} idx={idx} try={tries}: {e}")
                # try next index in the same sequence range to avoid infinite loop
                idx = (idx + 1) % max(1, len(self))
                sequence = self.data.sequences[idx // self.num_epochs_per_epoch]
            except Exception as e:
                LOG.exception(f"[error] seq={sequence} idx={idx} try={tries}: {e}")
                idx = (idx + 1) % max(1, len(self))
                sequence = self.data.sequences[idx // self.num_epochs_per_epoch]

        # give up
        return None


# -----------------------------------------------------------------------------
# Test / Eval Dataset (fixed samples list)
# -----------------------------------------------------------------------------
class DUST3RSplattingTestDataset(torch.utils.data.Dataset):
    """
    Test dataset driven by a list of (sequence, c1, c2, t) tuples.
    """

    def __init__(self, data, samples: Sequence[Tuple[str, int, int, int]], resolution):
        self.data = data
        self.samples = list(samples)
        self.resolution = resolution
        self.transform = _safe_imgnorm()
        self.org_transform = torchvision.transforms.ToTensor()
        self.max_retries_per_item = 4
        self.require_nonempty_mask = True

    def __len__(self):
        return len(self.samples)

    def _fetch_context(self, sequence: str, view_idx: int) -> Dict[str, Any]:
        view = self.data.get_view(sequence, view_idx, self.resolution)
        view = _maybe_align_view_to_resolution(view, self.resolution)
        view = _build_img_tensors(view, self.transform, self.org_transform)
        view = _maybe_add_geometry(view)

        if self.require_nonempty_mask and (view.get("valid_mask", None) is not None):
            if not np.asarray(view["valid_mask"]).any():
                raise SkipSample(
                    f"test: no valid mask seq={sequence} view={view_idx}"
                )
        return view

    def _fetch_target(self, sequence: str, view_idx: int) -> Dict[str, Any]:
        view = self.data.get_view(sequence, view_idx, self.resolution)
        view = _maybe_align_view_to_resolution(view, self.resolution)
        view["original_img"] = self.org_transform(_to_pil(view["original_img"]))
        return view

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        tries = 0
        while tries < self.max_retries_per_item:
            tries += 1
            try:
                sample = self.samples[idx]
                if len(sample) == 5:
                    sequence, c1, c2, t, asset_name = sample
                else:
                    sequence, c1, c2, t = sample
                    asset_name = f"{sequence}_pair_{c1}_{c2}_t_{t}"
                c1, c2, t = int(c1), int(c2), int(t)

                v1 = self._fetch_context(sequence, c1)
                v2 = self._fetch_context(sequence, c2)
                vt = self._fetch_target(sequence, t)

                return {
                    "context": [v1, v2],
                    "target": [vt],
                    "scene": sequence,
                    "asset_name": asset_name,
                }

            except SkipSample as e:
                LOG.warning(f"[skip-test] idx={idx} try={tries}: {e}")
                idx = (idx + 1) % max(1, len(self))
            except Exception as e:
                LOG.exception(f"[error-test] idx={idx} try={tries}: {e}")
                idx = (idx + 1) % max(1, len(self))
        return None


# -----------------------------------------------------------------------------
# Safe collate: drop None, return {} if batch fully corrupted
# -----------------------------------------------------------------------------
def collate_fn_skip_corrupted(batch: List[Optional[Dict[str, Any]]]):
    """
    Filter out None items. If batch becomes empty, return {} so upper loops can skip.
    """
    batch = [b for b in batch if b is not None]
    if not batch:
        return {}
    return torch.utils.data.dataloader.default_collate(batch)
