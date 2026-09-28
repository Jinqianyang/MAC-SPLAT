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

LOG = logging.getLogger(__name__)
if not LOG.handlers:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


class SkipSample(Exception):
    pass


def _to_pil(image):
    if isinstance(image, PIL.Image.Image):
        return image
    if isinstance(image, torch.Tensor):
        if image.ndim == 3 and image.shape[0] in (1, 3):
            image = torchvision.transforms.ToPILImage()(image)
        elif image.ndim == 3 and image.shape[-1] in (1, 3):
            image = torchvision.transforms.ToPILImage()(image.permute(2, 0, 1))
        else:
            image = torchvision.transforms.ToPILImage()(image)
        return image
    if isinstance(image, np.ndarray):
        return PIL.Image.fromarray(image)
    return PIL.Image.fromarray(np.array(image))


def crop_resize_if_necessary(
    image, depthmap, intrinsics, resolution: Tuple[int, int]
):
    image = _to_pil(image)

    W, H = image.size
    cx, cy = intrinsics[:2, 2].round().astype(int)
    min_margin_x = min(cx, W - cx)
    min_margin_y = min(cy, H - cy)
    if not (min_margin_x > W / 5 and min_margin_y > H / 5):
        raise SkipSample("Principal point too close to image boundary.")

    l, t = cx - min_margin_x, cy - min_margin_y
    r, b = cx + min_margin_x, cy + min_margin_y
    crop_bbox = (int(l), int(t), int(r), int(b))
    image, depthmap, intrinsics = cropping.crop_image_depthmap(
        image, depthmap, intrinsics, crop_bbox
    )

    target_resolution = np.array(resolution)
    image, depthmap, intrinsics = cropping.rescale_image_depthmap(
        image, depthmap, intrinsics, target_resolution
    )

    intrinsics2 = cropping.camera_matrix_of_crop(
        intrinsics, image.size, resolution, offset_factor=0.5
    )
    crop_bbox = cropping.bbox_from_intrinsics_in_out(intrinsics, intrinsics2, resolution)
    image, depthmap, intrinsics2 = cropping.crop_image_depthmap(
        image, depthmap, intrinsics, crop_bbox
    )

    return image, depthmap, intrinsics2


def _maybe_align_view_to_resolution(view: Dict[str, Any], resolution: Tuple[int, int]):
    Ht, Wt = resolution[1], resolution[0]
    img_pil = _to_pil(view["original_img"])
    W, H = img_pil.size

    if (W, H) == (resolution[0], resolution[1]):
        return view

    K = view.get("intrinsics") or view.get("K") or view.get("camera_intrinsics")
    depth = view.get("depthmap", None)

    if K is None or depth is None:
        view["original_img"] = img_pil.resize((resolution[0], resolution[1]), PIL.Image.BILINEAR)
        return view

    try:
        img2, depth2, K2 = crop_resize_if_necessary(img_pil, depth, np.array(K), resolution)
        view["original_img"] = img2
        view["depthmap"] = depth2
        view["intrinsics"] = np.array(K2, dtype=np.float32)
    except SkipSample:
        raise
    except Exception as e:
        LOG.warning(f"[align] failed to align view to {resolution}: {e}")
        view["original_img"] = img_pil.resize((resolution[0], resolution[1]), PIL.Image.BILINEAR)

    return view


def _maybe_add_geometry(view: Dict[str, Any]) -> Dict[str, Any]:
    try:
        pts3d, valid_mask = depthmap_to_absolute_camera_coordinates(**view)
        view["pts3d"] = pts3d
        view["valid_mask"] = valid_mask & np.isfinite(pts3d).all(axis=-1)
    except Exception as e:
        LOG.debug(f"[geom] geometry unavailable for view (non-fatal): {e}")
        view["pts3d"] = None
        view["valid_mask"] = None
    return view


def _build_img_tensors(view: Dict[str, Any], img_norm_callable, to_tensor):
    img_pil = _to_pil(view["original_img"])
    view["img"] = img_norm_callable(img_pil)
    view["original_img"] = to_tensor(img_pil)
    return view


def _safe_imgnorm():
    try:
        return ImgNorm()
    except TypeError:
        return ImgNorm


class DUST3RSplattingDataset(torch.utils.data.Dataset):

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

    def sample(
        self,
        sequence: str,
        num_target_views: int,
        context_overlap_threshold: float = 0.5,
        target_overlap_threshold: float = 0.6,
    ):
        first_context_view = random.randint(0, len(self.data.color_paths[sequence]) - 1)

        valid_second = [
            f
            for f in range(len(self.data.color_paths[sequence]))
            if f != first_context_view
            and self.coverage[sequence][first_context_view][f] > context_overlap_threshold
        ]
        if len(valid_second) > 0:
            second_context_view = random.choice(valid_second)
        else:
            best_view, best_overlap = None, None
            for f in range(len(self.data.color_paths[sequence])):
                if f == first_context_view:
                    continue
                ov = self.coverage[sequence][first_context_view][f]
                if best_view is None or ov > best_overlap:
                    best_view, best_overlap = f, ov
            second_context_view = best_view

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

    def _fetch_view(self, sequence: str, view_idx: int) -> Dict[str, Any]:
        seq_len = len(self.data.color_paths[sequence])
        if view_idx >= seq_len:
            raise SkipSample(
                f"view index out of range: seq={sequence} idx={view_idx} len={seq_len}"
            )
        view = self.data.get_view(sequence, view_idx, self.resolution)

        try:
            view = _maybe_align_view_to_resolution(view, self.resolution)
        except SkipSample as e:
            raise e

        view = _build_img_tensors(view, self.transform, self.org_transform)

        view = _maybe_add_geometry(view)

        if self.require_nonempty_mask and (view.get("valid_mask", None) is not None):
            if not np.asarray(view["valid_mask"]).any():
                raise SkipSample(
                    f"no valid mask after geometry: seq={sequence} idx={view_idx}"
                )

        return view

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        tries = 0
        n_seq = len(self.data.sequences)
        sequence = self.data.sequences[idx // self.num_epochs_per_epoch]

        while tries < self.max_retries_per_item:
            tries += 1
            try:
                ctx_views, tgt_views = self.sample(
                    sequence, self.num_target_views, self.alpha, self.beta
                )

                views = {"context": [], "target": [], "scene": sequence}

                for c_idx in ctx_views:
                    v = self._fetch_view(sequence, c_idx)
                    views["context"].append(v)

                for t_idx in tgt_views:
                    v = self.data.get_view(sequence, t_idx, self.resolution)
                    v = _maybe_align_view_to_resolution(v, self.resolution)
                    v["original_img"] = self.org_transform(_to_pil(v["original_img"]))
                    views["target"].append(v)

                return views

            except SkipSample as e:
                LOG.warning(f"[skip] seq={sequence} idx={idx} try={tries}: {e}")
                idx = (idx + 1) % max(1, len(self))
                sequence = self.data.sequences[idx // self.num_epochs_per_epoch]
            except Exception as e:
                LOG.exception(f"[error] seq={sequence} idx={idx} try={tries}: {e}")
                idx = (idx + 1) % max(1, len(self))
                sequence = self.data.sequences[idx // self.num_epochs_per_epoch]

        return None


class DUST3RSplattingTestDataset(torch.utils.data.Dataset):

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


def collate_fn_skip_corrupted(batch: List[Optional[Dict[str, Any]]]):
    batch = [b for b in batch if b is not None]
    if not batch:
        return {}
    return torch.utils.data.dataloader.default_collate(batch)
