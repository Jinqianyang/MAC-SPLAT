import os.path as osp
import numpy as np

from src.mast3r_src.dust3r.dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from src.mast3r_src.dust3r.dust3r.utils.image import imread_cv2


class BlendedMVS (BaseStereoViewDataset):

    def __init__(self, *args, ROOT, split=None, **kwargs):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        self._load_data(split)

    def _load_data(self, split):
        pairs = np.load(osp.join(self.ROOT, 'blendedmvs_pairs.npy'))
        if split is None:
            selection = slice(None)
        if split == 'train':
            selection = (pairs['seq_low'] % 10) > 0
        if split == 'val':
            selection = (pairs['seq_low'] % 10) == 0
        self.pairs = pairs[selection]

        self.scenes = np.unique(self.pairs['seq_low'])

    def __len__(self):
        return len(self.pairs)

    def get_stats(self):
        return f'{len(self)} pairs from {len(self.scenes)} scenes'

    def select_one_scene(self, scene, img1=None, img2=None):
        scene_low = int(scene[-16:], 16)
        valid = (self.pairs['seq_low'] == scene_low)
        if img1:
            valid &= (self.pairs['img1'] == int(img1))
        if img2:
            valid &= (self.pairs['img2'] == int(img2))
        self.pairs = self.pairs[valid]
        self._compute_pair_probas()

    def _get_views(self, pair_idx, resolution, rng):
        seqh, seql, img1, img2, score = self.pairs[pair_idx]

        seq = f"{seqh:08x}{seql:016x}"
        seq_path = osp.join(self.ROOT, seq)

        views = []

        for view_index in [img1, img2]:
            impath = f"{view_index:08n}"
            image = imread_cv2(osp.join(seq_path, impath + ".jpg"))
            depthmap = imread_cv2(osp.join(seq_path, impath + ".exr"))
            camera_params = np.load(osp.join(seq_path, impath + ".npz"))

            intrinsics = np.float32(camera_params['intrinsics'])
            camera_pose = np.eye(4, dtype=np.float32)
            camera_pose[:3, :3] = camera_params['R_cam2world']
            camera_pose[:3, 3] = camera_params['t_cam2world']

            image, depthmap, intrinsics = self._crop_resize_if_necessary(
                image, depthmap, intrinsics, resolution, rng, info=(seq_path, impath))

            views.append(dict(
                img=image,
                depthmap=depthmap,
                camera_pose=camera_pose,
                camera_intrinsics=intrinsics,
                dataset='Waymo',
                label=osp.relpath(seq_path, self.ROOT),
                instance=impath))

        return views


if __name__ == '__main__':
    from dust3r.datasets.base.base_stereo_view_dataset import view_name
    from dust3r.viz import SceneViz, auto_cam_size
    from dust3r.utils.image import rgb

    dataset = BlendedMVS(split='train', ROOT="data/blendedmvs_processed", resolution=224, aug_crop=16)

    for idx in np.random.permutation(len(dataset)):
        views = dataset[idx]
        assert len(views) == 2
        print(idx, view_name(views[0]), view_name(views[1]))
        viz = SceneViz()
        poses = [views[view_idx]['camera_pose'] for view_idx in [0, 1]]
        cam_size = max(auto_cam_size(poses), 0.001)
        for view_idx in [0, 1]:
            pts3d = views[view_idx]['pts3d']
            valid_mask = views[view_idx]['valid_mask']
            colors = rgb(views[view_idx]['img'])
            viz.add_pointcloud(pts3d, colors, valid_mask)
            viz.add_camera(pose_c2w=views[view_idx]['camera_pose'],
                           focal=views[view_idx]['camera_intrinsics'][0, 0],
                           color=(idx * 255, (1 - idx) * 255, 0),
                           image=colors,
                           cam_size=cam_size)
        viz.show()
