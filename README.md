# MAC-Splat

Code for **MAC-Splat: Multi-Attribute Consistency for High-Fidelity Sparse-View Reconstruction**. [Paper](https://arxiv.org/abs/2607.10792).

MAC-Splat combines Gaussian prediction, semantic fusion, and multi-attribute consistency for sparse-view reconstruction.

| Location | Contents |
| --- | --- |
| `main.py` | Model assembly, training/evaluation hooks, and MAC loss |
| `configs/main.yaml` | Model, data, and optimization configuration |
| `src/mast3r_src/` | Model heads and MASt3R/DUSt3R/CroCo source |
| `src/pixelsplat_src/` | Renderer and timing utility |
| `data/` | Dataset-loading code |
| `utils/` | Matching, geometry, metrics, masking, and export |

Install PyTorch and torchvision for your CUDA environment, then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Compile the optional CroCo RoPE extension:

```bash
cd src/mast3r_src/dust3r/croco/models/curope
python setup.py build_ext --inplace
cd ../../../../../..
```

Set `pretrained_mast3r_path` to a Lightning checkpoint containing an `encoder.`-prefixed `state_dict`, `DINOV3_LOCAL_DIR` to the local DINO model directory, and `data.root` to the ScanNet++ directory.

The dataset layout is defined in [data/scannetpp/scannetpp.py](data/scannetpp/scannetpp.py). Run:

```bash
python main.py configs/main.yaml \
  pretrained_mast3r_path=/path/to/model.ckpt \
  data.root=/path/to/scannetpp
```

We thank the authors of [Splatt3R](https://github.com/btsmart/splatt3r) and [MASt3R](https://github.com/naver/mast3r) for sharing their code. MAC-Splat adapts their implementations and incorporates code from [DUSt3R](https://github.com/naver/dust3r), [CroCo](https://github.com/naver/croco), [pixelSplat](https://github.com/dcharatan/pixelsplat), and [PyTorch3D](https://github.com/facebookresearch/pytorch3d).

License texts are provided in [License](License), the component `LICENSE` and `NOTICE` files, and [licenses/](licenses/). File-specific notices and source references are retained in [licenses/SOURCE-NOTICES.txt](licenses/SOURCE-NOTICES.txt).
