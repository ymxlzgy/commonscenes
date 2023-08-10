# CommonScenes

This is the official implementation of the paper **CommonScenes: Generating Commonsense 3D Indoor Scenes with Scene Graphs**

<a href="https://sites.google.com/view/commonscenes">Website</a> | <a href="https://arxiv.org/pdf/2305.16283.pdf">Arxiv</a>

Guangyao Zhai, Evin Pınar Örnek, Shun-Cheng Wu, Yan Di, Federico Tombari, Nassir Navab, and Benjamin Busam<br/>
**Arxiv Preprint**

We propose a method to enable a fully generative model of the entire 3D scene from scene graphs, encompassing its layout and 3D geometries holistically. 

If you find this code useful in your research, please cite
```
@article{zhai2023commonscenes,
  title={CommonScenes: Generating Commonsense 3D Indoor Scenes with Scene Graphs},
  author={Zhai, Guangyao and {\"O}rnek, Evin Pinar and Wu, Shun-Cheng and Di, Yan and Tombari, Federico and Navab, Nassir and Busam, Benjamin},
  journal={arXiv preprint arXiv:2305.16283},
  year={2023}
```

## Setup

We have tested it on Ubuntu 20.04 with Python 3.8, PyTorch 1.11.0, CUDA 11.3 and [Pytorch3D](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md#3-install-wheels-for-linux).

```bash
TBD
```

To install CLIP, follow this <a href="[https://github.com/TheoDEPRELLE/AtlasNetV2](https://github.com/openai/CLIP)">OpenAI CLIP repo</a>:
```
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```

### Dataset

1. Download the <a href="https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset">3D-FRONT dataset</a> from their official site.

2. Preprocess the dataset following <a href="https://github.com/tangjiapeng/DiffuScene#pickle-the-3d-future-dataset">Diffuscene</a> or  <a href="https://github.com/nv-tlabs/ATISS#data-preprocessing">ATISS</a>.

3. Download our SG-FRONT dataset.[TBD]

## Training

To train our model, run:

```
TBD
```

## Evaluation

To evaluate the models run
```
TBD
```
Set `--evaluate_diversity` to `True` if you want to compute diversity. This takes a while, so it's disabled by default.

## Acknowledgements

This repository contains code parts that are based on <a href="https://github.com/he-dhamo/graphto3d">Graph-to-3D</a> and <a href="https://github.com/yccyenchicheng/SDFusion">SDFusion</a>. We thank the authors for making their code available.
