<p>
  <h2 align="center">Steerable Visual Representations</h2>
  <p align="center">
    <a href="https://jonaruthardt.github.io" target="_blank">Jona Ruthardt</a><sup>1,*</sup>
    ·
    <a href="https://manugaurdl.github.io" target="_blank">Manu Gaur</a><sup>2,*</sup>
    ·
    <a href="https://www.cs.cmu.edu/~deva/" target="_blank">Deva Ramanan</a><sup>2</sup>
    ·
    <a href="https://scholar.google.com/citations?user=rJotb-YAAAAJ" target="_blank">Makarand Tapaswi</a><sup>3,†</sup>
    ·
    <a href="https://yukimasano.github.io" target="_blank">Yuki M. Asano</a><sup>1,†</sup>
  </p>
</p>

<p align="center">
  <sup>1</sup>University of Technology Nuremberg
  &nbsp;&nbsp;|&nbsp;&nbsp;
  <sup>2</sup>Carnegie Mellon University
  &nbsp;&nbsp;|&nbsp;&nbsp;
  <sup>3</sup>IIIT Hyderabad
</p>

<p align="center">
  <sup>*</sup>Equal contribution
  &nbsp;&nbsp;|&nbsp;&nbsp;
  <sup>†</sup>Equal advising
</p>

<p align="center">
  <a href="https://jonaruthardt.github.io/project/SteerViT/">
    <img src="https://img.shields.io/badge/Project-Website-0A66C2.svg" alt="Project Website">
  </a>
  <a href="https://arxiv.org/abs/2604.02327">
    <img src="https://img.shields.io/badge/arXiv-2604.02327-b31b1b.svg" alt="arXiv">
  </a>
  <a href="https://colab.research.google.com/drive/1Lf-95znqXaWUGyY9bPeJQI9Eq7yBcpFl?usp=sharing">
    <img src="https://img.shields.io/badge/Colab-Demo-F9AB00.svg?logo=googlecolab&logoColor=white" alt="Colab Demo">
  </a>
  <a href="https://huggingface.co/spaces/JonaRuthardt/SteerViT">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue" alt="Hugging Face Spaces">
  </a>
  <a href="https://huggingface.co/JonaRuthardt/SteerViT">
    <img src="https://img.shields.io/badge/Hugging%20Face-Model%20Weights-F9AB00.svg?logo=huggingface&logoColor=yellow" alt="Hugging Face Weights">
  </a>
  <a href="https://opensource.org/license/mit">
    <img src="https://img.shields.io/badge/License-MIT-red.svg" alt="MIT License">
  </a>
</p>

<p align="center">
  <img src="others/teaser_fig.png" alt="SteerViT qualitative example" width="100%">
</p>

SteerViT equips pretrained Vision Transformers with **steerable global and local visual representations**. Given an image and a natural-language prompt, the model conditions the visual encoder itself through lightweight gated cross-attention, producing prompt-aware patch features, global embeddings, and heatmaps while retaining the strengths of the pretrained ViT backbone.

> 🔔 Model checkpoints and inference code are available already. Full training and evaluation code will be released soon.

## ✈️ Overview

SteerViT turns any pretrained ViT into a **query-aware visual encoder** by injecting text directly into the visual backbone rather than only fusing text after image encoding. This makes it possible to steer:

- which regions contribute to the representation,
- what the global embedding encodes,
- the semantic granularity of the representation,
- and provides dense prompt-conditioned localization signals.

Across the project page demos and paper results, SteerViT is shown on conditional retrieval, prompt-guided attention, semantic control, and zero-shot anomaly segmentation.

## ⚙️ Installation

SteerViT requires Python `>=3.10`.

For regular use, install directly from GitHub:

```bash
python -m pip install "git+https://github.com/JonaRuthardt/SteerViT.git"
```

For development, clone the repository and install it in editable mode:

```bash
git clone https://github.com/JonaRuthardt/SteerViT.git
cd SteerViT

conda create -n steervit python=3.10
conda activate steervit
python -m pip install -e .
```

## 🏎️ Quick Start

```python
import torch
from PIL import Image

from steervit import SteerViT

device = "cuda" if torch.cuda.is_available() else "cpu"

model = SteerViT.from_pretrained("steervit_dinov2_base.pth", device=device)
transform = model.get_transforms()

image = Image.open("path/to/image.jpg").convert("RGB")
image_tensor = transform(image).unsqueeze(0)

prompt = ["the red car"]

global_features = model.get_global_features(image_tensor, texts=prompt)
dense_features = model.get_dense_features(image_tensor, texts=prompt)
heatmaps = model.get_heatmaps(image_tensor, texts=prompt)
attention_heatmaps = model.get_attention_heatmaps(image_tensor, texts=prompt)
```

If `texts=None`, SteerViT behaves like the underlying frozen ViT backbone and returns query-agnostic features.

## 📋 Available Checkpoints

The released demo currently uses the following checkpoint identifiers:

| Checkpoint | `from_pretrained(...)` identifier | Notes |
| --- | --- | --- |
| SteerDINOv2-Base | `steervit_dinov2_base.pth` | Model used for most experiments in paper |
| SteerMAE-Base | `steervit_mae_base.pth` | Alternative model based on MAE-backbone |

`SteerViT.from_pretrained(...)` accepts either:

- a local checkpoint path, or
- a checkpoint filename hosted on [🤗 Hugging Face](https://huggingface.co/JonaRuthardt/SteerViT)

## 📑 `SteerViT` API

The main public entry point is:

```python
from steervit import SteerViT
```

### Construction and Loading

| API | Description |
| --- | --- |
| `SteerViT(config)` | Low-level constructor for manual model creation from a config dictionary. Most users should use `from_pretrained(...)`. |
| `SteerViT.from_pretrained(checkpoint_name, device=None)` | Loads a released checkpoint from a local path or from the Hugging Face repo and returns an eval-mode model with frozen parameters. |

### Core Inference Methods

| API | Returns | Description |
| --- | --- | --- |
| `model.get_dense_features(images, texts=None)` | `(B, N_patches, D)` | Returns patch-level visual features, excluding prefix tokens such as `[CLS]`. |
| `model.get_global_features(images, texts=None)` | `(B, D)` | Returns pooled image embeddings. Pooling is determined by the checkpoint config (`cls` or mean pooling). |
| `model.get_heatmaps(images, texts=None)` | `(B, 1, H, W)` | Returns prompt-conditioned dense heatmaps from the learned linear segmentation head. |
| `model.get_attention_heatmaps(images, texts=None, **kwargs)` | `(B, H, W)` by default | Returns CLS-attention heatmaps extracted from the ViT attention maps. |

### Utilities and Properties

| API | Description |
| --- | --- |
| `model.get_transforms()` | Returns the `timm` preprocessing pipeline matching the loaded backbone. |
| `model.set_gate_factor(factor)` | Scales the learned cross-attention gates at inference time. `0.0` approximates the original frozen ViT behavior, while `1.0` uses the learned steering strength. |
| `model.patch_size` | Patch size of the vision backbone. |
| `model.feature_dim` | Feature dimension of the visual token embeddings. |
| `model.image_size` | Input resolution expected by the loaded checkpoint. |

## 📽️ Demo

For an interactive walkthrough, use one of the following:

- [Colab](https://colab.research.google.com/drive/1Lf-95znqXaWUGyY9bPeJQI9Eq7yBcpFl?usp=sharing)
- [Local notebook](demo.ipynb)

The current release focuses on inference and qualitative exploration. Training and full evaluation pipelines will be added in a future code release.

## BibTeX

If you use SteerViT in your research, please cite:

```bibtex
@misc{ruthardt2026steervit,
      title={Steerable Visual Representations}, 
      author={Jona Ruthardt and Manu Gaur and Deva Ramanan and Makarand Tapaswi and Yuki M. Asano},
      journal={arXiv:2604.02327},
      year={2026}
}
```
