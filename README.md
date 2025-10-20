[![header](https://capsule-render.vercel.app/api?type=rect&height=120&color=gradient&text=COVER%20-%20Vector%20Contrastive%20Learning&fontAlign=50&reversal=true&textBg=false&fontAlignY=37&fontSize=40&desc=For%20Pixel-Wise%20Pretraining%20In%20Medical%20Vision&descSize=40&descAlign=50&descAlignY=75)](https://arxiv.org/abs/2506.20850)
---
[![Paper](https://img.shields.io/badge/ICCV-Conference-purple)](https://arxiv.org/abs/2506.20850)

> **ICCV 2025**

> A new paradigm for pixel-wise pretraining — moving from **binary contrast** to **vector regression**.

> 🧩 *Learning in vector-like relationship for pixel-wise features.*

<p align="center"><img width="80%" src="fig/fig.png" /></p>

**Vector Contrastive Learning (VCL)** is a new contrastive framework that learns *vector-based relationships* between features instead of binary “close/far” classification. It’s designed for **pixel-wise self-supervised pretraining**, especially in **medical vision** (2D/3D, multi-modal imaging).

---

> [**Vector Contrastive Learning For Pixel-Wise Pretraining In Medical Vision**](https://arxiv.org/abs/2506.20850),  
> [Yuting He](https://yutinghe-list.github.io/), [Shuo Li](https://scholar.google.com/citations?user=6WNtJa0AAAAJ&hl=en)*,  
> In: Proc. International Conference on Computer Vision (ICCV), 2025,  
> *arXiv preprint ([arXiv 2506.20850](https://arxiv.org/abs/2506.20850))*

## 🚀 Highlights

- 🔄 **Vector Regression instead of Binary Contrast**  
  Predict displacement vectors between pixel-wise features rather than classifying them as positive or negative.

- 🧩 **COVER Framework (COntrast in VEctor Regression)**  
  A unified optimization flow combining vector regression and distance modeling.

- 🌐 **Multi-Scale Vector Pyramid**  
  Learns consistent pixel-wise representation across different spatial resolutions.

- 🧬 **Strong Pixel-Level Representation**  
  Outperforms existing pixel-wise SSL methods across **8 tasks** and **4 medical imaging modalities**.

---
## 🚀 Quick Start

### 1️⃣ Installation

```bash
git clone https://github.com/YutingHe-list/COVER.git
cd COVER
conda create -n cover python=3.9
conda activate cover
pip install -r requirements.txt

## 🛣️ Roadmap

We’re continuously improving **COVER** to make it more general, modular, and scalable for diverse pixel/voxel-wise self-supervised learning tasks.

| Status | Feature / Goal | Description |
|:------:|:----------------|:-------------|
| ✅ | **COVER 2D Implementation** | Core framework for 2D pixel-wise vector contrastive pretraining |
| ✅ | **COVER Loss Functions** | Includes vector regression |
| 🧩 | **COVER 3D Support** | Extend to 3D volumetric (CT/MRI) pretraining with voxel-level displacement regression |
| 🔜 | **Pre-trained Weights Release** | Release pretrained models on ChestXray datasets |
| 🔜 | **Benchmark Suite** | Provide unified training and evaluation scripts for downstream tasks |
| 🚧 | **Multi-Modal Extension** | Extend COVER to PET/CT and MRI/US cross-modal learning |
| 🚧 | **Video Representation Learning** | Apply COVER’s vector regression to temporal frame relationships |
| 💡 | **Natural Image Experiments** | Generalize COVER beyond medical imaging to natural datasets |
| 🧠 | **Hugging Face Hub** | Host pretrained checkpoints and demo notebooks for community usage |

---

### 🎯 Vision

> **From binary contrast to vector understanding.**

> Our long-term goal is to establish a unified *vector-based* contrastive framework that bridges spatial geometry and semantics — applicable across 2D, 3D, and multimodal domains.

---

