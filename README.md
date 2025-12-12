 # 📢 About This Repository
In this study, we propose UniVG, a novel generative foundation model for universal few-shot vascular image segmentation based on compositional learning and few-shot generative adaptation. By decomposing and recombining vascular structures with diverse backgrounds, our framework can synthesize highly realistic and diverse vessel images using minimal training data while achieving performance comparable to fully supervised approaches. UniVG has been systematically validated across 11 vascular datasets spanning five different modalities. Experimental results demonstrate its superior performance over existing methods in few-shot scenarios (using as few as five annotated images), confirming the model's excellent generalization capability and cross-domain adaptability.

---
![Methods](fig/method.png)

We have established this repository to support the **reproducibility** of our work:  
**"Generative Data-engine Foundation Model for Universal Few-shot 2D Vascular Image Segmentation"**

We are committed to releasing the complete codebase incrementally following manuscript acceptance.

---
# 🎯 Currently Available
## Dataset
The UniVG-58K dataset presented in this paper comprises both pre-training data and downstream task data, which can be accessed at https://huggingface.co/datasets/xinaloha/UniVG.

# R-SCA

| Component | Status | Release Date |
|-----------|--------|--------------|
| Spatial Colonization Algorithm (SCA) for vascular structure synthesis |✅ **Available** | **2025.12.12** |

---

## 📅 Planned Release Timeline

| Component | Status | Expected Release |
|-----------|--------|------------------|
| Pre-trained Foundation Model & Code | 📦 Coming Soon | **2025.1.10** |
| Downstream Modality Fine-tuning Code | 📦 Coming Soon | **2025.1.30** |

---

# 🚀 Quick Start: Spatial Colonization Algorithm (SCA)

## Installation

```bash
# Clone the repository
git clone https://github.com/XinAloha/UniVG.git
cd UniVG/R-SCA

# Install dependencies
pip install numpy opencv-python Pillow matplotlib scipy shapely PyYAML tqdm
```

## Usage

Generate synthetic vascular images:

```bash
python Main.py --input-dir ./RealCoronaryArteryMask --output-dir ./output --modality CoronaryArtery
```

 

