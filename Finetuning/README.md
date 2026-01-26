# UniVG Fine-tuning Guide

Fine-tune the UniVG pre-trained model using LoRA and LoHA methods.

---

## Environment Setup

```bash
# Clone SD-Trainer
git clone --recurse-submodules https://github.com/Akegarasu/lora-scripts
cd lora-scripts

# Windows
./install.ps1     
./run_gui.ps1

# Linux
bash install.bash
bash run_gui.sh
```

GUI URL: http://127.0.0.1:28000

---

## LoRA vs LoHA Comparison

| Feature | LoRA | LoHA |
|---------|------|------|
| Parameter Efficiency | Higher | Moderate |
| Expressiveness | Moderate | Stronger |
| Recommended For | Simple adaptation | Complex domain shift |
| Network Dim | 32 | 32 |
| Conv Dim | 32 | 4 |

---

## LoRA Fine-tuning

### Dataset Preparation

```
dataset/
├── 10_modality/          # Format: [repeats]_[trigger_word]
│   ├── image_001.png
│   ├── image_002.png
│   └── ... 
```

### Training Parameters

| Parameter | Value |
|-----------|-------|
| Network Type | `lora` |
| Network Dim | 32 |
| Network Alpha | 16 |
| Conv Dim | 32 |
| Conv Alpha | 16 |
| Learning Rate | 1e-4 |
| LR Scheduler | cosine |
| Batch Size | 2 |
| Training Steps | 3,000 |

### GUI Configuration

1. **Model Settings**: Load UniVG pre-trained model
2. **Network Settings**: 
   - Network Type: `lora`
   - Network Dim: `32`
   - Network Alpha: `16`
3. **Training Settings**:
   - Learning Rate: `1e-4`
   - Batch Size: `2`
   - Max Steps: `3000`
4. **Dataset**: Select image folder path
5. Click **Start Training**

---

## LoHA Fine-tuning (Paper Configuration)

### Dataset Preparation

Same as LoRA, place images in the corresponding folder.

### Training Parameters

| Parameter | Value |
|-----------|-------|
| Network Type | `loha` |
| Network Dim | 32 |
| Network Alpha | 32 |
| Conv Dim | 4 |
| Conv Alpha | 4 |
| Learning Rate | 1e-4 |
| LR Scheduler | cosine |
| Batch Size | 1 |
| Training Steps | 1,500 |

### GUI Configuration

1. **Model Settings**: Load UniVG pre-trained model
2. **Network Settings**: 
   - Network Type: `loha`
   - Network Dim: `32`
   - Network Alpha: `32`
   - Conv Dim: `4`
   - Conv Alpha: `4`
3. **Training Settings**:
   - Learning Rate: `1e-4`
   - LR Scheduler: `cosine`
   - Batch Size: `1`
   - Max Steps: `1500`
4. **Dataset**: Select image folder path
5. Click **Start Training**

---

## Quick Reference

| Method | Dim | Alpha | Conv Dim | Conv Alpha | LR | Steps | Batch |
|--------|-----|-------|----------|------------|-----|-------|-------|
| LoRA | 32 | 16 | 32 | 16 | 1e-4 | 3,000 | 2 |
| LoHA | 32 | 32 | 4 | 4 | 1e-4 | 1,500 | 1 |

---

## Output Files

After training, weights are saved in `output/` directory:
- LoRA: `model_lora.safetensors`
- LoHA: `model_loha.safetensors`
