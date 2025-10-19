<p align="center">
  <img src="runs/confusion_ALL9_multilayer_tau07.png" alt="Confusion Matrix" width="600">
</p>

<h1 align="center">🧬 A Multilayer Classification Framework for Skin Disease Detection Using the SkinBench Dataset</h1>

<p align="center">
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.1.0-EE4C2C?logo=pytorch&logoColor=white"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white"></a>
  <a href="#"><img src="https://img.shields.io/badge/License-MIT-green.svg"></a>
  <a href="#"><img src="https://img.shields.io/badge/Status-Completed-success"></a>
</p>

---

## 🧾 Abstract

This repository contains the full training and evaluation pipeline for the research project:

> **“A Multilayer Classification Framework for Skin Disease Detection Using the SkinBench Dataset.”**

It implements a **two-layer deep learning architecture** combining **ResNet-50** and **DenseNet-121**, with gating logic and performance calibration for accurate classification of **9 dermatological categories** from the **SkinBench** dataset.

---

## 📁 Repository Structure

```
SkinBench/
├── train_multilayer.py
├── inference_multilayer.py
├── utils/
│   └── datasets.py
├── models/
│   ├── resnet_model.py
│   ├── DenseNet121.py
│   ├── mobilenetv3.py
│   ├── cnn_model.py
│   ├── hybridmodel.py
│   ├── hybridSwinDenseNetMLP.py
│   └── __init__.py
├── eval_tools/
│   ├── run_all_evals.py
│   ├── dump_multilayer_preds.py
│   ├── stats_mcnemar.py
│   ├── xai_gradcam.py
│   ├── plot_curves.py
│   ├── make_my_thesis_figures.py
│   └── error_analysis.py
├── raw_data/
├── runs/ (checkpoints, figures, tables)
└── requirements.txt
```

---

## 🧠 Dataset Overview — *SkinBench*

The **SkinBench dataset** contains ~1,500 dermatoscopic images across 9 diagnostic categories:

| Label | Folder Name | Example ID Range |
|:------|:-------------|:----------------|
| 0 | Acne | (601–800) |
| 1 | Bacterial Infections (Impetigo) | (1351–1528) |
| 2 | Eczema (Atopic/Seborrhoeic) | (801–1100) |
| 3 | Fungal Infections (Tinea, Candidiasis) | (1101–1350) |
| 4 | Normal | — |
| 5 | Pigmentation Disorders | — |
| 6 | Pox (Chickenpox, Monkeypox) | (381–600) |
| 7 | Psoriasis | (176–380) |
| 8 | Scabies | (01–175) |

Images are resized to **256×256**, normalized using ImageNet statistics, and split (70/15/15) with a fixed random seed for reproducibility.

---

## ⚙️ Methodology

### 🔹 Layer 1 — ResNet-50 (Binary)
- **Task:** Normal vs Abnormal detection  
- **Output:** {Normal, Abnormal}  
- **Purpose:** Reduces false positives by filtering normal samples before detailed classification.

### 🔹 Layer 2 — DenseNet-121 (Multiclass)
- **Task:** 8-way classification for abnormal skin diseases  
- **Output:** {Acne, Impetigo, Eczema, Fungal, Pigmentation, Pox, Psoriasis, Scabies}

### 🔸 Decision Rule
\[
\text{If } p_{\text{Normal}}(x) \ge \tau \Rightarrow \text{Normal}; \quad \text{else} \Rightarrow \text{L2 prediction}
\]
Optimal τ = **0.70** (validated via Macro-F1 sweep).

---

## 🧮 Training Configuration

| Parameter | Value |
|:-----------|:------|
| Framework | PyTorch 2.1.0 |
| Image Size | 256×256 |
| Batch Size | 32 |
| Optimizer | Adam (lr=1e-4) |
| Loss | Weighted Cross-Entropy |
| Precision | Mixed (AMP) |
| Scheduler | StepLR |
| Early Stop | 10 epochs |
| Device | CUDA GPU |

---

## 📊 Results Summary

| Phase | Model | Accuracy | Macro-F1 |
|:------|:------|----------:|----------:|
| L1 | ResNet-50 | **1.0000** | **1.0000** |
| L2 | DenseNet-121 | **0.9708** | **0.9691** |
| ALL9 (flat) | Swin-DenseNet | 0.9687 | 0.9653 |
| ALL9 (flat) | ViT-ResNet | 0.9592 | 0.9584 |
| ALL9 (flat) | MobileNetV3 | 0.9498 | 0.9459 |
| ALL9 (flat) | CNN | 0.6426 | 0.6287 |
| **Multilayer (τ=0.70)** | **ResNet50 + DenseNet121** | **0.9781** | **0.9763** |

---

## 🔥 ROC & Confusion Visualization

| Metric | Visualization |
|:--|:--|
| Confusion Matrix | ![Confusion Matrix](runs/confusion_ALL9_multilayer_tau07.png) |
| ROC–AUC | ![ROC](runs/roc_ovr_ALL9_swin_densenet.png) |
| Grad-CAM | ![GradCAM](XAI_results/cam_L2_densenet_6.png) |

---

## 🧪 Threshold Tuning (τ-Sweep)

| τ | Accuracy | Macro-F1 |
|:--:|:--:|:--:|
| 0.50 | 0.9781 | 0.9761 |
| 0.55 | 0.9781 | 0.9761 |
| 0.60 | 0.9749 | 0.9733 |
| 0.65 | 0.9749 | 0.9733 |
| **0.70** | **0.9781** | **0.9763** |

---

## 📈 Calibration (Temperature Scaling)

| T | acc_raw | macroF1_raw | ECE_raw | acc_cal | macroF1_cal | ECE_cal |
|:--:|:--------:|:-------------:|:--------:|:---------:|:-------------:|:---------:|
| 1.4 | 0.9708 | 0.9691 | 0.0187 | 0.9708 | 0.9691 | 0.0267 |

---

## 📊 McNemar Significance Test

| Model 1 | Model 2 | n | c01 | c10 | χ² | p | Significant? |
|:----------|:----------|:--:|:--:|:--:|:--:|:--:|:--:|
| Multilayer (τ=0.70) | Swin-DenseNet | 319 | 7 | 4 | 0.364 | 0.546 | ❌ No |

> Improvement not statistically significant (p>0.05) but consistent across runs.

---

## 🧠 Explainability (XAI)

Grad-CAM overlays generated for L2 DenseNet show strong **lesion-focused activation** for correct predictions and dispersed attention in misclassifications.

---

## 🧰 Training Commands

```bash
python train_multilayer.py --data_dir raw_data --phase L1 --model resnet50
python train_multilayer.py --data_dir raw_data --phase L2 --model densenet121
python train_multilayer.py --data_dir raw_data --phase ALL9 --model mobilenetv3
python train_multilayer.py --data_dir raw_data --phase ALL9 --model cnn
python train_multilayer.py --data_dir raw_data --phase ALL9 --model vit_resnet
python train_multilayer.py --data_dir raw_data --phase ALL9 --model swin_densenet

python eval_tools/make_my_thesis_figures.py --data_dir raw_data
```

---

## 🧮 Multilayer Evaluation

```bash
python -m eval_tools.dump_multilayer_preds --data_dir raw_data --tau 0.70 --l1_ckpt runs/L1/resnet50/best.pt --l2_ckpt runs/L2/densenet121/best.pt --out runs/ALL9/multilayer_tau070_preds.csv --save_png_cm
```

---

## ⚖️ Statistical Comparison

```bash
python -m eval_tools.stats_mcnemar --pred1 runs/ALL9/multilayer_tau070_preds.csv --pred2 runs/ALL9/swin_densenet_preds.csv --name1 multilayer_tau070 --name2 swin_densenet
```

---

## 🎨 Grad-CAM Visualization

```bash
python -m eval_tools.xai_gradcam --data_dir raw_data --ckpt runs/L2/densenet121/best.pt --model densenet --phase L2
```

---

## 📜 Citation

```
@article{skinbench2025multilayer,
  title={A Multilayer Classification Framework for Skin Disease Detection Using the SkinBench Dataset},
  author={Anonymous Research Group},
  year={2025},
  note={Deep Learning for Medical Imaging Research Project}
}
```

---

## 🧩 Acknowledgements
- SkinBench dataset and community contributors.  
- PyTorch and torchvision teams for pretrained backbones.  
- Open-source medical imaging research inspiring this layered approach.

---

## 🪪 License
Released under the **MIT License** – for academic and research use only.

<p align="center"><i>“Bridging Deep Learning and Dermatology through Layered Intelligence.”</i></p>
