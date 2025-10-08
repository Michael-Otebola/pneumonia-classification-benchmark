# 🩺 Pneumonia Classification Benchmark  
*A Comparative Study of CNN Architectures for Chest X-Ray Pneumonia Detection*

---

<div align="center">
  
[![View Streamlit App](https://img.shields.io/badge/🔗_Open%20App-Streamlit-blue?style=for-the-badge&logo=streamlit)](https://pneumonia-classification-benchmark.streamlit.app/)
[![Connect on LinkedIn](https://img.shields.io/badge/👤_Michael%20Otebola-LinkedIn-darkblue?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/michael-otegbola)

</div>

---

## 📘 Abstract  
This research presents a **comparative benchmark of CNN architectures** — **Baseline CNN**, **ResNet50**, **DenseNet121**, **VGG16**, and **InceptionV3** — for automated detection of pneumonia from chest X-ray images.  
Each model was fine-tuned using transfer learning and evaluated on balanced test data, with metrics including accuracy, precision, recall, and F1-score.  

The study also features a **Streamlit web app** for real-time inference and visual explanation, aiming to bridge academic research and clinical AI deployment.

---

## 🎯 Research Objectives  
- Benchmark multiple CNN architectures on a unified X-ray dataset.  
- Evaluate transfer learning vs. full fine-tuning.  
- Analyze generalization via confusion matrices and F1-scores.  
- Visualize model attention using Grad-CAM heatmaps.  
- Deploy a working diagnostic demo for end-user interaction.

---

## 🧠 Methodology  

### Dataset  
- **Source:** Kaggle Chest X-Ray Pneumonia Dataset  
- **Images:** ~5,800 total  
- **Classes:** Normal (0), Pneumonia (1)  
- **Split:** 70% train | 15% val | 15% test  
- **Imbalance:** Pneumonia ≈ 3× Normal  
- **Mitigation:** Augmentation and class-balanced sampling  

### Preprocessing  
- Resize → 224×224 (InceptionV3: 299×299)  
- Convert grayscale → RGB  
- Normalize (ImageNet mean/std)  
- Augmentations → rotation ±10°, flips, contrast jitter, Gaussian blur  

---

## ⚙️ Training Configuration  

| Setting | Value |
|----------|-------|
| Optimizer | Adam |
| Loss | BCEWithLogits / CrossEntropyLoss |
| LR | 1e-4 |
| Batch Size | 32 |
| Epochs | 15 |
| Framework | PyTorch |
| Device | CPU / GPU |

---

## 🧩 Models Compared  

| Model | Pretrained | Params | Key Feature |
|:------|:------------|:--------:|:-------------|
| **Baseline CNN** | None | ~5M | Custom scratch CNN |
| **ResNet50** | ImageNet | 25M | Residual learning |
| **DenseNet121** | ImageNet | 8M | Dense connectivity |
| **VGG16** | ImageNet | 138M | Classic deep CNN |
| **InceptionV3** | ImageNet | 24M | Multi-scale filters |

---

## 📊 Results Summary  

| Model | Accuracy | Precision (0/1) | Recall (0/1) | F1 (0/1) | Notes |
|:------|:---------:|:---------------:|:-------------:|:---------:|:------|
| **Baseline CNN** | 0.82 | 0.80 / 0.84 | 0.75 / 0.88 | 0.77 / 0.86 | Decent scratch model |
| **ResNet50** | **0.95** | 0.98 / 0.93 | 0.88 / 0.99 | 0.93 / 0.96 | Balanced & stable |
| **DenseNet121** | **0.96** | 0.97 / 0.95 | 0.91 / 0.98 | 0.94 / 0.96 | Best overall performer |
| **VGG16** | 0.87 | 0.89 / 0.86 | 0.74 / 0.94 | 0.80 / 0.90 | Over-predicts Pneumonia |
| **InceptionV3** | 0.74 | 0.97 / 0.71 | 0.31 / 0.99 | 0.47 / 0.83 | Overfits on small data |

**Winner:** *DenseNet121*  
**Clinical Safety Pick:** *ResNet50* (higher Pneumonia recall)

---

## 🧾 Confusion Matrix (ResNet50)

| | Pred: Normal | Pred: Pneumonia |
|:--|:--:|:--:|
| **Actual: Normal** | 206 | 28 |
| **Actual: Pneumonia** | 5 | 385 |

**Accuracy:** 94.7% **Macro F1:** 0.94 **ROC-AUC:** 0.97  

---

## 📈 Training & Evaluation Visuals  

> 🖼️ *(Upload these to your `/results` folder and they’ll render automatically)*

### 🔹 1. Training Loss vs Validation Accuracy  
![Loss vs Accuracy](results/loss_accuracy_curve.png)

---

### 🔹 2. Confusion Matrices by Model  
| ResNet50 | DenseNet121 | VGG16 |
|:--:|:--:|:--:|
| ![ResNet50 CM](results/cm_resnet50.png) | ![DenseNet121 CM](results/cm_densenet.png) | ![VGG16 CM](results/cm_vgg16.png) |

---

### 🔹 3. ROC Curves  
![ROC Curves](results/roc_curves.png)

---

### 🔹 4. Grad-CAM Heatmaps  
| Normal | Pneumonia |
|:--:|:--:|
| ![Grad-CAM Normal](results/gradcam_normal.png) | ![Grad-CAM Pneumonia](results/gradcam_pneumonia.png) |

---

## 💬 Discussion  
- Transfer learning substantially boosts model generalization.  
- DenseNet and ResNet exhibit strong stability and recall balance.  
- VGG16’s high parameter count limits scalability.  
- InceptionV3 shows sensitivity to small validation sets.  
- Pneumonia recall prioritized — better false positives than missed diagnoses.  

---

## 🚀 Future Work  
- Add **EfficientNet / MobileNetV2 / ViT** for extended benchmarks.  
- Expand to **multi-class pneumonia types** (viral, bacterial, COVID).  
- Integrate **explainability dashboards** in the web app.  
- Deploy **ONNX / TorchScript** for edge and hospital devices.  
- Conduct **cross-dataset generalization** studies.

---

## 🗂️ Repository Layout  

