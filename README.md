# Driver Drowsiness Detection System

> **METCS 767 Advanced Machine Learning · Boston University**  
> Dr. Farshid Alizadeh-Shabdiz

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8-green)
![AUC](https://img.shields.io/badge/ROC--AUC-0.964-brightgreen)
![F1](https://img.shields.io/badge/F1--Score-0.923-brightgreen)

---

## Overview

Real-time driver drowsiness detection using a **4-stage modular computer vision + temporal ML pipeline** that processes 30 FPS video and produces a live alert decision with interpretable intermediate outputs at every stage.

**The problem:** Drowsy driving causes ~100,000 crashes annually in the US (NHTSA), resulting in 1,550 fatalities, 71,000 injuries, and $15.9B in economic losses. Existing physiological solutions (EEG, EOG) are intrusive and can't scale to millions of vehicles. A cabin-camera approach is practical and deployable.

**Why this architecture:** The system is decomposed into 4 sequential tasks rather than a single black-box end-to-end model. Each stage is independently interpretable, debuggable, and replaceable. This is intentional design, not a limitation.

---

## Pipeline

```
Video Input (30 FPS, RGB 640×480)
        │
        ▼  Task 1: Face Detection
   RetinaFace (ResNet-50 + FPN)
   → Bounding box + face landmarks
   → Loss: Focal Loss + Smooth L1
        │
        ▼  Task 2: Facial Keypoint Regression
   ResNet-18 backbone (ImageNet pre-trained)
   → 20 landmarks (6 per eye, 8 inner mouth) + confidence scores
   → Loss: MSE + Confidence-Aware Loss (novel)
        │
        ▼  Task 3: EAR / MAR Feature Extraction
   Eye Aspect Ratio (EAR) — quantifies eye opening
   Mouth Aspect Ratio (MAR) — quantifies yawning
   → Per-second aggregation: median, max, fraction closed, yawn count
        │
        ▼  Task 4: Temporal Drowsiness Classification
   5-second sliding window → feature vector
   Random Forest Classifier
   → P(drowsy), threshold τ*, ALARM decision
```

---

## Task 1 — Face Detection (RetinaFace)

**Datasets:** WIDER Face (train/val) → FDDB (cross-dataset test)  
**Architecture:** RetinaFace with ResNet-50 backbone + Feature Pyramid Network (FPN)  
**Loss:** Focal Loss (handles anchor class imbalance) + Smooth L1 (stabilizes box regression)  
**Model selection:** RetinaFace chosen after FasterRCNN-ResNet18 failed to generalize.

| Model | Dataset | mAP@0.5 | DICE | Notes |
|-------|---------|---------|------|-------|
| Custom ResNet-50 | WIDER | 0.601 | 0.643 | Initial attempt |
| **RetinaFace (pre-trained)** | **WIDER** | **0.6123** | **0.7159** | **Selected** |
| RetinaFace (pre-trained) | FDDB | **0.9528** | **0.9143** | Cross-dataset test |

The 0.9528 mAP on FDDB demonstrates strong cross-dataset generalization. The gap between WIDER (0.6123) and FDDB (0.9528) reflects domain shift: WIDER contains extreme scales and occlusions. For driving scenarios, 0.6123 is more than sufficient — false negatives are recoverable via temporal smoothing.

---

## Task 2 — Facial Keypoint Regression (ResNet-18 + Confidence-Aware Loss)

**Dataset:** 300W (600 training images, 68 landmarks → 20 selected: 6 per eye, 8 inner mouth)  
**Architecture:** ResNet-18 backbone (ImageNet pre-trained, 11.5M parameters) → Global Average Pooling → FC(512) → ReLU → Dropout(0.3) → FC(60) → 20 landmarks × (x, y, confidence)

### Novel Contribution: Confidence-Aware Loss Function

Standard MSE regression treats all predictions equally. Our loss function explicitly trains the network to assign high confidence when accurate and low confidence when uncertain:

```
L = Σ [ confidence_i × ||pred_i - gt_i||² + λ × (1 - confidence_i) × max(0, threshold - ||pred_i - gt_i||) ]
```

This enables downstream EAR/MAR computation to weight unreliable landmarks less — critical for occlusion (glasses, hair, poor lighting).

**Results on 300W validation set:**

| Metric | Value |
|--------|-------|
| Validation MSE (per sample) | 0.0002 – 0.0009 |
| Average landmark confidence | **~0.958** |
| Pixel-level error at 256×256 | ~0.71 px (negligible for EAR/MAR) |

Sample predictions (green = ground truth, red = predicted):

> Samples 1–5 show tight overlap between GT and predictions across diverse faces, poses, lighting conditions, and ages. Average confidence 0.958 indicates the model learned to distinguish reliable from unreliable predictions.

---

## Task 3 — EAR / MAR Feature Extraction

**Eye Aspect Ratio (EAR)** — based on Soukupová & Terzopoulos (2016):
```
EAR = (||p2-p6|| + ||p3-p5||) / (2 × ||p1-p4||)
```
- EAR ~0.20–0.25 → Eyes open
- EAR ~0.15–0.18 → Eyes partially closed  
- EAR ~0.10–0.15 → Eyes closed

**Mouth Aspect Ratio (MAR):**
```
MAR = Σ(vertical distances) / (2 × horizontal mouth width)
```
- MAR ~0.10–0.12 → Mouth closed
- MAR ~0.13+ → Mouth open / yawning

**Thresholds** calibrated on 300W validation set (100 alert subjects), then tuned for NTHU-DDD driving conditions:  
EAR threshold: **0.220** (more sensitive to closed eyes with glasses/backlighting)  
MAR threshold: **0.160**

**Task 3 Classification Performance on NTHU-DDD:**

| Condition | Eye Closed Accuracy | Yawn Accuracy |
|-----------|-------------------|---------------|
| Drowsy frames | **74.78%** | **68.11%** |
| Alert frames | **82.43%** | 53.28% |

Live output example: `Eye:0 Yawn:0 EAR:0.221 MAR:0.079` overlaid on video frame in real-time.

---

## Task 4 — Temporal Drowsiness Classification

**Input:** 5-second sliding windows of per-second EAR/MAR aggregated features  
**Feature vector:** `[eye_closed_avg, eye_closed_max, eye_closed_frac, MAR_avg, MAR_max, yawn_count, yawn_prob]`

### Model Selection — Why Random Forest over LSTM

Five models were compared on the NTHU-DDD test set (89 sequences, 441 five-second windows):

| Model | Test F1 | Val F1 | Decision |
|-------|---------|--------|---------|
| Logistic Regression | 0.752 | — | Baseline |
| SVM | ~0.78 | — | Rejected |
| Gradient Boosting | ~0.85 | — | Rejected |
| LSTM | ~0.90 | **1.000** | ❌ **Rejected — overfitting** |
| **Random Forest** | **0.923** | 0.923 | ✅ **Selected** |

**The LSTM trap:** LSTM achieved perfect F1=1.000 on validation — which looks ideal. On the test set it collapsed. Random Forest maintained F1=0.923 across train/val/test, demonstrating genuine generalization rather than memorization. For a safety-critical system, consistent generalization matters more than peak validation performance.

**Optimal threshold:** τ* = 0.510 (F1-optimized on validation set)

**Confusion matrix at τ* = 0.510:**
| | Predicted Awake | Predicted Drowsy |
|--|----------------|-----------------|
| **Actual Awake** | 40 (TN) | 2 (FP) |
| **Actual Drowsy** | 5 (FN) | 42 (TP) |

### Live Demo Overlay
```
[FRAME METRICS]  EAR_R=0.1760  EAR_L=0.1629
EAR_avg=0.1694   Eye_closed=1
MAR=0.1371       Yawn=0
[5-SEC WINDOW]   Eye_closed_avg=0.947
MAR_avg=0.1359   Yawn_prob=0.920
[DROWSINESS MODEL]  P(drowsy)=0.412
Threshold=0.619  F1_Score=0.725  ALARM=0
```
### Random Forest vs LSTM — Side-by-Side Comparison

| Scenario | RF Prediction | LSTM Prediction | Ground Truth |
|----------|-------------|----------------|-------------|
| Alert driver (eyes open) | **AWAKE** P=0.060 ✅ | DROWSY P=0.429 ❌ | Awake |
| Drowsy driver (eyes closed) | **DROWSY** P=1.000 ✅ | DROWSY P=0.524 ✅ | Drowsy |

RF correctly identifies the alert driver (false positive from LSTM would trigger unnecessary alerts, desensitizing the driver). Both catch the drowsy driver, but RF does so with higher confidence (P=1.000 vs P=0.524).

[View Slides](presentation/'DRIVER DROWSINESS DETECTION SYSTEM_PPT.pptx')
---

## Final Results Summary

| Stage | Metric | Value |
|-------|--------|-------|
| Task 1: Face Detection | mAP@0.5 (FDDB cross-dataset) | **0.9528** |
| Task 1: Face Detection | DICE (FDDB) | **0.9143** |
| Task 2: Keypoint Regression | Avg landmark confidence | **0.958** |
| Task 2: Keypoint Regression | Validation MSE | **0.0002–0.0009** |
| Task 3: EAR/MAR | Drowsy frame eye-closed accuracy | **74.78%** |
| Task 3: EAR/MAR | Alert frame eye-closed accuracy | **82.43%** |
| Task 4: Temporal Classification | **F1-Score** | **0.923** |
| Task 4: Temporal Classification | **ROC-AUC** | **0.964** |

---

## Loss Function Design

| Stage | Loss | Reason |
|-------|------|--------|
| Face Detection | Focal Loss + Smooth L1 | Focal handles anchor class imbalance in dense detectors; Smooth L1 stabilizes bounding-box regression |
| Keypoint Regression | MSE + Confidence-Aware | MSE for coordinate accuracy; confidence modeling improves reliability for downstream geometric estimation |
| Eye/Yawn Classification | Binary Cross-Entropy | Natural loss for binary probabilistic outputs |
| Temporal Classification | BCE / Gini Impurity (RF) | Threshold-calibrated probabilities; RF's Gini impurity behaves like regularized BCE |

---

## How to Run

```bash
pip install -r requirements.txt

# Full pipeline on video input
python pipeline/task1_face_detection.py --input video.mp4
python pipeline/task2_keypoint_regression.py
python pipeline/task3_ear_mar.py
python pipeline/task4_classification.py --threshold 0.510
```

---

## References

1. Soukupová & Terzopoulos (2016). *Real-Time Eye Blink Detection using Facial Landmarks.*
2. Deng et al. (2019). *RetinaFace: Single-Shot Multi-Level Face Localisation in the Wild.* CVPR.
3. He et al. (2016). *Deep Residual Learning for Image Recognition.* CVPR.
4. NTHU-DDD Dataset: Driver drowsiness detection benchmark.

---

**Aryan Meena** · [LinkedIn](https://linkedin.com/in/aryan-meena-32685415a) · araj7042@gmail.com
Boston University, METCS 767 Advanced Machine Learning
