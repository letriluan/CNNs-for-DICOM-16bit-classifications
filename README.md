# Chest X-ray DICOM 12-Bit Abnormality Classification using Deep CNNs

This repository contains the implementation and trained models from my **Bachelor Thesis (2022)**, focusing on **abnormality classification of chest X-ray DICOM images using deep convolutional neural networks (CNNs)**.

Unlike many medical imaging pipelines that convert DICOM files to JPG/PNG, this project **processes and trains models directly on DICOM pixel data**, preserving the original **12-bit image depth** and medical image fidelity.

---

## Project Overview

- Chest X-ray **multi-class classification** (e.g. Normal, COVID-19, Opacity, Aortic enlargement)
- Direct **DICOM image processing pipeline**
- Comparison between **standard CNN architectures** and **proposed improvements**
- Experiments with **VGG16** and **ResNet50**
- Trained models saved in `.h5` format
- Implemented and evaluated as part of an undergraduate thesis

---


## Motivation and Problem Statement

Most deep learning approaches for chest X-ray classification convert DICOM images into
8-bit JPG or PNG formats before training. This conversion can lead to:

- Loss of pixel intensity precision
- Altered grayscale distributions
- Reduced diagnostic information in subtle abnormalities

This project investigates whether **training CNNs directly on 12-bit DICOM pixel data**
can improve classification performance and better preserve clinically relevant features,
compared to conventional image conversion pipelines.


---
## Repository Structure

```text
.
├── model_trained/
│ ├── RES_128_proposed.h5
│ └── VGG16_224.h5
│
├── train/
│ ├── main_train.ipynb
│ ├── Resnet50_proposed+standard.ipynb
│ └── VGG16_proposed+standard.ipynb
│
├── embedded_info.ipynb
├── pre_processing_data.ipynb
├── requirements.txt
└── README.md
```


### Folder Description

- **`pre_processing_data.ipynb`**  
  Loads and preprocesses chest X-ray DICOM files, normalises pixel values, resizes images, and prepares data for training.

- **`train/`**  
  Training notebooks comparing standard CNN architectures with proposed modifications.

- **`embedded_info.ipynb`**  
  Experiments with embedding selected DICOM metadata into PNG images.

- **`model_trained/`**  
  Pre-trained CNN models saved in `.h5` format.

---

## Technologies Used

- **Python**
- **TensorFlow / Keras**
- **NumPy, Pandas**
- **OpenCV**
- **pydicom**
- **Matplotlib**
- **Jupyter Notebook**

---

## Model Architectures

This project evaluates both **standard CNN architectures** and **proposed modifications**:

- **VGG16**
  - Baseline implementation using ImageNet-style architecture
  - Input resolution: 112x112, 224×224, 512x512

- **ResNet50**
  - Deep residual network to address vanishing gradients
  - Input resolution: 112x112, 224×224, 512x512

### Proposed Improvements
- Architectural adjustments to convolutional blocks
- Optimised input resolution for medical images
- Modified training strategies to improve convergence speed and accuracy

All models were trained and evaluated under the same preprocessing conditions for fair comparison.

---
## Evaluation Metrics

Model performance was evaluated using:

- Classification accuracy
- Validation loss
- Confusion matrix analysis
- Class-wise performance comparison

These metrics were used to assess both predictive performance and model robustness
across different abnormality classes.

---

## Key Contributions

- Designed a **DICOM-native deep learning pipeline** for chest X-ray analysis
- Demonstrated the impact of **12-bit pixel preservation** on model performance
- Conducted systematic comparisons between standard CNNs and proposed improvements
- Delivered a reproducible and well-documented experimental framework

