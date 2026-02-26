# Handwritten Digit Recognition System

**Course:** HDD5040 Foundations of AI and ML
**Project:** Intelligent Character Recognition (Handwritten Digits)

## Project Overview
This project is a GUI-based system that recognizes handwritten digits (0-9) using two different AI approaches:
1. **Traditional Machine Learning:** Support Vector Machine (SVM) with HOG features.
2. **Deep Learning:** Convolutional Neural Network (LeNet-5 architecture).

The system allows users to upload an image, select a model, and view the predicted digit along with the confidence score.

## Folder Structure
* `src/`: Contains all source code for the GUI, models, and data processing.
* `outputs/models/`: Contains the trained model files (`svm_hog.pkl`, `cnn_lenet.pt`, `mlp.pt`).
* `test_samples/`: Contains generated test images (digits 0-9) for quick testing.

## Installation & Setup
1. **Create a Virtual Environment** (Recommended):
   ```bash
   python -m venv venv
   # Activate on Windows:
   venv\Scripts\activate
   # Activate on Mac/Linux:
   source venv/bin/activate

Role: Contributor
