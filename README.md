# Gastrointestinal Disease Detection using Deep Learning

This repository contains code and models for detecting and classifying gastrointestinal diseases from endoscopic images. Developed as part of a research project, this work leverages deep learning models to achieve high accuracy in identifying various types of gastrointestinal issues.

## Project Overview

This project aims to build a robust image classification system for detecting gastrointestinal diseases using multiple deep learning architectures. The project includes:
1. **Data Preprocessing**: Loading, resizing, and normalizing images for deep learning.
2. **Data Augmentation**: Using Augmentor to generate additional samples and improve model robustness.
3. **Model Training**: Fine-tuning pre-trained models like EfficientNet, Xception, and VGG19.
4. **Evaluation and Visualization**: Assessing models with performance metrics and visualizing results.

## Key Steps and Components

### 1. Data Preparation and Augmentation

The dataset consists of endoscopic images from the Kvasir dataset. To improve model generalization, we apply data augmentation techniques:
- **Rotation**, **Zooming**, **Flipping**, and **Random Distortion** to create more diverse training data.

Augmentation is done using the [Augmentor](https://augmentor.readthedocs.io/en/master/) library, producing 18,000 images for training.

### 2. Model Architectures

The project explores and fine-tunes multiple deep learning architectures for image classification:
- **EfficientNet**: Known for balancing accuracy and efficiency, it serves as a primary model.
- **Xception** and **VGG19**: Used for comparison, providing alternative architectures to improve results and benchmark performance.

Additional layers are added to these base models:
- **Convolutional Layers** for feature extraction.
- **Batch Normalization** and **Gaussian Noise** for regularization.
- **Global Average Pooling** and fully connected layers to enhance accuracy and reduce overfitting.

### 3. Model Training and Evaluation

The models are trained on the augmented dataset and evaluated using various metrics:
- **Accuracy**
- **Precision** and **Recall**
- **AUC (Area Under the Curve)**
- **Confusion Matrix** and **Classification Report** for class-wise performance insights.

The training includes early stopping and learning rate reduction to prevent overfitting and optimize performance.

### 4. Visualization and Analysis

The project includes code for visualizing:
- **Training History**: Accuracy and loss curves for each model.
- **Confusion Matrix**: Helps understand model performance across different classes.

## Getting Started

### Prerequisites

Install the necessary packages using:
```bash
pip install tensorflow pandas numpy matplotlib seaborn Augmentor
```

### Running the Code

1. **Data Preparation**: Place the dataset in the specified directory (e.g., `/kaggle/input/kvasir/kvasir-dataset-v2/`).
2. **Data Augmentation**: Run the augmentation section to create a larger training dataset.
3. **Model Training**: Run the notebook sections to train each model.
4. **Evaluation**: Use the provided metrics and visualizations to evaluate model performance.

### Example Results

After training, each model will output metrics such as accuracy, precision, recall, and AUC, along with confusion matrix plots and classification reports to evaluate model effectiveness.

## Repository Structure

- `Kelompok_14_models_final.ipynb`: Main Jupyter notebook with all code for preprocessing, augmentation, training, and evaluation.
- `images/`: Folder to store sample images and visualizations for documentation.

## Future Improvements

- **Hyperparameter Tuning**: Optimize parameters for further accuracy improvements.
- **Additional Augmentation**: Experiment with other augmentation techniques to improve model robustness.
- **Deployment**: Integrate the trained model into a web or mobile application for real-world use.

