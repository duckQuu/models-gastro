EfficientNet-based Image Classification for Gastrointestinal Disease Detection

This repository contains code for training an image classification model using the EfficientNet architecture to classify gastrointestinal images. The model is fine-tuned with data augmentation and uses advanced layers for enhanced accuracy and robustness.
Project Overview

The main steps in this project include:
Data Augmentation: Using Augmentor to generate diverse samples.
Image Preprocessing: Loading, resizing, and normalizing images.
Model Architecture: A deep learning model built on EfficientNet with added convolutional, batch normalization, and noise layers.
Evaluation: Using various metrics to assess model performance.
Steps in Detail

1. Data Augmentation
Data augmentation techniques such as rotation, flipping, and zooming are applied using Augmentor to enhance dataset diversity:
Rotation with a probability of 0.8
Zooming with a probability of 0.2
Flipping horizontally and vertically
Random distortion
Augmented samples are generated for robust training on 18,000 images.
2. Image Preprocessing
Image Loading: The get_images function loads and resizes images from a specified directory, handling errors and unreadable files.
Label Encoding: Labels are encoded for compatibility with the model.
Train-Test Split: The dataset is split into training and testing sets with stratification.
3. Model Architecture
The model is based on EfficientNetB1, a pre-trained CNN architecture known for its efficiency and accuracy. Additional layers are added to the base model:
Convolutional layers with ReLU activation
Batch normalization and Gaussian noise layers for regularization
Global average pooling and fully connected layers
Dropout layers to prevent overfitting
Final softmax layer for multi-class classification
The model is compiled with:
Optimizer: Adam with a learning rate of 0.0001
Loss Function: Categorical cross-entropy
Metrics: Accuracy, Precision, Recall, and AUC
4. Model Evaluation
The model is evaluated using:
Confusion Matrix: Provides insights into classification accuracy for each class.
Balanced Accuracy Score (BAS) and Matthews Correlation Coefficient (MCC): Metrics for balanced performance assessment.
F1 Score: Measures model robustness, with a custom weighted F1 score for imbalanced datasets.
How to Run

Dependencies: Install necessary libraries with:
pip install tensorflow pandas numpy matplotlib seaborn Augmentor
Data Preparation: Place the dataset in the specified directory.
Run the Code: Execute the script to train and evaluate the model.
Example Output

The script will display model architecture, training progress, evaluation metrics, and a confusion matrix for visual inspection of model performance.
