# Multiclass ML and DL Approaches for Parkinson's Disease Detection

This project explores the detection of Parkinson's Disease using both machine learning (ML) and deep learning (DL) techniques. By comparing the performance of classic ML models such as Random Forest, Support Vector Machines, K-Nearest Neighbors, and Decision Trees with a Feed-Forward Neural Network (FNN), the study aims to identify the most effective approach for accurate classification.

## Overview

The goal of this project is to evaluate the potential of ML and DL approaches in distinguishing between individuals with Parkinson's Disease and healthy individuals. Using biomedical voice measurements from the UCI Parkinson's Disease dataset, the project implements preprocessing, feature selection, hyperparameter optimization, and performance evaluation to achieve robust and reliable models.

## Key Features

- **Data Preprocessing**: Includes normalization, handling imbalanced data with SMOTE, and splitting datasets for training and testing.
- **Feature Selection**: Utilizes `SelectKBest` and mutual information to identify the most relevant features for classification.
- **Hyperparameter Optimization**: Applies RandomizedSearchCV for tuning model parameters.
- **Model Training**:
  - Classic ML Models: Random Forest, Decision Tree, K-Nearest Neighbors, Kernelized SVM.
  - Deep Learning Model: Feed-Forward Neural Network with L2 regularization and dropout.
- **Performance Metrics**: Accuracy, Precision, Recall, F1 Score, and ROC AUC are calculated for comprehensive evaluation.

## Results

The project demonstrates that:
- Deep Learning approaches, specifically Feed Neural Networks, achieve the highest accuracy and classification performance.
- Among classic ML approaches, K-Nearest Neighbors provides the best trade-off between simplicity and accuracy.

## Dataset

The dataset used for this project is the [UCI Parkinson's Disease Dataset](https://archive.ics.uci.edu/dataset/174/parkinsons), containing 195 records with 22 biomedical voice measurements and a target variable indicating the presence of Parkinson's Disease.


