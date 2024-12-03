# Spam Email Classification Challenge

This project leverages supervised machine learning to classify emails as spam or not spam. By training and evaluating Logistic Regression and Random Forest models, the goal is to enhance the email filtering system of an Internet Service Provider (ISP).

## Project Overview

This repository includes:

- Data preprocessing and feature scaling using `StandardScaler`.
- Training a Logistic Regression model and a Random Forest Classifier.
- Evaluating model performance based on accuracy scores.
- Comparing the effectiveness of the two models in spam detection.


## Table of Contents

- [Project Overview](#project-overview)
- [File Structure](#file-structure)
- [Dataset](#dataset)
- [Workflow](#workflow)
- [Results](#results)
- [Getting Started](#getting-started)
- [Conclusion](#conclusion)


## File Structure

- `spam_detector.ipynb`: The primary Jupyter Notebook containing data preparation, model implementation, and performance evaluation.
- `README.md`: Overview and instructions for the project.
- `data/`: Placeholder for datasets used for training and testing.

## Dataset

The dataset contains numerical features extracted from emails and a target column, `spam`, indicating whether an email is classified as:
- **Spam (1)**: Unwanted email.
- **Not Spam (0)**: Legitimate email.

## Workflow

1. **Data Preparation**:
   - Splitting the data into training and testing sets using `train_test_split`.
   - Scaling features with `StandardScaler` for improved model performance.

2. **Model Training**:
   - **Logistic Regression**: A linear model for binary classification tasks.
   - **Random Forest**: An ensemble model that uses decision trees to boost accuracy.

3. **Evaluation**:
   - Models are evaluated using accuracy scores.
   - Insights into which model performs better for spam detection.

## Results

Initial findings suggest that the Random Forest model outperforms Logistic Regression in detecting spam emails, achieving higher accuracy.

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/jsaintfleur/classification-challenge.git
   ```
2. Navigate to the project directory:
   ```bash
   cd classification-challenge
   ```
4. Install the required dependencies:
   ```python
   import pandas as pd
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import accuracy_score
   from sklearn.linear_model import LogisticRegression
   from sklearn.ensemble import RandomForestClassifier
   ```
5. Open `spam_detector.ipynb` to explore the code and run the models.


## Conclusion
This project provides a foundation for improving email filtering systems using machine learning. The insights gained from comparing the models can help ISPs enhance their spam detection capabilities.

[Back to Top](#Table-of-Contents)
