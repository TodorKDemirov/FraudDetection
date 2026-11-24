Credit Card Fraud Detection

This project focuses on building an end-to-end machine learning system for detecting fraudulent credit card transactions. The main goal was to develop a classification model and build the infrastructure around it, including data preprocessing, feature engineering, model evaluation, experiment tracking, API development and containerization.
The project uses Python and XGBoost for model development, MLflow for experiment tracking and model management, FastAPI for serving predictions, and Docker for running the application.

1. Project Overview

Credit card fraud detection is a binary classification problem where each transaction is classified as either legitimate or fraudulent.
One of the main challenges is the highly imbalanced nature of the dataset, since fraudulent transactions represent only a very small percentage of all transactions. Because of this, accuracy alone is not a reliable metric for evaluating the model.
The focus of this project is on building a model that can identify fraudulent transactions while keeping the number of false positive predictions relatively low.

2. Dataset

The project uses a credit card fraud detection dataset containing anonymized transaction information.
The dataset includes:
Time: Seconds elapsed between the first transaction and each subsequent transaction.
Amount: The transaction amount.
V1-V28: Anonymized numerical features obtained through PCA transformation.
Class: The target variable, where 0 represents a legitimate transaction and 1 represents a fraudulent transaction.

The data is split into training and testing sets using stratified sampling to preserve the original class distribution.

3. Data Preprocessing and Feature Engineering

I implemented a separate feature engineering module to keep data transformations reusable and consistent between training and inference.
The original Time feature is converted into an hour-of-day feature. Since time is cyclical, I used sine and cosine transformations to represent the hour:
hour_sin = sin(2 * pi * hour / 24)
hour_cos = cos(2 * pi * hour / 24)

This allows the model to capture the cyclical relationship between different hours of the day.
I also categorized transaction amounts into four groups: zero, small, medium and large. These categories are then converted into binary features using one-hot encoding.

4. Model Development

I experimented with different classification algorithms, including Logistic Regression, Random Forest and XGBoost.
XGBoost was selected as the final model because of its performance on the validation data and its ability to capture nonlinear relationships between features.
Hyperparameter tuning was performed to improve model performance and reduce overfitting.
The final configuration uses the following parameters:
n_estimators: 300
max_depth: 4
learning_rate: 0.05
subsample: 0.8
colsample_bytree: 0.8
min_child_weight: 5
gamma: 0.3
reg_alpha: 0.1
reg_lambda: 1

5. Model Evaluation

Since the dataset is highly imbalanced, I used Precision-Recall AUC as one of the main evaluation metrics.
The final model achieved a cross-validation PR-AUC of 0.853239 and a test PR-AUC of 0.895715.
I also experimented with different classification thresholds instead of using the default threshold of 0.5.
The selected threshold is 0.6, which produced the following results on the test set:
Precision: 0.976190
Recall: 0.836735
F1-score: 0.901099

The results show that the model achieves high precision while maintaining a good recall, which is important in fraud detection because both missed fraudulent transactions and incorrectly flagged legitimate transactions have costs.

6. MLflow Integration

MLflow is used for experiment tracking and model management.
It allows me to track experiments, log model parameters and metrics, and manage different model versions.
The final XGBoost model is registered in the MLflow Model Registry under the name FraudDetectionXGBoost.
The registered model is assigned the champion alias, which is used by the API to load the selected model version.

7. API Development

I used FastAPI to create an API for making predictions with the trained model.
The API accepts transaction data, validates the input, applies the required feature engineering transformations and returns a prediction together with the estimated fraud probability.
The main endpoints are:
GET / - Basic API status.
GET /health - Health check and model information.
POST /predict - Predicts whether a transaction is fraudulent.

The API also provides interactive documentation through Swagger UI.

8. Docker

Docker and Docker Compose are used to containerize and run the different components of the application.
The application consists of an MLflow server, a model registration service and a FastAPI service.
This setup makes it easier to run the project in a consistent environment without having to configure each component manually.

9. Continuous Integration

I implemented a GitHub Actions workflow to automate parts of the development process.
The workflow runs when changes are pushed or pull requests are created for the main branch.
It installs the required dependencies, runs the automated tests using pytest, builds the Docker image and publishes it to GitHub Container Registry.

11. Technologies Used

Python
Pandas
NumPy
Scikit-learn
XGBoost
MLflow
FastAPI
Pydantic
Docker
Docker Compose
Pytest
GitHub Actions

Author: Todor Demirov

GitHub: https://github.com/TodorKDemirov
