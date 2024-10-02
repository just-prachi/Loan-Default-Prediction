# Loan Default Prediction on Peer-to-Peer Lending Platform

## Introduction
This project aims to predict loan defaults using data from a peer-to-peer lending platform. Various machine learning models, including Linear Regression, Ridge Regression, Lasso Regression, Random Forest, and Neural Networks, were evaluated to determine which model best predicts loan default. The performance of each model was assessed using metrics such as Mean Squared Error (MSE), F1 Score, Recall, Precision, and Accuracy on both training and test datasets.

## Dataset
The analysis uses two datasets (training and testing) from the lending platform, each with 226,067 rows and 33 columns, comprising both numerical and categorical data:
- **Numeric features**: loan amount, interest rate, annual income, payment metrics.
- **Categorical features**: loan grade, employment length, home ownership, application type.
  
The target variable is "loan status" (whether a loan is fully paid or charged off). The status was converted into a binary classification: 1 for "Charged Off" and 0 for any other status.

## Data Preprocessing
1. **Handling Missing Values**:
   - Numerical columns: Missing values were replaced with the column mean.
   - Categorical columns: Missing values were filled with the most frequent category.

2. **Scaling Numeric Data**:
   - Numerical data was standardized to have a mean of zero and a standard deviation of one.

3. **Encoding Categorical Data**:
   - Categorical data was encoded using one-hot encoding.

These steps were combined into a single transformation pipeline for efficient data preprocessing before model training.

## Feature Correlation Analysis
The top 10 features most correlated with loan default include:
- **Highly Positive Correlations**: 
  - "recoveries" (0.517), "collection_recovery_fee" (0.493), indicating a strong link to charged-off loans.
  - "int_rate" (0.199) suggests that higher interest rates increase the likelihood of default.
  
- **Highly Negative Correlations**: 
  - "total_rec_prncp" (-0.216) has a strong inverse relationship with loan default.

Features with minimal correlation include "tot_coll_amt" (-0.000453) and "collections_12_mths_ex_med" (0.004), which show little relationship with loan status and thus have limited predictive power.

## Model Evaluation

### 1. Linear Regression
- **MSE**: 0.068 (training), 0.069 (test)
- **Accuracy**: 91.40%
- **Precision**: 0.989
- **Recall**: 0.223
- **F1 Score**: 0.364
- The model performs well in accuracy and precision but struggles with recall, meaning it fails to identify a significant portion of charged-off loans.

### 2. Ridge Regression (λ = 3.0)
- **MSE**: 0.068 (training), 0.068 (test)
- **Accuracy**: 91.43%
- **Precision**: 0.988
- **Recall**: 0.225
- **F1 Score**: 0.416
- Ridge Regression offers similar results to Linear Regression, with a slight improvement in regularization but still low recall.

### 3. Lasso Regression (λ = 0.01)
- **MSE**: 0.070 (training), 0.071 (test)
- **Accuracy**: 91.10%
- **Precision**: 0.985
- **Recall**: 0.224
- **F1 Score**: 0.365
- Lasso Regression shows higher prediction errors (MSE) and lower recall, making it less effective for predicting charged-off loans.

### 4. Random Forest
- **MSE**: 0.011 (training), 0.028 (test)
- **Accuracy**: 96.60%
- **Precision**: 0.967
- **Recall**: 0.713
- **F1 Score**: 0.830
- Random Forest outperforms all other models, with the highest accuracy, recall, and F1 score. It makes the fewest prediction errors and excels in identifying both charged-off and non-charged-off loans.

### 5. Neural Network
- **MSE**: 0.012 (training), 0.032 (test)
- **Accuracy**: 96.10%
- **Precision**: 0.964
- **Recall**: 0.701
- **F1 Score**: 0.807
- The Neural Network model shows strong performance, with high accuracy and balanced precision and recall. However, it is slightly outperformed by the Random Forest model.

## Conclusion
Among the models evaluated, the **Random Forest** model achieved the best performance with the lowest MSE, highest F1 score, and recall, making it the most reliable model for predicting loan defaults. Although the **Neural Network** also performed well, Random Forest showed superior accuracy (96.6%) and was more effective at identifying charged-off loans. Linear Regression, Ridge Regression, and Lasso Regression provided decent accuracy but lacked in recall, limiting their effectiveness in detecting loan defaults.
