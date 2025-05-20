This project applies machine learning techniques to detect anomalies in aircraft engine sensor data and predict the Remaining Useful Life (RUL) of engines. Using NASA’s 
C-MAPSS dataset, we built a pipeline that integrates both supervised and unsupervised learning to support predictive maintenance strategies and enhance aviation safety.
Objectives:
1) Predict engine failures before they occur using historical multivariate sensor data
2) Estimate Remaining Useful Life (RUL) for each engine
3) Detect anomalies in sensor behavior using Z-score analysis
4) Evaluate the effectiveness of classical models (e.g., linear regression) in high-stakes domains like aviation

Dataset
Source: NASA C-MAPSS
Files Used:

PM_train.xlsx – Training set
PM_test.xlsx – Test set
PM_truth.xlsx – Ground truth for RUL

Features:

3 operational settings (setting1, setting2, setting3)
21 sensor readings (s1 to s21)
Engine ID and cycle number

Methodology
Data Preprocessing
Computed RUL: RUL = max(cycle) - current(cycle)
Created binary labels:

1 = "Will fail soon" (RUL ≤ 30)
0 = "Healthy" (RUL > 30)

Modeling
Model Used: Linear Regression
Input: 24 features (3 settings + 21 sensors)
Output:
Continuous RUL prediction
Binary classification via thresholding (RUL ≤ 30)

Anomaly Detection
Z-score method applied across sensor features
Flagged any value with Z-score > 3 as an anomaly
Helps detect extreme operating conditions and supports the predictive model

Evaluation
Regression Metrics:
MAE: 38.83
RMSE: 50.62
R²: 0.4525

Classification Metrics (RUL ≤ 30 as positive class):
Accuracy: 94.5%
Precision: 91%
Recall: 48%
ROC AUC: 0.953
PR AUC: 0.808
Confusion Matrix:
TP: 305 | FN: 326 | FP: 29 | TN: 5854

Key Findings
Linear regression captured ~45% of the variance in RUL.
Strong classification performance despite using a simple model.
Z-score anomaly detection flagged early deviations up to 100 cycles before failure.
Most false negatives occurred when true RUL was 30–50 cycles — suggesting opportunity for smarter thresholding.
