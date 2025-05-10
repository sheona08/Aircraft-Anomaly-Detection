import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score,
    roc_curve, auc, precision_recall_curve, average_precision_score
)

# ─────────────── Load Data ───────────────
df_train = pd.read_excel(r'C:\Users\sheon\Downloads\archive\Dataset\PM_train.xlsx')
df_test = pd.read_excel(r'C:\Users\sheon\Downloads\archive\Dataset\PM_test.xlsx')
df_truth = pd.read_excel(r'C:\Users\sheon\Downloads\archive\Dataset\PM_truth.xlsx')

df = pd.concat([df_train, df_test], ignore_index=True)
df['RUL'] = df.groupby('id')['cycle'].transform('max') - df['cycle']

feature_columns = ['setting1', 'setting2', 'setting3', 's1', 's2', 's3', 's4', 's5', 's6',
                   's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16',
                   's17', 's18', 's19', 's20', 's21']

df = pd.merge(df, df_truth, on='id', how='left')

threshold = 30
df['label'] = (df['RUL'] <= threshold).astype(int)

# ─────────────── Z-Score Anomaly Detection ───────────────
z_scores = np.abs(zscore(df[feature_columns]))
df['anomaly'] = (z_scores > 3).any(axis=1).astype(int)

# ─────────────── Train-Test Split ───────────────
X_train, X_test, y_train, y_test = train_test_split(df[feature_columns], df['RUL'], test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

y_test_binary = (y_test <= threshold).astype(int)
y_pred_binary = (y_pred <= threshold).astype(int)

# ─────────────── Evaluation ───────────────
accuracy = accuracy_score(y_test_binary, y_pred_binary)
print(f'Accuracy: {accuracy:.4f}')
print(classification_report(y_test_binary, y_pred_binary))

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n--- Regression Metrics ---")
print(f"MAE  = {mae:.2f}")
print(f"RMSE = {rmse:.2f}")
print(f"R²   = {r2:.4f}")

# ─────────────── True vs Predicted RUL ───────────────
plt.figure(figsize=(6, 5))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('True RUL')
plt.ylabel('Predicted RUL')
plt.title('True vs Predicted RUL')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ─────────────── Confusion Matrix ───────────────
cm = confusion_matrix(y_test_binary, y_pred_binary)
plt.figure(figsize=(4, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Not Soon', 'Fail Soon'],
            yticklabels=['Not Soon', 'Fail Soon'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# ─────────────── ROC and PR Combined Plot ───────────────
y_score = -y_pred

fpr, tpr, _ = roc_curve(y_test_binary, y_score)
roc_auc = auc(fpr, tpr)
precision, recall, _ = precision_recall_curve(y_test_binary, y_score)
pr_auc = average_precision_score(y_test_binary, y_score)
baseline = y_test_binary.mean()

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

axs[0].plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})', lw=2)
axs[0].plot([0, 1], [0, 1], 'k--', lw=1, label='Chance')
axs[0].set_xlabel('False Positive Rate')
axs[0].set_ylabel('True Positive Rate')
axs[0].set_title('ROC Curve')
axs[0].legend(loc='lower right')
axs[0].grid(alpha=0.3)

axs[1].plot(recall, precision, label=f'PR (AP = {pr_auc:.3f})', lw=2)
axs[1].hlines(baseline, 0, 1, colors='grey', ls='--', lw=1,
              label=f'No-skill (P={baseline:.3f})')
axs[1].set_xlabel('Recall')
axs[1].set_ylabel('Precision')
axs[1].set_title('Precision–Recall Curve')
axs[1].legend(loc='upper right')
axs[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ─────────────── Plot Anomalies ───────────────
plt.figure(figsize=(10, 5))
plt.scatter(df.index, df['RUL'], c=df['anomaly'], cmap='coolwarm', alpha=0.6)
plt.xlabel('Sample Index')
plt.ylabel('Remaining Useful Life')
plt.title('Z-Score Anomaly Detection in Sensor Data')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
