import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import rankdata
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import shap

# 1. Import data
df = pd.read_csv('https://raw.githubusercontent.com/prasertcbs/basic-dataset/refs/heads/master/Loan-Approval-Prediction.csv')  # Replace with your file path

# 2. Identify target variable (assuming it's called 'target')
target = 'Loan_Status'  # Replace with your actual target column name
X = df.drop(columns=[target])
y = df[target]

# 3. Univariate statistics
print("Univariate Statistics:")
print(X.describe())  # Shows count, mean, std, min, max, quartiles

# 4. Box plots and outlier removal
plt.figure(figsize=(12, 6))
for i, column in enumerate(X.columns):
    plt.subplot(2, (len(X.columns)+1)//2, i+1)
    sns.boxplot(y=X[column])
    plt.title(column)
plt.tight_layout()
plt.show()

# Remove outliers using IQR method
def remove_outliers(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    return df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

X_clean = remove_outliers(X)
y_clean = y[X_clean.index]

# 5. Univariate plots and KDE
plt.figure(figsize=(12, 6))
for i, column in enumerate(X_clean.columns):
    plt.subplot(2, (len(X_clean.columns)+1)//2, i+1)
    sns.histplot(X_clean[column], kde=True)
    plt.title(column)
plt.tight_layout()
plt.show()

# 6. Monotonicity constraint (preparing data with rank transformation)
X_mono = X_clean.copy()
for column in X_mono.columns:
    X_mono[column] = rankdata(X_mono[column])  # Rank transformation for monotonicity

# 7. Correlation analysis
correlation_matrix = X_mono.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()

# Remove highly correlated variables (>0.85)
upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper_tri.columns if any(upper_tri[column].abs() > 0.85)]
X_final = X_mono.drop(columns=to_drop)
print(f"Removed variables due to high correlation: {to_drop}")

# 8. Split data (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y_clean, test_size=0.3, random_state=42
)

# 9. XGBoost model
params = {
    'max_depth': 3,
    'n_estimators': 30,
    'objective': 'binary:logistic',  # Assuming binary classification
    'learning_rate': 0.1,           # Low regularization
    'reg_alpha': 0.01,             # L1 regularization
    'reg_lambda': 0.01,            # L2 regularization
    'eval_metric': 'auc',
    'monotone_constraints': tuple([1] * X_final.shape[1])  # Enforce monotonicity
}

model = xgb.XGBClassifier(**params)
model.fit(X_train, y_train)

# 10. Calculate AUC
train_pred = model.predict_proba(X_train)[:, 1]
test_pred = model.predict_proba(X_test)[:, 1]

train_auc = roc_auc_score(y_train, train_pred)
test_auc = roc_auc_score(y_test, test_pred)
print(f"Train AUC: {train_auc:.4f}")
print(f"Test AUC: {test_auc:.4f}")

# 11. SHAP plots
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Summary plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, plot_type="bar")
plt.show()

# Detailed SHAP summary plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test)
plt.show()

# Save the model
model.save_model('xgboost_model.json')
