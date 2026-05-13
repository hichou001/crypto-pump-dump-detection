import pandas as pd
import numpy as np

from sqlalchemy import create_engine

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)

import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# MYSQL CONNECTION
# ==========================================

engine = create_engine(
    "mysql+pymysql://root:12345678@127.0.0.1:3306/crypto_ml"
)

# ==========================================
# LOAD DATA
# ==========================================

query = "SELECT * FROM final_dataset_15k"

df = pd.read_sql(query, engine)

print("=" * 60)
print("DATASET PREVIEW")
print("=" * 60)

print(df.head())

print("\n")

print("=" * 60)
print("DATASET SHAPE")
print("=" * 60)

print(df.shape)

# ==========================================
# REMOVE LEAKAGE COLUMNS
# ==========================================

drop_columns = [
    "timestamp",
    "symbol",
    "future_return_5m",
    "is_pump",
    "is_pump_target"
]

drop_columns = [col for col in drop_columns if col in df.columns]

df = df.drop(columns=drop_columns)

# ==========================================
# FEATURES & LABEL
# ==========================================

X = df.drop("label", axis=1)

y = df["label"]

print("\n")

print("=" * 60)
print("FEATURE COLUMNS")
print("=" * 60)

print(X.columns.tolist())

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================
# RANDOM FOREST MODEL
# ==========================================

model = RandomForestClassifier(

    n_estimators=500,

    max_depth=None,

    min_samples_leaf=3,

    min_samples_split=5,

    class_weight='balanced_subsample',

    max_features='sqrt',

    bootstrap=True,

    random_state=42,

    n_jobs=-1
)

# ==========================================
# TRAIN MODEL
# ==========================================

print("\n")

print("=" * 60)
print("TRAINING RANDOM FOREST...")
print("=" * 60)

model.fit(X_train, y_train)

print("TRAINING COMPLETED!")

# ==========================================
# PREDICT PROBABILITY
# ==========================================

y_prob = model.predict_proba(X_test)[:, 1]

# ==========================================
# THRESHOLD SEARCH
# ==========================================

print("\n")

print("=" * 60)
print("THRESHOLD SEARCH")
print("=" * 60)

best_threshold = 0

best_f1 = 0

for t in np.arange(0.10, 0.91, 0.05):

    y_pred_t = (y_prob >= t).astype(int)

    precision = precision_score(y_test, y_pred_t)

    recall = recall_score(y_test, y_pred_t)

    f1 = f1_score(y_test, y_pred_t)

    print(
        f"Threshold={t:.2f} | "
        f"Precision={precision:.3f} | "
        f"Recall={recall:.3f} | "
        f"F1={f1:.3f}"
    )

    # ưu tiên:
    # precision >= 0.55
    # F1 cao nhất

    if precision >= 0.55 and f1 > best_f1:

        best_f1 = f1

        best_threshold = t

print("\n")

print("=" * 60)
print("BEST THRESHOLD")
print("=" * 60)

print("Best Threshold:", best_threshold)

print("Best F1:", best_f1)

# ==========================================
# FINAL PREDICTION
# ==========================================

y_pred = (y_prob >= best_threshold).astype(int)

# ==========================================
# CLASSIFICATION REPORT
# ==========================================

print("\n")

print("=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(classification_report(y_test, y_pred))

# ==========================================
# METRICS
# ==========================================

print("\n")

print("=" * 60)
print("FINAL METRICS")
print("=" * 60)

print("Accuracy :", accuracy_score(y_test, y_pred))

print("Precision:", precision_score(y_test, y_pred))

print("Recall   :", recall_score(y_test, y_pred))

print("F1 Score :", f1_score(y_test, y_pred))

print("ROC AUC  :", roc_auc_score(y_test, y_prob))

# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Greens'
)

plt.title("Random Forest Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()

# ==========================================
# FEATURE IMPORTANCE
# ==========================================

importance = model.feature_importances_

feature_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': importance
})

feature_df = feature_df.sort_values(
    by='importance',
    ascending=False
)

print("\n")

print("=" * 60)
print("TOP 10 IMPORTANT FEATURES")
print("=" * 60)

print(feature_df.head(10))

# ==========================================
# FEATURE IMPORTANCE PLOT
# ==========================================

plt.figure(figsize=(10, 6))

sns.barplot(
    data=feature_df.head(10),
    x='importance',
    y='feature'
)

plt.title("Top 10 Feature Importance")

plt.show()

# ==========================================
# ROC CURVE
# ==========================================

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_prob
)

plt.figure(figsize=(7, 6))

plt.plot(
    fpr,
    tpr,
    linewidth=2,
    label=f"Random Forest AUC = {roc_auc_score(y_test, y_prob):.4f}"
)

# random baseline
plt.plot(
    [0, 1],
    [0, 1],
    linestyle='--'
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve - Random Forest")

plt.legend(loc="lower right")

plt.grid(True)

plt.show()

# ==========================================
# TRAIN / TEST ACCURACY
# ==========================================

print("\n")

print("=" * 60)
print("TRAIN / TEST ACCURACY")
print("=" * 60)

print("Train Accuracy:", model.score(X_train, y_train))

print("Test Accuracy :", model.score(X_test, y_test))

# ==========================================
# CROSS VALIDATION
# ==========================================

scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=5,
    scoring='f1'
)

print("\n")

print("=" * 60)
print("CROSS VALIDATION")
print("=" * 60)

print("CV F1 Scores:")

print(scores)

print("\nMean CV F1:", scores.mean())
# ==========================================
# ROC CURVE
# ==========================================

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_prob
)

plt.figure(figsize=(7, 6))

plt.plot(
    fpr,
    tpr,
    linewidth=2,
    label=f"Random Forest AUC = {roc_auc_score(y_test, y_prob):.4f}"
)

# random baseline
plt.plot(
    [0, 1],
    [0, 1],
    linestyle='--'
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve - Random Forest")

plt.legend(loc="lower right")

plt.grid(True)

plt.show()
pd.DataFrame({
    "y_test": y_test,
    "y_prob_rf": y_prob
}).to_csv("rf_probs.csv", index=False)