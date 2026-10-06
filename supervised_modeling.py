# ============================================================
# Supervised Classification / Regression Modeling & Tuning
# ============================================================

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)

warnings.filterwarnings("ignore")


# ============================================================
# 1. Configuration
# ============================================================

DATA_PATH = "data/processed_data.csv"
TARGET = "target"

RANDOM_STATE = 42
TEST_SIZE = 0.20

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


# ============================================================
# 2. Load Dataset
# ============================================================

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())


# ============================================================
# 3. Separate Features and Target
# ============================================================

X = df.drop(columns=[TARGET])
y = df[TARGET]

print("\nFeature Shape:", X.shape)
print("Target Shape:", y.shape)

print("\nTarget Distribution:")
print(y.value_counts())


# ============================================================
# 4. Train-Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE
)

print("\nTraining Data:", X_train.shape)
print("Testing Data :", X_test.shape)


# ============================================================
# 5. Define Models
# ============================================================

models = {

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(
                max_iter=2000,
                random_state=RANDOM_STATE
            )
        )
    ]),

    "Random Forest": RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        random_state=RANDOM_STATE
    ),

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        (
            "model",
            SVC(
                probability=True,
                random_state=RANDOM_STATE
            )
        )
    ])
}


# ============================================================
# 6. Hyperparameter Grids
# ============================================================

param_grids = {

    "Logistic Regression": {
        "model__C": [0.01, 0.1, 1, 10, 100],
        "model__solver": [
            "liblinear",
            "lbfgs"
        ]
    },

    "Random Forest": {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2]
    },

    "Gradient Boosting": {
        "n_estimators": [100, 200],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [2, 3, 5]
    },

    "SVM": {
        "model__C": [0.1, 1, 10],
        "model__kernel": [
            "linear",
            "rbf"
        ],
        "model__gamma": [
            "scale",
            "auto"
        ]
    }
}


# ============================================================
# 7. Stratified K-Fold Cross Validation
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)


# ============================================================
# 8. Hyperparameter Optimization
# ============================================================

best_models = {}
results = {}

print("\n" + "=" * 70)
print("HYPERPARAMETER OPTIMIZATION")
print("=" * 70)

for name in models:

    print(f"\nTuning: {name}")

    grid_search = GridSearchCV(
        estimator=models[name],
        param_grid=param_grids[name],
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    best_models[name] = grid_search.best_estimator_

    print("\nBest Parameters:")
    print(grid_search.best_params_)

    print(
        "Best CV ROC-AUC:",
        round(grid_search.best_score_, 4)
    )


# ============================================================
# 9. Evaluate Models
# ============================================================

print("\n" + "=" * 70)
print("MODEL EVALUATION")
print("=" * 70)

comparison = []

for name, model in best_models.items():

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    results[name] = {
        "model": model,
        "y_pred": y_pred,
        "y_prob": y_prob
    }

    comparison.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC-AUC": roc_auc
    })

    print("\n" + "-" * 60)
    print(name)
    print("-" * 60)

    print(
        classification_report(
            y_test,
            y_pred
        )
    )


# ============================================================
# 10. Model Comparison Table
# ============================================================

comparison_df = pd.DataFrame(comparison)

comparison_df = comparison_df.sort_values(
    by="ROC-AUC",
    ascending=False
).reset_index(drop=True)

print("\nModel Comparison:")
print(comparison_df)

comparison_df.to_csv(
    "results/model_comparison.csv",
    index=False
)


# ============================================================
# 11. Confusion Matrices
# ============================================================

print("\nGenerating confusion matrices...")

for name, result in results.items():

    cm = confusion_matrix(
        y_test,
        result["y_pred"]
    )

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.title(
        f"Confusion Matrix - {name}"
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")

    filename = (
        name.lower()
        .replace(" ", "_")
        .replace("-", "")
    )

    plt.tight_layout()

    plt.savefig(
        f"results/confusion_matrix_{filename}.png",
        dpi=300
    )

    plt.show()
    plt.close()


# ============================================================
# 12. ROC-AUC Curves
# ============================================================

plt.figure(figsize=(10, 7))

for name, result in results.items():

    fpr, tpr, _ = roc_curve(
        y_test,
        result["y_prob"]
    )

    auc_score = roc_auc_score(
        y_test,
        result["y_prob"]
    )

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"{name} (AUC = {auc_score:.3f})"
    )


# Random classifier baseline
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(
    "ROC-AUC Comparison of Classification Models"
)

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "results/roc_auc_comparison.png",
    dpi=300
)

plt.show()
plt.close()


# ============================================================
# 13. Select Champion Model
# ============================================================

champion_name = comparison_df.iloc[0]["Model"]

champion_model = best_models[
    champion_name
]

champion_auc = comparison_df.iloc[0]["ROC-AUC"]

print("\n" + "=" * 70)
print("CHAMPION MODEL")
print("=" * 70)

print("Model:", champion_name)
print("ROC-AUC:", round(champion_auc, 4))


# ============================================================
# 14. Save Champion Model
# ============================================================

model_path = "models/champion_model.joblib"

joblib.dump(
    champion_model,
    model_path
)

print(
    f"\nChampion model saved to: {model_path}"
)


# ============================================================
# 15. Verify Serialized Model
# ============================================================

loaded_model = joblib.load(
    model_path
)

loaded_predictions = loaded_model.predict(
    X_test
)

print(
    "\nSerialized model verification:"
)

print(
    "Predictions:",
    loaded_predictions[:10]
)

print("\nModel successfully loaded and verified.")


# ============================================================
# 16. Final Summary
# ============================================================

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print("\nModel Ranking:")
print(comparison_df)

print(
    f"\nChampion Model: {champion_name}"
)

print(
    f"Champion ROC-AUC: {champion_auc:.4f}"
)

print(
    "\nAll results saved successfully."
)
