import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

from preprocessing import load_preprocessed_data
from Linear_regression import train_and_predict as lr_run
from polynomial_regression import train_and_predict as poly_run
from mlp_regression import train_and_predict as mlp_run
from random_forest import train_and_predict as rf_run
from xgboost_alg import train_and_predict as xgb_run

os.makedirs("plots", exist_ok=True)

models = {
    "Linear Regression": lr_run,
    "Polynomial Regression": poly_run,
    "MLP": mlp_run,
    "Random Forest": rf_run,
    "XGBoost": xgb_run
}

results = {}

X_raw, y, preprocessor = load_preprocessed_data()
X_train_raw, X_val_raw, y_train, y_val = train_test_split(
    X_raw, y, test_size=0.2, random_state=42
)

X_train = preprocessor.fit_transform(X_train_raw)
X_val = preprocessor.transform(X_val_raw)

for name, func in models.items():
    model = func(return_model=True)
    preds = model.predict(X_val)

    mse = mean_squared_error(y_val, preds)
    mae = mean_absolute_error(y_val, preds)
    results[name] = (mse, mae)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_val, preds, alpha=0.5)
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], "r--")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{name} – Actual vs Predicted")
    plt.savefig(f"plots/{name}_scatter.png")
    plt.close()

    residuals = y_val - preds
    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=40, alpha=0.7)
    plt.title(f"{name} – Residuals")
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.savefig(f"plots/{name}_residuals.png")
    plt.close()

model_names = list(results.keys())
mses = [results[m][0] for m in model_names]
maes = [results[m][1] for m in model_names]

plt.figure(figsize=(8, 5))
plt.bar(model_names, mses)
plt.ylabel("MSE")
plt.title("Model Comparison – MSE")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("plots/comparison_mse.png")
plt.close()

plt.figure(figsize=(8, 5))
plt.bar(model_names, maes)
plt.ylabel("MAE")
plt.title("Model Comparison – MAE")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("plots/comparison_mae.png")
plt.close()

print("Done. Plots saved in /plots/")
