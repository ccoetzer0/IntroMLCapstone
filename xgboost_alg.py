from preprocessing import load_preprocessed_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

def train_and_predict(return_model=False):
    X, y, preprocessor = load_preprocessed_data()
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train = preprocessor.fit_transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)

    model = XGBRegressor(
        n_estimators=1100,
        learning_rate=0.02,
        max_depth=3,
        min_child_weight=1,
        subsample=0.75,
        colsample_bytree=0.5,
        gamma=0.0,
        reg_lambda=2.0,
        objective="reg:squarederror",
        n_jobs=-1,
        random_state=42
    )

    model.fit(X_train, y_train)

    if return_model:
        return model

    preds = model.predict(X_val)
    print("XGBoost Results")
    print("MSE:", mean_squared_error(y_val, preds))
    print("MAE:", mean_absolute_error(y_val, preds))

if __name__ == "__main__":
    train_and_predict()
