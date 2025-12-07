from preprocessing import load_preprocessed_data
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

def train_and_predict(return_model=False):
    X, y, preprocessor = load_preprocessed_data()
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train = preprocessor.fit_transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)

    model = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        alpha=0.0005,
        batch_size=16,
        early_stopping=True,
        max_iter=2000,
        random_state=8 #high performer from batch initilisation
    )

    model.fit(X_train, y_train)

    if return_model:
        return model

    preds = model.predict(X_val)
    print("MLP Regression Results")
    print("MSE:", mean_squared_error(y_val, preds))
    print("MAE:", mean_absolute_error(y_val, preds))

if __name__ == "__main__":
    train_and_predict()
