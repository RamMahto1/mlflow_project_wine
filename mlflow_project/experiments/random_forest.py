# experiments/random_forest.py

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils.data_loader import load_data
import numpy as np
import pandas as pd

def run():
    X_train, X_test, y_train, y_test = load_data()

    with mlflow.start_run(run_name="RandomForestRegressor"):
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Metrics
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)

        # Print metrics
        print(f"MAE: {mae:.2f}")
        print(f"MSE: {mse:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R2 Score: {r2:.2f}")

        # Log metrics
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)

        # Log the model
        input_example = pd.DataFrame([X_test.iloc[0]])
        mlflow.sklearn.log_model(model, "random_forest_model", input_example=input_example)
