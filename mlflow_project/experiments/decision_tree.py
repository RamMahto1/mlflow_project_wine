# experiments/decision_tree.py

import mlflow
import mlflow.sklearn
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils.data_loader import load_data
import numpy as np

def run():
    X_train, X_test, y_train, y_test = load_data()

    with mlflow.start_run(run_name="DecisionTreeRegressor"):
        model = DecisionTreeRegressor()
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

        # Log metrics to MLflow
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)

        # Log the model
        mlflow.sklearn.log_model(model, "decision_tree_model")
