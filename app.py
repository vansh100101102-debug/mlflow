import logging
import sys
import warnings

import dagshub
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Setup logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def eval_metrics(actual, predicted):
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    return rmse, mae, r2

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # 1. Initialize DagsHub tracking (Sets remote URI automatically)
    dagshub.init(repo_owner='vansh100101102', repo_name='mlflow', mlflow=True)

    # 2. Set your experiment name
    mlflow.set_experiment("Wine Quality Experiment")

    # Load dataset
    csv_url = (
        "https://raw.githubusercontent.com/mlflow/mlflow/master/"
        "tests/datasets/winequality-red.csv"
    )

    try:
        data = pd.read_csv(csv_url, sep=";")
    except Exception as error:
        logger.exception("Unable to download dataset: %s", error)
        sys.exit(1)

    # Train/test split
    train, test = train_test_split(
        data,
        test_size=0.25,
        random_state=40,
    )

    train_x = train.drop("quality", axis=1)
    test_x = test.drop("quality", axis=1)
    train_y = train["quality"]
    test_y = test["quality"]

    # Read hyperparameter arguments from CLI
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    # Train model
    model = ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        random_state=42,
    )
    model.fit(train_x, train_y)
    predicted_qualities = model.predict(test_x)

    rmse, mae, r2 = eval_metrics(test_y, predicted_qualities)
    signature = infer_signature(train_x, predicted_qualities)

    print(f"ElasticNet model (alpha={alpha:f}, l1_ratio={l1_ratio:f}):")
    print(f"  RMSE: {rmse}")
    print(f"  MAE: {mae}")
    print(f"  R2: {r2}")

    # 3. Log parameters, metrics, and model to DagsHub remote server
    with mlflow.start_run():
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=signature,
            registered_model_name="ElasticnetWineModel",
        )