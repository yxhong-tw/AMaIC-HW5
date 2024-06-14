import pandas as pd
import xgboost as xgb

from sklearn.metrics import accuracy_score
from typing import Tuple


def load_data() -> Tuple:
    train_x = pd.read_csv("data/train_X.csv", index_col=0)
    train_y = pd.read_csv("data/train_Y.csv", index_col=0).squeeze()
    test_x = pd.read_csv("data/test_X.csv", index_col=0)
    test_y = pd.read_csv("data/test_Y.csv", index_col=0).squeeze()

    dtrain = xgb.DMatrix(train_x, label=train_y)
    dtest = xgb.DMatrix(test_x, label=test_y)

    return dtrain, dtest, test_y


if __name__ == "__main__":
    params = {
        "objective": "binary:logistic",
        "max_depth": 6,
        "eta": 0.1,
        "eval_metric": "logloss",
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "device": "cuda"
    }

    dtrain, dtest, test_y = load_data()

    num_boost_round = 250

    eval_list = [
        (dtrain, "train"),
        (dtest, "eval"),
    ]

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=eval_list,
        early_stopping_rounds=5,
    )

    predictions = model.predict(data=dtest)
    predictions = [round(prediction) for prediction in predictions]

    accuracy = accuracy_score(y_true=test_y, y_pred=predictions)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
