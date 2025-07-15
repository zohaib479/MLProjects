import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
import os
import dill
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from src.exception import CustomException
def evaluate_model(X_train, y_train, X_test, y_test, models: dict, parameters: dict):
    try:
        report = {}

        for model_name in models:
            model = models[model_name]
            param_grid = parameters.get(model_name, {})

            print(f"Training model: {model_name}")

            if param_grid:
                grid = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    cv=3,
                    n_jobs=-1,
                    verbose=2
                )
                grid.fit(X_train, y_train)
                best_model = grid.best_estimator_
            else:
                model.fit(X_train, y_train)
                best_model = model

            # Predictions
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            # Scoring
            train_score = r2_score(y_train, y_train_pred)
            test_score = r2_score(y_test, y_test_pred)

            print(f"{model_name} Test R2 Score: {test_score}")
            report[model_name] = test_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)