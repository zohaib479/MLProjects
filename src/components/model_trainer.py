from catboost import CatBoostRegressor
from sklearn.ensemble import(
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor
)
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from src.exception import CustomException
from src.logger import logger
from dataclasses import dataclass
import sys
import os 
from src.utils import save_object,evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_training(self,train_array,test_array):
        try:
            logger.info("split Training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Classifier": KNeighborsRegressor(),
                "XGBClassifier": XGBRegressor(),
                "CatBoosting Classifier": CatBoostRegressor(verbose=False),
                "AdaBoost Classifier": AdaBoostRegressor(),
            }
            params = {
                    "Random Forest": {
                        "n_estimators": [100, 200, 300],
                        "max_depth": [None, 20, 30],
                        "min_samples_split": [2, 3, 4]
                    },
                    "Decision Tree": {
                        "criterion": ["squared_error", "friedman_mse"],
                        "max_depth": [None, 10, 20],
                        "min_samples_split": [2, 4, 6]
                    },
                    "Gradient Boosting": {
                        "learning_rate": [0.01, 0.1, 0.2],
                        "n_estimators": [100, 150, 200],
                        "subsample": [0.8, 1.0],
                        "max_depth": [3, 5, 10]
                    },
                    "Linear Regression": {                    },
                    "K-Neighbors Regressor": {
                        "n_neighbors": [3, 5, 7],
                        "weights": ["uniform", "distance"],
                        "algorithm": ["auto", "ball_tree", "kd_tree"]
                    },
                    "XGBoost": {
                        "learning_rate": [0.01, 0.1, 0.2],
                        "n_estimators": [100, 150, 200],
                        "max_depth": [3, 5, 7]
                    },
                    "CatBoost": {
                        "depth": [6, 8, 10],
                        "learning_rate": [0.01, 0.1, 0.2],
                        "iterations": [100, 200]
                    },
                    "AdaBoost": {
                        "n_estimators": [50, 100, 150],
                        "learning_rate": [0.01, 0.1, 1.0]
                    }
                }

            model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,parameters=params )
            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)] 
            best_model=models[best_model_name]
            if best_model_score<0.6:
                raise CustomException("NO best Model Found_")
            
            logger.info(
                f"Best Found model on both training and testing dataset"
            )
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predicited=best_model.predict(X_test)
            r2_scoree=r2_score(y_test,predicited)
            return r2_scoree
        except Exception as e:
            raise CustomException(sys,e)