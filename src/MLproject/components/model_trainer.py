import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
# from catboost import CatBoostRegressor

from xgboost import XGBRegressor

from src.MLproject.exception import CustomException
from src.MLproject.logger import logging
from src.MLproject.utils import evaluate_models, save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("Artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-2],  # All columns except the last two (targets)
                train_array[:, -2:],  # Select both target columns
                test_array[:, :-2],
                test_array[:, -2:],
            )

            models = {
                "MultiOutputRandomForest": MultiOutputRegressor(RandomForestRegressor()),
                "MultiOutputGradientBoosting": MultiOutputRegressor(GradientBoostingRegressor()),
                "MultiOutputXGBRegressor": MultiOutputRegressor(XGBRegressor())
            }

            params = {
                "MultiOutputRandomForest": {
                    "estimator__n_estimators": [8, 16, 32, 64]
                },
                "MultiOutputGradientBoosting": {
                    "estimator__learning_rate": [0.1, 0.05, 0.01],
                    "estimator__n_estimators": [8, 16, 32]
                },
                "MultiOutputXGBRegressor": {
                    "estimator__learning_rate": [0.1, 0.05, 0.01],
                    "estimator__n_estimators": [16, 32, 64],
                    "estimator__max_depth": [3, 5, 7]
                }
            }
            
            model_report = evaluate_models(
                X_train, y_train, X_test, y_test, models, params
            )

            # Find the best model based on R2 score
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            print("This is the best model:")
            print(best_model_name)
            print("Best model score:", best_model_score)

            # Retrieve best model parameters (handling potential KeyErrors)
            actual_model = best_model_name
            try:
                best_params = params[actual_model]
            except KeyError:
                raise ValueError(f"Model '{actual_model}' not found in the 'params' dictionary.") from None

            # Check if best model score meets the 0.8 threshold
            if best_model_score < 0.8:
                raise CustomException("No best model found with score exceeding 0.8")

            logging.info(f"Best found model on both training and testing dataset")

            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Make predictions with the best model
            predicted = best_model.predict(X_test)

            # Calculate R2 score for evaluation
            r2_square_results = r2_score(y_test, predicted)

            return r2_square_results



        except Exception as e:
            logging.error("An error occurred during model training: " + str(e))
            raise CustomException(sys,e)
        