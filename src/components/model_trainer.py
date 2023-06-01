from dataclasses import dataclass
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
# Modelling
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from catboost import CatBoostRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("Artifacts", "model.pkl")

class ModelTrainer():
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info("Splting train and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]           
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                'Deccision Tree': DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Adaboost Resgressor": AdaBoostRegressor(),
                "Linear Regression": LinearRegression(),
                "Ridge Regression": Ridge(),
                "Lasso Regression": Lasso(),
                "Suport Vector Regression": SVR(),
                "k-nearest Regressor": KNeighborsRegressor()
            }

            models_report:dict = evaluate_models(X_train=X_train, y_train=y_train,
                                                 X_test=X_test, y_test=y_test, models=models)
            
            print(models_report)

            ## to get the best model score from dict

            best_model_score = max(sorted(models_report.values()))

            ## to get the best model name from dict

            best_model_name = list(models_report.keys())[
                list(models_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            print([best_model_name, best_model])

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            
            logging.info("Best model have been found on training and testing data")

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)
            
            r2_square = r2_score(y_test, predicted)

            return r2_square


        except Exception as e:
            raise CustomException(e, sys)

