import numpy as np
import os
import sys
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import  r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score


from src.components.data_transformation import DataTransformation
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def evaluate_models(self, X_train, y_train, X_test, y_test, models, params):
        """Evaluate multiple models with cross-validation and hyperparameter tuning."""
        try:
            report = {}

            for model_name, model in models.items():
                param_grid = params.get(model_name, {})
                logging.info(f"Performing GridSearchCV for {model_name}...")

                gs = GridSearchCV(model, param_grid, cv=3, scoring="r2", n_jobs=-1)
                gs.fit(X_train, y_train)

                best_model = gs.best_estimator_
                logging.info(f"Best parameters for {model_name}: {gs.best_params_}")

                # Evaluate the model on training and testing data
                y_train_pred = best_model.predict(X_train)
                y_test_pred = best_model.predict(X_test)

                train_r2 = r2_score(y_train, y_train_pred)
                test_r2 = r2_score(y_test, y_test_pred)

                # Perform cross-validation
                cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring="r2")

                report[model_name] = {
                    "train_r2": train_r2,
                    "test_r2": test_r2,
                    "cv_r2_mean": np.mean(cv_scores),
                    "cv_r2_std": np.std(cv_scores),
                    "best_model": best_model,
                }

            return report

        except Exception as e:
            raise CustomException(e, sys)
        

    def initiate_model_trainer(self, X_train, X_test, y_train, y_test):
        """Train and evaluate models, saving the best one."""
        try:
            logging.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            logging.info("Starting model training and evaluation.")

            models = {
                "Random Forest Regressor": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Gradient Boosting Regressor": GradientBoostingRegressor(),
                
            }

            params = {
                "Random Forest Regressor": {
                "n_estimators": [100, 200],
                "max_depth": [None, 20],
                "min_samples_split": [2, 4],
                "min_samples_leaf": [1, 3],
                },
                "Decision Tree": {
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 5],
                },

                "K-Neighbors Regressor": {
                    "n_neighbors": [3, 5, 10, 20],
                    "weights": ["uniform", "distance"],
                },
                
                "Gradient Boosting Regressor": {
                "n_estimators": [100, 200],
                "learning_rate": [0.001, 0.01, 0.1],
                "max_depth": [3, 5, 7],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2],
                },

            }

            evaluation_report = self.evaluate_models(X_train, y_train, X_test, y_test, models, params)

            # Log all model results
            logging.info("Model performance:")
            for model_name, metrics in evaluation_report.items():
                logging.info(
                    f"{model_name}: Train R2 = {metrics['train_r2']:.3f}, "
                    f"Test R2 = {metrics['test_r2']:.3f}, "
                    f"CV R2 Mean = {metrics['cv_r2_mean']:.3f} (+/- {metrics['cv_r2_std']:.3f})"
                )

            # Find the best model
            best_model_name = None
            best_model_r2 = -float("inf")
            best_model = None

            for model_name, metrics in evaluation_report.items():
                if metrics["test_r2"] > best_model_r2:
                    best_model_name = model_name
                    best_model_r2 = metrics["test_r2"] 
                    best_train_r2 = metrics["train_r2"]
                    best_model = metrics["best_model"]

            if best_model_r2 < 0.6:
                raise CustomException("No suitable model found with R2 score above the threshold.",sys)

            # Save the best model
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)
            logging.info(
                f"Best model: {best_model_name} with Train R2: {best_train_r2:.3f}, "
                f"Test R2: {best_model_r2:.3f}"
            )

            return evaluation_report
            

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        # Instantiate the data transformation class
        data_transformation = DataTransformation()

        # Define file paths for training and testing data
        train_file_path = os.path.join("data", "processed", "train.csv")
        test_file_path = os.path.join("data", "processed", "test.csv")

        # Perform data transformation
        X_train, X_test, y_train, y_test, preprocessor_path = data_transformation.initiate_data_transformation(
            train_file_path, test_file_path
        )
        logging.info(f"Transformation complete. Preprocessor saved at: {preprocessor_path}")

        # Train models
        trainer = ModelTrainer()
        evaluation_report = trainer.initiate_model_trainer(X_train, X_test, y_train, y_test)

        # Display all models' R2 scores
        print("Model Evaluation Results:")
        for model_name, metrics in evaluation_report.items():
            print(
                f"{model_name}: Train R2 = {metrics['train_r2']:.3f}, "
                f"Test R2 = {metrics['test_r2']:.3f}, "
                f"CV R2 Mean = {metrics['cv_r2_mean']:.3f} (+/- {metrics['cv_r2_std']:.3f})"
            )

    except Exception as e:
        raise CustomException("Error in main", sys)
