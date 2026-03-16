import sys
from typing import Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from vehicle_insurance.exception import MyException
from vehicle_insurance.logger import logging
from vehicle_insurance.utils.main_utils import load_numpy_array_data, load_object, save_object, save_json
from vehicle_insurance.entity.config_entity import ModelTrainerConfig
from vehicle_insurance.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact
from vehicle_insurance.entity.estimator import MyModel

class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        """
        :param data_transformation_artifact: Output reference of data transformation artifact stage
        :param model_trainer_config: Configuration for model training
        """
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    def get_model_object_and_report(self, train: np.array, test: np.array) -> Tuple[object, object]:
        """
        Method Name :   get_model_object_and_report
        Description :   This function trains a LogisticRegression Classifier with specified parameters
        
        Output      :   Returns metric artifact object and trained model object
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info("Training LogisticRegression Classifier with specified parameters")

            # Splitting the train and test data into features and target variables
            x_train, y_train, x_test, y_test = train[:, :-1], train[:, -1], test[:, :-1], test[:, -1]
            logging.info("train-test split done.")

            # Initialize LogisticRegression Classifier with specified parameters
            model = LogisticRegression(
                max_iter= self.model_trainer_config._max_iter,
                class_weight= self.model_trainer_config._class_weight,
                penalty= self.model_trainer_config._penalty,
                C = self.model_trainer_config._C,
                solver= self.model_trainer_config._solver,
                random_state = self.model_trainer_config._random_state,
                
            )

            # Fit the model
            logging.info("Model training going on...")
            model.fit(x_train, y_train)
            logging.info("Model training done.")

            # Predictions and evaluation metrics
            y_pred = model.predict(x_test)
            y_probs = model.predict_proba(x_test)[:, 1]
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_probs)

            logging.info(f"Accuracy: {accuracy}")
            logging.info(f"F1 Score: {f1}")
            logging.info(f"Precision: {precision}")
            logging.info(f"Recall: {recall}")
            logging.info(f"ROC-AUC: {roc_auc}")

            # Creating metric artifact
            metric_artifact = ClassificationMetricArtifact(
                accuracy_score=accuracy,
                f1_score=f1,
                precision_score=precision,
                recall_score=recall,
                roc_auc_score=roc_auc)
            
            return model, metric_artifact
        
        except Exception as e:
            raise MyException(e, sys) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")
        """
        Method Name :   initiate_model_trainer
        Description :   This function initiates the model training steps
        
        Output      :   Returns model trainer artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            # Load transformed train and test data
            train_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path)
            logging.info("train-test data loaded")
            
            # Train model and get metrics
            trained_model, metric_artifact = self.get_model_object_and_report(train=train_arr, test=test_arr)
            logging.info("Model object and artifact loaded.")
            
            # Load preprocessing object
            preprocessing_obj = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            logging.info("Preprocessing obj loaded.")

            # Check if the model's accuracy meets the expected threshold
            if accuracy_score(test_arr[:, -1], trained_model.predict(test_arr[:, :-1])) < self.model_trainer_config.expected_accuracy:
                logging.info("No model found with score above the base score")
                raise Exception("No model found with score above the base score")

            # Save the final model object that includes both preprocessing and the trained model
            logging.info("Saving new model as performace is better than previous one.")
            # Create metric dictionary
            metric_dict = {
                "accuracy": metric_artifact.accuracy_score,
                "f1_score": metric_artifact.f1_score,
                "precision": metric_artifact.precision_score,
                "recall": metric_artifact.recall_score,
                "roc_auc": metric_artifact.roc_auc_score
            }

            # Save model
            my_model = MyModel(
                preprocessing_object=preprocessing_obj,
                trained_model_object=trained_model
            )

            save_object(self.model_trainer_config.trained_model_file_path, my_model)

            # Save metrics
            save_json(self.model_trainer_config.metric_file_path, metric_dict)

            logging.info("Saved model and metrics successfully")
            logging.info("Saved final model object that includes both preprocessing and the trained model")

            # Create and return the ModelTrainerArtifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_file_path=self.model_trainer_config.metric_file_path,
                metric_artifact=metric_artifact,
            )
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        
        except Exception as e:
            raise MyException(e, sys) from e