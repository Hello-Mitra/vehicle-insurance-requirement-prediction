from vehicle_insurance.entity.config_entity import ModelEvaluationConfig
from vehicle_insurance.entity.artifact_entity import ModelTrainerArtifact, DataIngestionArtifact, ModelEvaluationArtifact
from sklearn.metrics import recall_score
from vehicle_insurance.exception import MyException
from vehicle_insurance.constants import TARGET_COLUMN
from vehicle_insurance.logger import logging
from vehicle_insurance.utils.main_utils import load_object, read_data, create_features, change_dtype

import sys
import pandas as pd
from typing import Optional
from vehicle_insurance.entity.s3_estimator import Proj1Estimator
from dataclasses import dataclass


@dataclass
class EvaluateModelResponse:
    trained_model_recall_score: float
    best_model_recall_score: float
    is_model_accepted: bool
    difference: float


class ModelEvaluation:

    def __init__(self, model_eval_config: ModelEvaluationConfig, data_ingestion_artifact: DataIngestionArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise MyException(e, sys) from e

    def get_best_model(self) -> Optional[Proj1Estimator]:
        """
        Method Name :   get_best_model
        Description :   This function is used to get model from production stage.
        
        Output      :   Returns model object if available in s3 storage
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            bucket_name = self.model_eval_config.bucket_name
            model_path=self.model_eval_config.s3_model_key_path
            proj1_estimator = Proj1Estimator(bucket_name=bucket_name,
                                               model_path=model_path)

            if proj1_estimator.is_model_present(model_path=model_path):
                return proj1_estimator
            logging.info("No production model found in S3.")
            return None
        except Exception as e:
            raise  MyException(e,sys)
    
    
    def evaluate_model(self) -> EvaluateModelResponse:
        """
        Method Name :   evaluate_model
        Description :   This function is used to evaluate trained model 
                        with production model and choose best model 
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info("Loading test dataset for evaluation.")
            test_df = read_data(self.data_ingestion_artifact.test_file_path)

            test_df = create_features(test_df)
            test_df = change_dtype(test_df)

            y = test_df[TARGET_COLUMN]
            x = test_df.drop(columns=[TARGET_COLUMN])

            logging.info("Test data loaded and now transforming it for prediction...")

            trained_model = load_object(file_path=self.model_trainer_artifact.trained_model_file_path)
            logging.info("Trained model loaded/exists.")
            trained_model_recall_score = self.model_trainer_artifact.metric_artifact.recall_score
            logging.info(f"Recall_Score for this model: {trained_model_recall_score}")

            best_model_recall_score=None
            best_model = self.get_best_model()

            if best_model is not None:
                logging.info(f"Computing F1_Score for production model..")
                y_hat_best_model = best_model.predict(x)
                best_model_recall_score = recall_score(y, y_hat_best_model)
                logging.info(f"Recall_Score-Production Model: {best_model_recall_score}, Recall_Score-New Trained Model: {trained_model_recall_score}")
            
            tmp_best_model_score = 0 if best_model_recall_score is None else best_model_recall_score
            result = EvaluateModelResponse(trained_model_recall_score=trained_model_recall_score,
                                           best_model_recall_score=best_model_recall_score,
                                           is_model_accepted=trained_model_recall_score > tmp_best_model_score,
                                           difference=trained_model_recall_score - tmp_best_model_score
                                           )
            logging.info(f"Result: {result}")
            return result

        except Exception as e:
            raise MyException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Method Name :   initiate_model_evaluation
        Description :   This function is used to initiate all steps of the model evaluation
        
        Output      :   Returns model evaluation artifact
        On Failure  :   Write an exception log and then raise an exception
        """  
        try:
            evaluate_model_response = self.evaluate_model()
            s3_model_path = self.model_eval_config.s3_model_key_path

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                s3_model_path=s3_model_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=evaluate_model_response.difference)

            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
        except Exception as e:
            raise MyException(e, sys) from e