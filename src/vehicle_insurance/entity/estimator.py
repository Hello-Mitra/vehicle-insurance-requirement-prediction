import sys

import pandas as pd
from pandas import DataFrame
from sklearn.pipeline import Pipeline

from vehicle_insurance.exception import MyException
from vehicle_insurance.logger import logging


class MyModel:
    def __init__(self, preprocessing_object: Pipeline, trained_model_object: object):
        """
        :param preprocessing_object: Input Object of preprocesser
        :param trained_model_object: Input Object of trained model 
        """
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self, dataframe: pd.DataFrame):
        """
        This method accepts raw input features as a pandas DataFrame.
        The saved preprocessing pipeline (feature engineering, encoding,
        scaling, etc.) is applied before the trained model generates predictions.
        """
        try:
            logging.info("Starting prediction process.")

            # Step 1: Apply scaling transformations using the pre-trained preprocessing object
            transformed_feature = self.preprocessing_object.transform(dataframe)

            # Step 2: Perform prediction using the trained model
            logging.info("Using the trained model to get predictions")
            predictions = self.trained_model_object.predict(transformed_feature)

            return predictions

        except Exception as e:
            logging.error("Error occurred in predict method", exc_info=True)
            raise MyException(e, sys) from e
        
    
    def predict_proba(self, dataframe: pd.DataFrame):
        """
        This method accepts raw input features as a pandas DataFrame,
        applies the saved preprocessing pipeline, and returns the
        predicted probability distribution across all possible classes.
        """
        try:
            logging.info("Starting probability prediction process.")

            # Apply preprocessing transformations
            logging.info("Applying preprocessing transformations.")
            transformed_feature = self.preprocessing_object.transform(dataframe)

            # Generate probability predictions
            logging.info("Generating class probability estimates.")
            probabilities = self.trained_model_object.predict_proba(transformed_feature)

            logging.info("Probability prediction completed successfully.")
            return probabilities

        except Exception as e:
            logging.error("Error occurred in predict_proba method.", exc_info=True)
            raise MyException(e, sys) from e


    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"