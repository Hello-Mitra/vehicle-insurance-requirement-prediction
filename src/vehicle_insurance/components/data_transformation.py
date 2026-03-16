import sys
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from vehicle_insurance.constants import TARGET_COLUMN, SCHEMA_FILE_PATH, CURRENT_YEAR
from vehicle_insurance.entity.config_entity import DataTransformationConfig
from vehicle_insurance.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from vehicle_insurance.exception import MyException
from vehicle_insurance.logger import logging
from vehicle_insurance.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file, read_data, create_features, change_dtype


class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys)


    def get_data_transformer_object(self) -> Pipeline:
        """
        Creates and returns a data transformer object for the data, 
        including gender mapping, dummy variable creation, column renaming,
        feature scaling, and type adjustments.
        """
        logging.info("Entered get_data_transformer_object method of DataTransformation class")

        try:
            # Initialize transformers
            numeric_transformer = StandardScaler()
            min_max_scaler = MinMaxScaler()
            logging.info("Transformers Initialized: StandardScaler-MinMaxScaler")

            # Load schema configurations
            num_features = self._schema_config['num_features']
            mm_columns = self._schema_config['mm_columns']
            ohe_columns = self._schema_config['ohe_columns']
            logging.info("Cols loaded from schema.")

            # Creating preprocessor pipeline
            preprocessor = ColumnTransformer(
                transformers=[
                    ("StandardScaler", numeric_transformer, num_features),
                    ("MinMaxScaler", min_max_scaler, mm_columns),
                    ('OneHotEncoder', OneHotEncoder(drop='first', sparse_output=False), ohe_columns)
                ],
                remainder='passthrough'  # Leaves other columns as they are
            )

            # preprocessor.set_output(transform="pandas")
            # When your FastAPI endpoint receives data, you can simply do:
            # transformed_df = preprocessor.transform(input_df)
            # prediction = model.predict(transformed_df)
            # No manual feature alignment needed.
            # preprocessor.get_feature_names_out()
    
            # Wrapping everything in a single pipeline
            final_pipeline = Pipeline(steps=[("Preprocessor", preprocessor)])
            logging.info("Final Pipeline Ready!!")
            logging.info("Exited get_data_transformer_object method of DataTransformation class")
            return final_pipeline

        except Exception as e:
            logging.exception("Exception occurred in get_data_transformer_object method of DataTransformation class")
            raise MyException(e, sys) from e
        

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Initiates the data transformation component for the pipeline.
        """
        try:
            logging.info("Data Transformation Started !!!")
            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)

            # Load train and test data
            train_df = read_data(file_path=self.data_ingestion_artifact.trained_file_path)
            test_df = read_data(file_path=self.data_ingestion_artifact.test_file_path)
            logging.info("Train-Test data loaded")

            # Apply custom transformations in specified sequence
            input_feature_train_df = create_features(train_df)
            input_feature_train_df = change_dtype(input_feature_train_df)

            input_feature_test_df = create_features(test_df)
            input_feature_test_df = change_dtype(input_feature_test_df)
            logging.info("Custom transformations applied to train and test data")

            target_feature_train_df = input_feature_train_df[TARGET_COLUMN]
            input_feature_train_df = input_feature_train_df.drop(columns=[TARGET_COLUMN])

            target_feature_test_df = input_feature_test_df[TARGET_COLUMN]
            input_feature_test_df = input_feature_test_df.drop(columns=[TARGET_COLUMN])
            logging.info("Input and Target cols defined for both train and test df.")

            logging.info("Starting data transformation")
            preprocessor = self.get_data_transformer_object()
            logging.info("Got the preprocessor object")

            logging.info("Initializing transformation for Training-data")
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            logging.info("Initializing transformation for Testing-data")
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)
            logging.info("Transformation done end to end to train-test df.")

            logging.info("Applying SMOTEENN for handling imbalanced dataset.")
            smt = SMOTEENN(sampling_strategy="minority")
            input_feature_train_final, target_feature_train_final = smt.fit_resample(
                input_feature_train_arr, target_feature_train_df
            )
            
            input_feature_test_final = input_feature_test_arr
            target_feature_test_final = target_feature_test_df
            logging.info("SMOTEENN applied to train df.")

            train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final)]
            test_arr = np.c_[input_feature_test_final, np.array(target_feature_test_final)]
            logging.info("feature-target concatenation done for train-test df.")

            # train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            # test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            # logging.info("feature-target concatenation done for train-test df.")

            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
            logging.info("Saving transformation object and transformed files.")

            logging.info("Data transformation completed successfully")
            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

        except Exception as e:
            raise MyException(e, sys) from e