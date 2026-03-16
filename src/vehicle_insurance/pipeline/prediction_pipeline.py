import sys
from vehicle_insurance.entity.config_entity import VehiclePredictorConfig
from vehicle_insurance.entity.s3_estimator import Proj1Estimator
from vehicle_insurance.exception import MyException
from vehicle_insurance.logger import logging
from pandas import DataFrame


class VehicleData:
    def __init__(
        self,
        Gender,
        Age,
        Driving_License,
        Region_Code,
        Previously_Insured,
        Vehicle_Age,
        Vehicle_Damage,
        Annual_Premium,
        Policy_Sales_Channel,
        Vintage
    ):
        """
        Vehicle Data constructor
        Input: all features of the trained model for prediction
        """
        try:
            self.Gender = Gender
            self.Age = Age
            self.Driving_License = Driving_License
            self.Region_Code = Region_Code
            self.Previously_Insured = Previously_Insured
            self.Vehicle_Age = Vehicle_Age
            self.Vehicle_Damage = Vehicle_Damage
            self.Annual_Premium = Annual_Premium
            self.Policy_Sales_Channel = Policy_Sales_Channel
            self.Vintage = Vintage

        except Exception as e:
            raise MyException(e, sys) from e

    def get_vehicle_input_data_frame(self)-> DataFrame:
        """
        This function returns a DataFrame from VehicleData class input
        """
        try:
            
            vehicle_input_dict = self.get_vehicle_data_as_dict()
            return DataFrame(vehicle_input_dict)
        
        except Exception as e:
            raise MyException(e, sys) from e


    def get_vehicle_data_as_dict(self):
        """
        This function returns a dictionary from VehicleData class input
        """
        logging.info("Entered get_vehicle_data_as_dict method as VehicleData class")

        try:
            input_data = {
                "Gender": [self.Gender],
                "Age": [self.Age],
                "Driving_License": [self.Driving_License],
                "Region_Code": [self.Region_Code],
                "Previously_Insured": [self.Previously_Insured],
                "Vehicle_Age": [self.Vehicle_Age],
                "Vehicle_Damage": [self.Vehicle_Damage],
                "Annual_Premium": [self.Annual_Premium],
                "Policy_Sales_Channel": [self.Policy_Sales_Channel],
                "Vintage": [self.Vintage]

            }

            logging.info("Created vehicle data dict")
            logging.info("Exited get_vehicle_data_as_dict method as VehicleData class")
            return input_data

        except Exception as e:
            raise MyException(e, sys) from e

class VehicleDataClassifier:
    def __init__(self,prediction_pipeline_config: VehiclePredictorConfig = VehiclePredictorConfig(),) -> None:
        """
        :param prediction_pipeline_config: Configuration for prediction the value
        """
        try:
            self.prediction_pipeline_config = prediction_pipeline_config
        except Exception as e:
            raise MyException(e, sys)

    def predict(self, dataframe):
        """
        This is the method of VehicleDataClassifier
        Returns: Prediction in string format
        """
        try:
            logging.info("Entered predict method of VehicleDataClassifier class")
            model = Proj1Estimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path,
            )
            prediction = model.predict(dataframe)[0]
            probs = model.predict_proba(dataframe)[0]

            label_map = {
                0: "Not Interested",
                1: "Interested"
            }

            prob_dict = {
                label_map[i]: float(probs[i]) for i in range(len(probs))
            }

            response = {
                "prediction": label_map[int(prediction)],
                "confidence": float(max(probs)),
                "class_probabilities": prob_dict
            }

            return response
        
        except Exception as e:
            raise MyException(e, sys)