from fastapi import FastAPI
from pydantic import BaseModel

from vehicle_insurance.pipeline.prediction_pipeline import (
    VehicleData,
    VehicleDataClassifier
)

from vehicle_insurance.schema.user_input import UserInput
from vehicle_insurance.schema.prediction_response import PredictionResponse

app = FastAPI(title="Vehicle Insurance Prediction API")


@app.get("/")
def home():
    return {"message": "Vehicle Insurance Prediction API running"}


@app.post("/predict", response_model=PredictionResponse)
def predict(data: UserInput):

    vehicle_data = VehicleData(
        Gender=data.Gender,
        Age=data.Age,
        Driving_License=data.Driving_License,
        Region_Code=data.Region_Code,
        Previously_Insured=data.Previously_Insured,
        Vehicle_Age=data.Vehicle_Age,
        Vehicle_Damage=data.Vehicle_Damage,
        Annual_Premium=data.Annual_Premium,
        Policy_Sales_Channel=data.Policy_Sales_Channel,
        Vintage=data.Vintage
    )

    vehicle_df = vehicle_data.get_vehicle_input_data_frame()

    model = VehicleDataClassifier()

    result = model.predict(vehicle_df)

    return PredictionResponse(**result)