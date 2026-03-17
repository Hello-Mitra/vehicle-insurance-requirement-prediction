from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi import HTTPException

from vehicle_insurance.pipeline.prediction_pipeline import (
    VehicleData,
    VehicleDataClassifier
)

from vehicle_insurance.constants import APP_HOST, APP_PORT
from vehicle_insurance.schema.user_input import UserInput
from vehicle_insurance.schema.prediction_response import PredictionResponse
from vehicle_insurance.pipeline.training_pipeline import TrainPipeline

app = FastAPI(title="Vehicle Insurance Prediction API")

# Allow all origins for Cross-Origin Resource Sharing (CORS)
origins = ["*"]

# Configure middleware to handle CORS, allowing requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def home():
    return {"message": "Vehicle Insurance Prediction API running"}


@app.get("/health")
async def health_check():
    return {"status":"OK"}

# Route to trigger the model training process
@app.get("/train")
async def trainRouteClient():
    """
    Endpoint to initiate the model training pipeline.
    """
    try:
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
        return {"message": "Training successful"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictionResponse)
async def predictRouteClient(data: UserInput):
    """
    Endpoint to initiate the model training pipeline.
    """
    try:
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))