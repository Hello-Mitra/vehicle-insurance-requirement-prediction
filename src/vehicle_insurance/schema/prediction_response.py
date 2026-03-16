from pydantic import BaseModel, Field
from typing import Dict, Annotated


class PredictionResponse(BaseModel):

    prediction: Annotated[str,Field(description="Predicted customer interest in vehicle insurance")]
    confidence: Annotated[float,Field(description="Confidence score of the predicted class (0-1)")]
    class_probabilities: Annotated[Dict[str, float],Field(description="Probability distribution across both classes")]