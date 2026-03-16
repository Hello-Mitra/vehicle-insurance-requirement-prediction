from pydantic import BaseModel, Field, computed_field, field_validator
from typing import Literal, Annotated

# Pydantic Model to validate incoming data 
class UserInput(BaseModel):

    Gender : Annotated[Literal['Male', 'Female'], Field(..., description='Gender of User')]
    Age : Annotated[int, Field(..., gt=0, lt=120, description="Age of User")]
    Driving_License : Annotated[Literal[0, 1], Field(..., description="Does the User have Driving License?")]
    Region_Code : Annotated[int, Field(..., gt=0, lt=52, description='Unique code for the Region of the User')]
    Previously_Insured : Annotated[Literal[0, 1], Field(..., description="Was the User previously Insured??")]
    Vehicle_Age: Annotated[Literal['> 2 Years', '1-2 Year', '< 1 Year'],Field(..., description="Vehicle Age category")]
    Vehicle_Damage : Annotated[Literal["Yes", "No"], Field(..., description="Was the Vehicle damaged before?")]
    Annual_Premium : Annotated[float, Field(..., gt=0, description='The Amount User needs to pay as premium in the year')]
    Policy_Sales_Channel : Annotated[int, Field(..., gt=0, lt=165, description='Anonymized Code for the channel of outreaching to the customer ie. Different Agents, Over Mail, Over Phone, In Person, etc.')]
    Vintage : Annotated[int, Field(..., gt=0, description="Number of Days, Customer has been associated with the company")]

    