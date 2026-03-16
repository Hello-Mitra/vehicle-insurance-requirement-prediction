import os
import sys
import pandas as pd
import numpy as np
import dill
import yaml
from pandas import DataFrame
import json

from vehicle_insurance.exception import MyException
from vehicle_insurance.logger import logging


def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)

    except Exception as e:
        raise MyException(e, sys) from e


def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise MyException(e, sys) from e


def load_object(file_path: str) -> object:
    """
    Returns model/object from project directory.
    file_path: str location of file to load
    return: Model/Obj
    """
    try:
        with open(file_path, "rb") as file_obj:
            obj = dill.load(file_obj)
        return obj
    except Exception as e:
        raise MyException(e, sys) from e

def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise MyException(e, sys) from e


def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise MyException(e, sys) from e


def save_object(file_path: str, obj: object) -> None:
    logging.info("Entered the save_object method of utils")

    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

        logging.info("Exited the save_object method of utils")

    except Exception as e:
        raise MyException(e, sys) from e
    

def save_json(file_path: str, data: dict) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)

    except Exception as e:
        raise MyException(e, sys)
    

def create_features(df):
        channel_mean = df.groupby("Policy_Sales_Channel")["Response"].mean()
        df["channel_target"] = df["Policy_Sales_Channel"].map(channel_mean)

        region_mean = df.groupby("Region_Code")["Response"].mean()
        df["region_target"] = df["Region_Code"].map(region_mean)

        vehicle_age_map = {
            "< 1 Year": 0,
            "1-2 Year": 1,
            "> 2 Years": 2
        }

        df["Vehicle_Age_num"] = df["Vehicle_Age"].map(vehicle_age_map)

        df["Age_x_Vehicle_Age"] = df["Age"] * df["Vehicle_Age_num"]

        channel_freq = df["Policy_Sales_Channel"].value_counts()
        df["Channel_freq"] = df["Policy_Sales_Channel"].map(channel_freq)

        region_freq = df["Region_Code"].value_counts()
        df["Region_Code_freq"] = df["Region_Code"].map(region_freq)

        df["Age_x_Channel"] = df["Age"] * df["Channel_freq"]

        df["Vehicle_Damage_bin"] = (df["Vehicle_Damage"] == "Yes").astype(int)

        df["Damage_x_NotInsured"] = ((df["Vehicle_Damage"] == "Yes") &(df["Previously_Insured"] == 0)).astype(int)

        df["Age_x_Region"] = df["Age"] * df["Region_Code_freq"]

        df["Damage_Age"] = (df["Vehicle_Damage"] == "Yes") & (df["Vehicle_Age"] == "> 2 Years")

        df.drop(columns=['Policy_Sales_Channel','Region_Code','Vehicle_Age','Vehicle_Damage'], inplace=True)
        logging.info("Data Transformation completed")
        logging.info(f"Transformed dataframe has {df.shape[1]} columns : {[col for col in df.columns]}")

        return df


def change_dtype(df):
        df = df.astype({
        "Gender" : "category",
        "Driving_License" : "category",
        "Previously_Insured" : "category",
        "Response" : "category",
        "Vehicle_Damage_bin" : "category"
        })
        return df


def read_data(file_path) -> pd.DataFrame:
        
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)
# def drop_columns(df: DataFrame, cols: list)-> DataFrame:

#     """
#     drop the columns form a pandas DataFrame
#     df: pandas DataFrame
#     cols: list of columns to be dropped
#     """
#     logging.info("Entered drop_columns methon of utils")

#     try:
#         df = df.drop(columns=cols, axis=1)

#         logging.info("Exited the drop_columns method of utils")
        
#         return df
#     except Exception as e:
#         raise MyException(e, sys) from e