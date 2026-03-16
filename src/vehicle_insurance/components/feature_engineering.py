from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class CustomFeatureEngineer(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.channel_mean = None
        self.region_mean = None
        self.channel_freq = None
        self.region_freq = None

    def fit(self, X, y=None):

        df = X.copy()

        if y is not None:
            df["Response"] = y

        # Target encoding
        self.channel_mean = df.groupby("Policy_Sales_Channel")["Response"].mean()
        self.region_mean = df.groupby("Region_Code")["Response"].mean()

        # Frequency encoding
        self.channel_freq = df["Policy_Sales_Channel"].value_counts()
        self.region_freq = df["Region_Code"].value_counts()

        return self

    def transform(self, X):

        df = X.copy()

        vehicle_age_map = {
            "< 1 Year": 0,
            "1-2 Year": 1,
            "> 2 Years": 2
        }

        df["channel_target"] = df["Policy_Sales_Channel"].map(self.channel_mean).fillna(self.channel_mean.mean())
        df["region_target"] = df["Region_Code"].map(self.region_mean).fillna(self.region_mean.mean())

        df["Vehicle_Age_num"] = df["Vehicle_Age"].map(vehicle_age_map)

        df["Channel_freq"] = df["Policy_Sales_Channel"].map(self.channel_freq).fillna(0)
        df["Region_Code_freq"] = df["Region_Code"].map(self.region_freq).fillna(0)

        df["Age_x_Vehicle_Age"] = df["Age"] * df["Vehicle_Age_num"]
        df["Age_x_Channel"] = df["Age"] * df["Channel_freq"]
        df["Age_x_Region"] = df["Age"] * df["Region_Code_freq"]

        df["Vehicle_Damage_bin"] = (df["Vehicle_Damage"] == "Yes").astype(int)

        df["Damage_x_NotInsured"] = (
            (df["Vehicle_Damage"] == "Yes") &
            (df["Previously_Insured"] == 0)
        ).astype(int)

        df["Damage_Age"] = (
            (df["Vehicle_Damage"] == "Yes") &
            (df["Vehicle_Age"] == "> 2 Years")
        )

        df = df.astype({
            "Gender": "category",
            "Driving_License": "category",
            "Previously_Insured": "category",
            "Vehicle_Damage_bin": "category"
        })

        df.drop(
            columns=[
                "Policy_Sales_Channel",
                "Region_Code",
                "Vehicle_Age",
                "Vehicle_Damage"
            ],
            inplace=True
        )

        return df