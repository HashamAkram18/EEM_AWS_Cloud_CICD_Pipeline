import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.MLproject.utils import save_object
from src.MLproject.exception import CustomException
from src.MLproject.logger import logging


@dataclass
class DataTransformationconfig:
    preprocessor_obj_file_path = os.path.join('Artifacts', 'preprocesser.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationconfig()

    def get_data_transformer_object(self):
        """
        This function is responsible for data transformation
        """
        try:
            # Updated numerical and categorical columns
            numerical_columns = [
                'Relative Compactness (ratio)',
                'Orientation (Degrees)',
                'Glazing Area Distribution (Ratio)',
                'Glazing_Orientation',  # Added feature
                'Aspect Ratio',  # Added feature
                'Total Area'  # Added feature
            ]
            categorical_columns = []  # No categorical columns

            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy="median")),
                ('scaler', StandardScaler())
            ])

            # No categorical pipeline needed

            preprocesser = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns)
                ]
            )
            return preprocesser

        except Exception as e:
            raise CustomException(sys, e)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading the train and test file")

            preprocessing_obj = self.get_data_transformer_object()

            # Updated target columns
            target_columns = ['Heating Load (kWh)', 'Cooling Load (kWh)']

            ## Feature engineering on both train and test sets
            for df in [train_df, test_df]:
                df['Glazing_Orientation'] = df['Glazing Area(m²)'] * df['Orientation (Degrees)']
                df['Aspect Ratio'] = df['Surface Area(m²)'] / df['Overall Height (m)']
                df['Total Area'] = df['Surface Area(m²)'] + df['Wall Area(m²)'] + df['Roof Area(m²)']
                df = df.drop(['Surface Area(m²)', 'Wall Area(m²)', 'Roof Area(m²)'], axis=1,inplace=True )

            ## Divide the data into features and targets
            input_features_train_df = train_df.drop(columns=target_columns, axis=1)
            target_features_train_df = train_df[target_columns]
            input_features_test_df = test_df.drop(columns=target_columns, axis=1)
            target_features_test_df = test_df[target_columns]
            
            # print("First 10 rows of input_features_train_df:\n", input_features_train_df.columns)
            # print("First 10 rows of target_features_train_df:\n", target_features_train_df.columns)
            # print("First 10 rows of input_features_train_df:\n", input_features_train_df.head(10))            
            # print("First 10 rows of target_features_train_df:\n", target_features_train_df.head(10))
            logging.info("Applying Preprocessing on training and test dataframe")
    
            input_feature_train_transformed_arr = preprocessing_obj.fit_transform(input_features_train_df)
            input_feature_test_transformed_arr = preprocessing_obj.transform(input_features_test_df)
            
            train_arr = np.c_[input_feature_train_transformed_arr, target_features_train_df]
            test_arr = np.c_[input_feature_test_transformed_arr, target_features_test_df]
            
            # print("train_arr shape ",train_arr.shape)
            # print("test_arr shape ",test_arr.shape)
            

            logging.info(f"Saved preprocessing object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(sys,e)
        


        
