import os 
import sys
import pandas as pd
from src.MLproject.exception import CustomException
from src.MLproject.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join('Artifacts','model.pkl')
            preprocessing_path=os.path.join('Artifacts','preprocesser.pkl')
            print("Before Loading")

            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessing_path)
            print("After loading")

            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            
        
            return preds
        except Exception as e:
            raise CustomException(sys,e)


class CustomData:
    def __init__(self,
                 relative_compactness: float,
                 overall_height: float,
                 orientation_degrees: float,
                 glazing_area: float,
                 glazing_area_distribution_ratio: float,
                 glazing_orientation: float,
                 aspect_ratio: float,
                 total_area: float):
        
        self.relative_compactness = relative_compactness
        self.overall_height = overall_height
        self.orientation_degrees = orientation_degrees
        self.glazing_area = glazing_area
        self.glazing_area_distribution_ratio = glazing_area_distribution_ratio
        self.glazing_orientation = glazing_orientation
        self.aspect_ratio = aspect_ratio
        self.total_area = total_area


    
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Relative Compactness (ratio)": [self.relative_compactness],
                "Overall Height (m)": [self.overall_height],
                "Orientation (Degrees)": [self.orientation_degrees],
                "Glazing Area(m²)": [self.glazing_area],
                "Glazing Area Distribution (Ratio)": [self.glazing_area_distribution_ratio],
                "Glazing_Orientation": [self.glazing_orientation],
                "Aspect Ratio": [self.aspect_ratio],
                "Total Area": [self.total_area]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
