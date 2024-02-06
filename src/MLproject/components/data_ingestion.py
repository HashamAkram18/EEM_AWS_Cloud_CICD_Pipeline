import os 
import sys
from src.MLproject.exception import CustomException
from src.MLproject.logger import logging
import pandas as pd 
from src.MLproject.utils import read_MySql_data
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.MLproject.components.data_transformation import DataTransformation
from src.MLproject.components.data_transformation import DataTransformationconfig

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('Artifacts', 'train.csv')
    test_data_path: str = os.path.join('Artifacts', 'test.csv')
    raw_data_path: str = os.path.join('Artifacts', 'raw.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            # # choice(1):code to read form the mysql data
            # df=read_MySql_data()            
            # logging.info("Reading completed from MySQL database")
            
            ## choice(2): code to read csv file data from local matchine files
            df = pd.read_csv(os.path.join('Notebook', 'Data', 'ENB2012_data.csv'))
            
             # Rename the columns
            df.columns = ['Relative Compactness (ratio)', 'Surface Area(m²)', 'Wall Area(m²)', 'Roof Area(m²)', 
                           'Overall Height (m)', 'Orientation (Degrees)', 'Glazing Area(m²)', 
                           'Glazing Area Distribution (Ratio)', 'Heating Load (kWh)', 'Cooling Load (kWh)']


            logging.info("Reading the data from the local Files in system")
            logging.info('''Mapping the columns names as follow:
                         
                                                                X1	Relative Compactness
                                                                X2	Surface Area
                                                                X3	Wall Area
                                                                X4	Roof Area
                                                                X5	Overall Height
                                                                X6	Orientation
                                                                X7	Glazing Area
                                                                X8	Glazing Area Distribution
                                                                y1	Heating Load
                                                                y2	Cooling Load

                        ''')


            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            # code to split and train-test data
            train_set, test_set = train_test_split(df,test_size=0.2, random_state=24 )
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data ingestion is completed") 

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)    