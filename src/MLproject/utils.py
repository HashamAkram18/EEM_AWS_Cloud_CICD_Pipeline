import os 
import sys
import pandas as pd
# from cassandra.cluster import Cluster
# from cassandra.auth import PlainTextAuthProvider
import json
from src.MLproject.exception import CustomException

from src.MLproject.logger import logging
import pandas as pd 
import pymysql
import pickle
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score 
from dotenv import load_dotenv

load_dotenv()

host = os.getenv("host")
user = os.getenv("user")
password = os.getenv("password")
db = os.getenv("db")

def read_MySql_data():
    logging.info("Reading SQL database started")
    try:
        mydb=pymysql.connect(
            host=host,
            user=user,
            password=password,
            db=db
        )
        logging.info("connection established", mydb)
        df=pd.read_sql_query('', mydb)
        print(df.head)
        return df
    
    except Exception as ex:
        raise CustomException(ex)


# def fetch_cassandra_data():
   
#    logging.info("Reading Cassendra database started")
#    try:
#         """
#         Connects to Cassandra, fetches data from the specified table,
#         and returns it as a pandas DataFrame, using hardcoded credentials and query.
#         """

#         with open("Your-db-token.json") as f:
#             secrets = json.load(f)

#         cloud_config = {'secure_connect_bundle': 'secure-connect-Your-db.zip'}

#         auth_provider = PlainTextAuthProvider(secrets["clientId"], secrets["secret"])
#         cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
#         session = cluster.connect()

#         query = "SELECT * FROM your_keyspace.your_datatable;"
#         df = pd.DataFrame(list(session.execute(query)))

#         return df
#    except Exception as ex:
#         raise CustomException(ex)
   
   # Set up logging
logging.basicConfig(level=logging.INFO)

def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}

        for model_name, model in models.items():
            logging.info(f"Evaluating {model_name}...")
            gs = GridSearchCV(model, params[model_name], cv=3)  # Use GridSearchCV for all models

            gs.fit(X_train, y_train)
            best_params = gs.best_params_
            model.set_params(**best_params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2_scores = [r2_score(y_true, y_pred_col) for y_true, y_pred_col in zip(y_test.T, y_pred.T)]
            test_model_score = np.mean(r2_scores)
            report[model_name] = test_model_score

        return report
    
    except Exception as e:
        logging.error("An error occurred during model evaluation: " + str(e))
        raise CustomException("Error in model evaluation", error_details=str(e)) from e


def save_object(file_path, obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
            
    except Exception as e:
        raise CustomException(sys,e)
    

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    
    except Exception as e:
        error_message = f"Error occurred while loading object from file: {file_path}. Error: {str(e)}"
        logging.error(error_message)
        raise CustomException(sys,e)