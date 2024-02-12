from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.MLproject.pipelines.prediction_pipeline import PredictPipeline, CustomData

application = Flask(__name__)

app = application

## Routing for home page 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home', methods=["GET","POST"])
def home():
    if request.method=='GET':
        return render_template('home.html')
    else:
        if all(request.form.values()):
            print("Form values:", request.form)
            data = CustomData(
                relative_compactness=float(request.form.get('Relative Compactness (ratio)')),
                overall_height=float(request.form.get('Overall Height (m)')),
                orientation_degrees=float(request.form.get('Orientation (Degrees)')),
                glazing_area=float(request.form.get('Glazing Area(m²)')),
                glazing_area_distribution_ratio=float(request.form.get('Glazing Area Distribution (Ratio)')),
                glazing_orientation=float(request.form.get('Glazing Orientation')),
                aspect_ratio=float(request.form.get('Aspect Ratio')),
                total_area=float(request.form.get('Total Area'))
            )

            pred_data_point = data.get_data_as_data_frame()
            print(pred_data_point)
            print("Before Prediction")

            predict_pipeline = PredictPipeline()
            print("Mid Prediction")

            results = predict_pipeline.predict(pred_data_point)
            print("After Prediction")
            print("predictions: ",results,"   shape:",results.shape)

            return render_template('home.html', result1=results[0, 0], result2=results[0, 1])

        else:
            error_message = "Some form fields are missing or empty: {}".format(request.form)
            print(error_message)
            return render_template('error.html', message="Form data is incomplete. Please fill out all fields.")

        



if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True, port=7070)

