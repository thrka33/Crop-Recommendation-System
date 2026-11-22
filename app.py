from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load trained machine learning models
mlp_model = joblib.load("mlp_crop_model.pkl")
knn_model = joblib.load("knn_crop_model.pkl")
label_encoder = joblib.load("crop_label_encoder.pkl")

# Input feature names
features = [
    "Nitrogen",
    "Phosphorus",
    "Potassium",
    "Temperature",
    "Humidity",
    "pH",
    "Rainfall"
]


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    selected_algorithm = None
    error = None
    input_values = {}

    if request.method == "POST":

        try:
            # Get values entered by the user
            nitrogen = float(request.form["nitrogen"])
            phosphorus = float(request.form["phosphorus"])
            potassium = float(request.form["potassium"])
            temperature = float(request.form["temperature"])
            humidity = float(request.form["humidity"])
            ph = float(request.form["ph"])
            rainfall = float(request.form["rainfall"])

            selected_algorithm = request.form["algorithm"]

            # Store user input
            input_values = {
                "nitrogen": nitrogen,
                "phosphorus": phosphorus,
                "potassium": potassium,
                "temperature": temperature,
                "humidity": humidity,
                "ph": ph,
                "rainfall": rainfall
            }

            # Arrange input in the correct feature order
            user_data = np.array([[
                nitrogen,
                phosphorus,
                potassium,
                temperature,
                humidity,
                ph,
                rainfall
            ]])

            # Select the machine learning model
            if selected_algorithm == "MLP":
                model = mlp_model

            elif selected_algorithm == "KNN":
                model = knn_model

            else:
                raise ValueError("Invalid algorithm selected.")

            # Predict crop
            predicted_label = model.predict(user_data)

            # Convert encoded label into crop name
            prediction = label_encoder.inverse_transform(
                predicted_label.astype(int)
            )[0]

        except ValueError as e:
            error = str(e)

        except Exception as e:
            error = "Prediction error: " + str(e)

    return render_template(
        "index.html",
        prediction=prediction,
        selected_algorithm=selected_algorithm,
        input_values=input_values,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)