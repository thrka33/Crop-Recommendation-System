from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

rf_model = joblib.load("rf_model.pkl")
le = joblib.load("label_encoder.pkl")

from xgboost import XGBClassifier

xgb_model = XGBClassifier()
xgb_model.load_model("xgb_model.json")


@app.route("/", methods=["GET", "POST"])
def home():

    rf_result = None
    xgb_result = None
    final_result = None

    if request.method == "POST":

        # Get user inputs
        N = float(request.form["N"])
        P = float(request.form["P"])
        K = float(request.form["K"])

        temperature = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        ph = float(request.form["ph"])
        rainfall = float(request.form["rainfall"])

        # Create input dataframe
        user_input = pd.DataFrame([{
            "N": N,
            "P": P,
            "K": K,
            "temperature": temperature,
            "humidity": humidity,
            "ph": ph,
            "rainfall": rainfall
        }])

        # Random Forest prediction
        rf_prediction = rf_model.predict(user_input)

        rf_result = le.inverse_transform(
            rf_prediction
        )[0]

        # XGBoost prediction
        xgb_prediction = xgb_model.predict(user_input)

        xgb_result = le.inverse_transform(
            xgb_prediction
        )[0]

        # Final recommendation
        if rf_result == xgb_result:
            final_result = rf_result

    return render_template(
        "index.html",
        rf_result=rf_result,
        xgb_result=xgb_result,
        final_result=final_result
    )


if __name__ == "__main__":
    app.run(debug=True)