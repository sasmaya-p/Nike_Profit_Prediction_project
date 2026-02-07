from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("profit_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # collect values in SAME ORDER as training
        features = [
            float(request.form["Units_Sold"]),
            float(request.form["MRP"]),
            float(request.form["Discount_Applied"]),
            float(request.form["Revenue"]),
            float(request.form["Recalculated_Revenue"]),
            float(request.form["Profit_Margin"]),
            float(request.form["Gender_Category_Kids"]),
            float(request.form["Gender_Category_Men"]),
            float(request.form["Gender_Category_Women"]),
            float(request.form["Product_Line_Basketball"]),
            float(request.form["Product_Line_Lifestyle"]),
            float(request.form["Product_Line_Running"]),
            float(request.form["Product_Line_Soccer"]),
            float(request.form["Product_Line_Training"]),
            float(request.form["Product_Name_Air_Force_1"]),
            float(request.form["Product_Name_Air_Jordan"]),
            float(request.form["Product_Name_Air_Zoom"]),
            float(request.form["Product_Name_Blazer_Mid"]),
            float(request.form["Product_Name_Dunk_Low"]),
            float(request.form["Product_Name_Flex_Trainer"]),
            float(request.form["Product_Name_Free_RN"]),
            float(request.form["Product_Name_Kyrie_Flytrap"]),
            float(request.form["Product_Name_LeBron_20"]),
            float(request.form["Product_Name_Mercurial_Superfly"]),
            float(request.form["Product_Name_Metcon_7"]),
            float(request.form["Product_Name_Pegasus_Turbo"]),
            float(request.form["Product_Name_Phantom_GT"]),
            float(request.form["Product_Name_Premier_III"]),
            float(request.form["Product_Name_React_Infinity"]),
            float(request.form["Product_Name_SuperRep_Go"]),
            float(request.form["Product_Name_Tiempo_Legend"]),
            float(request.form["Product_Name_Waffle_One"]),
            float(request.form["Product_Name_Zoom_Freak"]),
            float(request.form["Product_Name_ZoomX_Invincible"]),
            float(request.form["Sales_Channel_online"]),
            float(request.form["Sales_Channel_retail"]),
            float(request.form["Region_Bangalore"]),
            float(request.form["Region_Delhi"]),
            float(request.form["Region_Hyderabad"]),
            float(request.form["Region_Kolkata"]),
            float(request.form["Region_Mumbai"]),
            float(request.form["Region_Pune"])
        ]

        final_features = np.array(features).reshape(1, -1)
        final_features = scaler.transform(final_features)

        prediction = model.predict(final_features)[0]

        return render_template("index.html", prediction_text=f"Predicted Profit: {prediction:.2f}")

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)
