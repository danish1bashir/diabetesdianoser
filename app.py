 
from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load and train model
data = pd.read_csv("DiabComplete.csv")

# Strip column names of whitespace
data.columns = data.columns.str.strip()

# Prepare features and target
X = data.drop("Diabetic", axis=1)
y = data["Diabetic"]

# Train model
model = GaussianNB()
model.fit(X, y)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.json
    values = [
        input_data['age'],
        input_data['sex'],
        input_data['family'],
        input_data['smoking'],
        input_data['drinking'],
        input_data['thirst'],
        input_data['urination'], 
        input_data['height'],
        input_data['weight'],
        input_data['fatuge']
    ]
    prediction = model.predict([values])[0]
    return jsonify({"result": "unfortunately your report is Positive" if prediction == 1 else "congratulations your report is Negative"})

if __name__ == '__main__':
    app.run(debug=True)
