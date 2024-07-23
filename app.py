from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the model
model = joblib.load('house_price_model.pkl')

# Load the feature names
with open('feature_names.txt') as f:
    feature_names = [line.strip() for line in f]

@app.route('/')
def home():
    return render_template('index.html', feature_names=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        data = request.form.to_dict()

        # Convert form data to dataframe
        df = pd.DataFrame([data])

        # Convert columns to appropriate data types
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Handle missing values or invalid data
        df.fillna(0, inplace=True)  # Replace NaNs with 0

        # Make prediction
        prediction = model.predict(df)
        
        # Return result
        return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
