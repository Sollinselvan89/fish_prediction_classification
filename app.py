from flask import Flask, request
import numpy as np
import joblib

app = Flask(__name__)

# Load model and preprocessing objects
model = joblib.load('fish_species_model.pkl')
scaler = joblib.load('scaler.pkl')
encoder = joblib.load('encoder.pkl')

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_result = ""
    input_values = {}
    
    if request.method == 'POST':
        try:
            # Get input values from form
            features = [
                float(request.form['weight']),
                float(request.form['length1']),
                float(request.form['length2']),
                float(request.form['length3']),
                float(request.form['height']),
                float(request.form['width'])
            ]
            
            input_values = {
                'weight': features[0],
                'length1': features[1],
                'length2': features[2],
                'length3': features[3],
                'height': features[4],
                'width': features[5]
            }
            
            # Convert to numpy array and reshape
            features_array = np.array(features).reshape(1, -1)
            
            # Scale features
            features_scaled = scaler.transform(features_array)
            
            # Make prediction
            prediction = model.predict(features_scaled)
            
            # Convert prediction index to species name
            species = encoder.inverse_transform(prediction)[0]
            
            prediction_result = f'<div class="prediction-result"><h2>Result</h2><p>Predicted Fish Species: {species}</p></div>'
            
        except Exception as e:
            prediction_result = f'<div class="prediction-result"><h2>Error</h2><p>{str(e)}</p></div>'
    
    # Return complete HTML page
    return f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Fish Species Predictor</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
                font-family: Arial, sans-serif;
            }}
            
            body {{
                background-color: #f0f8ff;
                color: #333;
                line-height: 1.6;
                padding: 20px;
            }}
            
            .container {{
                max-width: 800px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }}
            
            header {{
                text-align: center;
                margin-bottom: 30px;
                padding: 20px 0;
                background-color: #1e88e5;
                color: white;
                border-radius: 8px;
            }}
            
            header h1 {{
                font-size: 2.5rem;
                margin-bottom: 10px;
            }}
            
            .form-group {{
                margin-bottom: 20px;
            }}
            
            label {{
                display: block;
                margin-bottom: 8px;
                font-weight: bold;
            }}
            
            input[type="number"] {{
                width: 100%;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 4px;
                font-size: 16px;
            }}
            
            .submit-btn {{
                background-color: #1e88e5;
                color: white;
                border: none;
                padding: 12px 20px;
                font-size: 16px;
                border-radius: 4px;
                cursor: pointer;
                width: 100%;
            }}
            
            .submit-btn:hover {{
                background-color: #1565c0;
            }}
            
            .prediction-result {{
                background-color: #e3f2fd;
                padding: 20px;
                border-radius: 8px;
                margin: 20px 0;
                border-left: 5px solid #1e88e5;
            }}
            
            .info-card {{
                margin-top: 30px;
                padding: 20px;
                background-color: #f5f5f5;
                border-radius: 8px;
            }}
            
            .info-card h3 {{
                color: #1565c0;
                margin-bottom: 15px;
            }}
            
            .info-card ul {{
                padding-left: 20px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>Fish Species Predictor</h1>
                <p>Enter fish measurements to predict the species</p>
            </header>
            
            <form action="/" method="post">
                <div class="form-group">
                    <label for="weight">Weight (g):</label>
                    <input type="number" id="weight" name="weight" step="0.01" value="{input_values.get('weight', '')}" required>
                </div>
                
                <div class="form-group">
                    <label for="length1">Length1 (cm):</label>
                    <input type="number" id="length1" name="length1" step="0.01" value="{input_values.get('length1', '')}" required>
                </div>
                
                <div class="form-group">
                    <label for="length2">Length2 (cm):</label>
                    <input type="number" id="length2" name="length2" step="0.01" value="{input_values.get('length2', '')}" required>
                </div>
                
                <div class="form-group">
                    <label for="length3">Length3 (cm):</label>
                    <input type="number" id="length3" name="length3" step="0.01" value="{input_values.get('length3', '')}" required>
                </div>
                
                <div class="form-group">
                    <label for="height">Height (cm):</label>
                    <input type="number" id="height" name="height" step="0.01" value="{input_values.get('height', '')}" required>
                </div>
                
                <div class="form-group">
                    <label for="width">Width (cm):</label>
                    <input type="number" id="width" name="width" step="0.01" value="{input_values.get('width', '')}" required>
                </div>
                
                <div class="form-group">
                    <input type="submit" value="Predict Species" class="submit-btn">
                </div>
            </form>
            
            {prediction_result}
            
            <div class="info-card">
                <h3>About this model</h3>
                <p>This machine learning model predicts fish species based on various measurements. The model was trained on the Fish Market Dataset using a Random Forest classifier.</p>
                <h4>Fish species in the dataset:</h4>
                <ul>
                    <li>Bream</li>
                    <li>Roach</li>
                    <li>Whitefish</li>
                    <li>Parkki</li>
                    <li>Perch</li>
                    <li>Pike</li>
                    <li>Smelt</li>
                </ul>
            </div>
        </div>
    </body>
    </html>
    '''

if __name__ == '__main__':
    app.run(debug=True, port=8080)