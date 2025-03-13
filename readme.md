# Fish Market Machine Learning Project

## Overview
This project uses machine learning to analyze the Fish Market dataset from Kaggle. The implementation focuses on a classification model that identifies fish species based on their physical measurements.

## Dataset
The dataset contains information about common fish species sold in fish markets, with the following features:
- Species: Fish species (7 different species)
- Weight: Weight of the fish (in grams)
- Length1: Vertical length (in cm)
- Length2: Diagonal length (in cm)
- Length3: Cross length (in cm)
- Height: Height (in cm)
- Width: Width (in cm)

Data source: [Fish Market Dataset on Kaggle](https://www.kaggle.com/aungpyaeap/fish-market)

## Model
The project implements a classification approach:

### Classification Model
- **Model**: Random Forest Classifier
- **Target**: Fish species
- **Features**: Weight, Length1, Length2, Length3, Height, Width
- **Preprocessing**: Feature scaling with StandardScaler
- **Evaluation Metrics**: Accuracy, Classification Report
- **Hyperparameter Tuning**: GridSearchCV for optimizing n_estimators and max_depth

## Files in this Repository

- `Fish_Market_ML_Model.ipynb`: Jupyter notebook with data exploration, preprocessing, and model building
- `Fish.csv`: Original dataset
- `fish_market_model.pkl`: Saved model file
- `app.py`: Flask API for serving predictions
- `templates/`: HTML templates for the web interface
- `static/`: CSS and JavaScript files for styling and interactivity

## Interactive Web Application

The project includes a web-based user interface where users can:
1. Input fish measurements (length, height, width)
2. Get instant predictions of fish species from the trained model

## Installation and Setup

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Setup Instructions
1. Clone this repository:
   ```
   git clone https://github.com/yourusername/fish-market-ml.git
   cd fish-market-ml
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

3. Run the Flask application:
   ```
   python app.py
   ```

4. Open your web browser and navigate to:
   ```
   http://localhost:8080
   ```

## Usage

1. Enter the fish measurements in the input fields
2. Click the "Predict" button
3. View the predicted fish species

## Future Improvements

- Implement more advanced models for better prediction accuracy
- Add visualization of prediction confidence
- Include more detailed exploratory data analysis
- Add support for uploading fish images for measurement extraction

