from flask import Flask, request
import pandas as pd
import numpy as np
import flasgger
from flasgger import Swagger # Swagger will automatically generate front-end UI Part
import pickle
app = Flask(__name__)
Swagger(app)
pickle_in = open('Income_Classifier.pkl', 'rb')
classifier = pickle.load(pickle_in)

@app.route("/") # It is just root page
def welcome():
    return 'Hello My Name is Uttam'

@app.route('/predict', methods=['GET'])
def Predict_Income():
    """Let's Predict the Income 
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: age
        in: query
        type: number
        required: true
      - name: fnlwgt
        in: query
        type: number
        required: true
      - name: education-num
        in: query
        type: number
        required: true
      - name: marital-status
        in: query
        type: number
        required: true
      - name: relationship
        in: query
        type: number
        required: true
      - name: race
        in: query
        type: number
        required: true
      - name: sex
        in: query
        type: number
        required: true
      - name: capital-gain
        in: query
        type: number
        required: true
      - name: capital-loss
        in: query
        type: number
        required: true
      - name: hours-per-week
        in: query
        type: number
        required: true
      - name: country
        in: query
        type: number
        required: true
      - name: employment_type
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
        
    """
    age = request.args.get('age')
    fnlwgt = request.args.get('fnlwgt')
    education_num = request.args.get('education-num')
    marital_status = request.args.get('marital-status')
    relationship = request.args.get('relationship')
    race = request.args.get('race')
    sex = request.args.get('sex')
    capital_gain = request.args.get('capital-gain')
    capital_loss = request.args.get('capital-loss')
    hours_per_week = request.args.get('hours-per-week')
    country = request.args.get('country')
    employment_type = request.args.get('employment_type')
    prediction = classifier.predict([[age, fnlwgt, education_num, 
                                      marital_status, relationship, 
                                      race, sex, capital_gain, capital_loss, 
                                      hours_per_week, country, 
                                      employment_type]])
    return 'This is the Predicted Value:-->' + str(prediction)

@app.route('/predict_file', methods = ['POST'])
def Predict_Income_File():
    """Let's Predict the Income 
    This is using docstrings for specifications.
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
      
    responses:
        200:
            description: The output values
        
    """
    
    df_test = pd.read_csv(request.files.get('file'))
    prediction = classifier.predict(df_test)
    return 'This is the Predicted for csv is Value:-->' + str(list(prediction))
    
    
    

if __name__ == "__main__":
    app.run()