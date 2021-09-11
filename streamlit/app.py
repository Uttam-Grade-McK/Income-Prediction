# from flask import Flask, request
import pandas as pd
import numpy as np
import pickle
import sklearn
import streamlit as st
from PIL import Image
pickle_in = open('Income_Classifier.pkl', 'rb')
classifier = pickle.load(pickle_in)


def Predict_Income(age, fnlwgt, education_num, marital_status, relationship, race, sex, capital_gain, capital_loss, hours_per_week, country, employment_type):
    
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
    '''age = request.args.get('age')
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
    employment_type = request.args.get('employment_type')'''
    prediction = classifier.predict([[age, fnlwgt, education_num, 
                                      marital_status, relationship, 
                                      race, sex, capital_gain, capital_loss, 
                                      hours_per_week, country, 
                                      employment_type]])
    print("prediction:", prediction)
    return 'This is the Predicted Value:-->' , prediction


def main():
    st.title('Adult Income Prediction')
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Adult Income Prediction ML App</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    age = st.text_input('age', 'Type Here')
    fnlwgt = st.text_input('fnlwgt', 'Type Here')
    education_num = st.text_input('education-num', 'Type Here')
    marital_status = st.text_input('marital-status', 'Type Here')
    relationship = st.text_input('relationship', 'Type Here')
    race = st.text_input('race', 'Type Here')
    sex = st.text_input('sex', 'Type Here')
    capital_gain = st.text_input('capital-gain', 'Type Here')
    capital_loss = st.text_input('capital-loss', 'Type Here')
    hours_per_week = st.text_input('hours-per-week', 'Type Here')
    country = st.text_input('country', 'Type Here')
    employment_type = st.text_input('employment-type', 'Type Here')
    result = ""
    if st.button("Predict"):
        result = Predict_Income(age, fnlwgt, education_num, marital_status, relationship, race, sex, capital_gain, capital_loss, hours_per_week, country, employment_type)
    st.success("The Output is {}".format(result))
    if st.button('About'):
        st.text('Let,s Learn')
        st.text('Built with Streamlit!')
        
    
if __name__ == '__main__':
    main()