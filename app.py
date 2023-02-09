from http.client import responses
from sqlite3 import DatabaseError
from flask import Flask, request, jsonify
import pickle
import pandas as pd
import xgboost as xgb
import shap
import psycopg2
import os


app = Flask(__name__)

threshold = 0.365
#need to load the model, the explainer and the data to the api
model = pickle.load(open('pipeline.pkl', 'rb'))
DATABASE_URL = os.environ['DATABASE_URL']
conn = psycopg2.connect(DATABASE_URL, sslmode='require')
#explainer = pickle.load(open('explainer.pkl', 'rb'))

@app.route('/')
def index():
    routes = ['/predict', '/explain']
    return '---Pret a depenser API.---'

@app.route('/test')
def test():
    test = {'test':'OK!'}
    return jsonify(test)

def get_data(ID):
    client_data = data[data.index==int(ID)]
    if client_data.shape[0] != 0:
        return client_data

@app.route('/predict', methods=['GET'])
def predict():
    #retrive the id from the request and pull it's data
    client_id = request.get_json()
    client_id = client_id['client_id']
    client_data = get_data(client_id)

    if client_data is None:
        response = 'Client inconnu'
    else:
        #make prediction and probability
        proba = model.predict_proba(client_data)
        if proba[0,1] > threshold:
            pred = 1
        else: pred = 0
        response = {'id' : client_id,
                'data' : client_data.to_dict(),
                'prediction' : pred,
                'probability' : int(round(proba[0,1],2)*100)}
    return jsonify(response)

#def explain():
    #retrive the id from the request and pull it's data
    #client_id = request.get_json()
   # client_id = client_id['client_id']
    #client_data = get_data(client_id)
   # if client_data is None:
       # response = 'Client inconnu'
    
