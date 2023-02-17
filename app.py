from http.client import responses
from sqlite3 import DatabaseError
from webbrowser import BackgroundBrowser
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
BRONZE_DATABASE_URL = os.environ['HEROKU_POSTGRESQL_BRONZE_URL']
conn = psycopg2.connect(DATABASE_URL, sslmode='require')
conn2 = psycopg2.connect(BRONZE_DATABASE_URL, sslmode='require')
explainer = pickle.load(open('explainer.pkl', 'rb'))

#calculate shap values for the summary plot
def get_rand(n = 10000):
    cur = conn.cursor()
    cur.execute(f'SELECT * FROM client ORDER BY RANDOM() LIMIT {n}')
    rand = cur.fetchall()
    col_names = [desc[0] for desc in cur.description]
    rand = pd.DataFrame(rand, columns=col_names).set_index('SK_ID_CURR')
    rand = rand.drop(labels='index', axis=1)
    return rand

rand = get_rand()
rand_sv = explainer.shap_values(rand)


@app.route('/')
def index():
    routes = ['/predict', '/explain']
    return '---Pret a depenser API.---'

@app.route('/test')
def test():
    response = request.get_json()
    test = response['test']
    if test != '42':
        #check the primary db
        cur = conn.cursor()
        cur.execute('SELECT "AMT_CREDIT" FROM client WHERE "SK_ID_CURR"=100001')
        client_db = cur.fetchall()
        #check the raw data db
        cur2 = conn2.cursor()
        cur2.execute('SELECT "CODE_GENDER" FROM full_data WHERE "SK_ID_CURR"=100001')
        raw_db = cur2.fetchall()

        return_test = {'Status' : 'OK', 'client_db': client_db, 'raw_data' : raw_db}
    else: return_test = {f'Status' : 'Error with request, got : {test}'}
    return jsonify(return_test)




def get_data(ID):
    cur = conn.cursor()
    #get data from postgresql db
    cur.execute('SELECT * FROM client WHERE "SK_ID_CURR" = '+str(ID))
    client_data = cur.fetchall()
    #get colnames
    col_names = [desc[0] for desc in cur.description]
    if len(client_data) != 0:
        return client_data, col_names

@app.route('/predict', methods=['GET'])
def predict():
    #retrive the id from the request and pull it's data
    client_id = request.get_json()
    client_id = client_id['client_id']
    client_data, col_names = get_data(client_id)

    if client_data is None:
        response = 'Client inconnu'
    else:
        #make prediction and probability
        client_data = pd.DataFrame(client_data, columns=col_names).set_index('SK_ID_CURR')
        client_data = client_data.drop(labels='index', axis=1)
        proba = model.predict_proba(client_data)
        shap_values = explainer.shap_values(client_data)
        if proba[0,1] > threshold:
            pred = 1
        else: pred = 0
        response = {'id' : client_id,
                'data' : client_data.to_dict(),
                'prediction' : pred,
                'probability' : int(round(proba[0,1],2)*100),
                'shap_values' : shap_values.tolist(),
                'expected_val' : explainer.expected_value,
                'rand_sv' : rand_sv.tolist(),
                }
    return jsonify(response)

@app.route('/feat')
def feat():
    response = request.get_json()
    client_id = response['id']
    feat_name = response['feat']
    cur = conn2.cursor()
    cur.execute(f'SELECT "{feat_name}" FROM full_data WHERE "SK_ID_CURR" = {client_id}')
    raw_data = cur.fetchall()
    col_names = [desc[0] for desc in cur.description]
    raw_data = pd.DataFrame(raw_data, columns=col_names)
    response = {'raw_data' : raw_data.to_dict()}
    return jsonify(response)
    
