import pandas as pd
import streamlit as st
import requests

def make_pred(api_uri, client_id, database):
    headers = {"Content-Type": "application/json"}
    data = {'dataframe_records' : [database.loc[client_id].fillna(0).to_dict()]}
    response = requests.request(method='POST',
                                headers=headers,
                                url=api_uri,
                                json=data
                               )
    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))
    return response.json()
    
def main():
    model_uri = 'http://127.0.0.1:5000/invocations'
    database = pd.read_csv('test_df_prepro.csv').set_index('SK_ID_CURR')
    proba = 0.3
    st.title('PAUVROMETRE')
    
    client_selector = st.sidebar.number_input("Identifiant client",
                                              min_value = 100001,
                                              max_value = 456250,
                                              
                                             )
    
    predict_btn = st.button('Prédire')
    if predict_btn:
        st.header('Identifiant : {}'.format(client_selector))
        proba = make_pred(model_uri, client_selector, database)['predictions'][0][1]
        col1, col2 = st.columns(2)
        with col1:
            st.header('Probabilité de remboursement :')
        with col2:
            if 1-proba > 0.5:
                st.header(':green[{}]'.format(1-proba)+':green[%]')
            else: st.header(':red[{}]'.format(1-proba)+':red[%]')
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('TOP 6 SHAP INDICATOR (3 top, 3 down)')
        with col2:
            st.markdown('VALUE OF TOP6 INDICATOR VS MEAN VALUE OF NEGATIVE INDIVIDUALS')
        with col3:
            st.markdown('DISPLAY OF FEATURE SELECTION')
        with col4:
            st.markdown('FEATURE SELECTION')

    
    
if __name__ == '__main__':
    main()