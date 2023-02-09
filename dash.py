import pandas as pd
import streamlit as st
import requests
import os

waitress_port = os.environ['PORT']

def make_pred(api_uri, client_id):
    response = requests.get(api_uri+'/predict', json={'client_id' : client_id}
                               )
    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))
    return response.json()
    
def main():
    api_uri = 'http://pad-app.herokuapp.com/api'
    st.title('Prêt à dépenser')
    
    client_selector = st.sidebar.number_input("Identifiant client",
                                              min_value = 100001,
                                              max_value = 456250,
                                              
                                             )
   
    test_btn = st.button('Test')
    if test_btn:
        st.header('Is good ? :')
        response = requests.get(api_uri+'/test', json={'client_id': 42})

        st.header(str(response.status_code))
        st.header(str(response.text))
    predict_btn = st.button('Prédire')
    if predict_btn:
        st.header('Identifiant : {}'.format(client_selector))
        response = make_pred(api_uri, client_selector)
        proba = response['probability']
        pred = response['prediction']
        if pred == 0:
            st.header(':green[ACCEPTE]')
        else: st.header(':red[REFUSE]')
        col1, col2 = st.columns(2)
        with col1:
            st.header('Probabilité de non-remboursement :')
        with col2:
            if pred == 0 :
                st.header(':green[{}]'.format(proba)+':green[%]')
            else: st.header(':red[{}]'.format((proba))+':red[%]')

        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if response['data'] is None:
                st.markdown('data pb')
            else: st.markdown(str(len(response['data']['CNT_CHILDREN'])))
        with col2:
            st.markdown('VALUE OF TOP6 INDICATOR VS MEAN VALUE OF NEGATIVE INDIVIDUALS')
        with col3:
            st.markdown('DISPLAY OF FEATURE SELECTION')
        with col4:
            st.markdown('FEATURE SELECTION')

    
    
if __name__ == '__main__':
    main()