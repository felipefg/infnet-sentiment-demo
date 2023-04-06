import streamlit as st
import requests

st.set_page_config(page_title='Análise de Sentimento', layout='wide')

def perform_inference(text, url = 'http://localhost:5001/invocations'):

    data = {
        "dataframe_records": [
            {'text': text},
        ]
    }

    response = requests.post(url, json=data)
    results = response.json()
    pred = results['predictions'][0]
    return pred


intro = """
# Exemplo de Análise de Sentimento

Este exemplo usará um modelo treinado como dataset [Twitter US Airline Sentiment
](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment).
"""

st.write(intro)

entrada = st.text_input("Texto para Avaliar")

if entrada:
    pred = perform_inference(entrada)
    st.write(f"A predição é {pred}")
