import streamlit as st
import pandas as pd
from clean_data import clean_text
from joblib import load
# from train_model import tfv
# model = load_model('insurance_pipeline')
model = load('rf_pipeline.pkl')
def predict(model, input_df):
    # text = clean_text(str(input_df.text))
    # text = tfv.transform(text)
    return model.predict([clean_text(str(input_df.text))])

def run(model):

    st.title("COVID-19 Vaccines Emotions Analysis")
    text = st.text_input('text')

    output="WRITE EMOTIONS TO ANALYSIS"

    input_df = pd.DataFrame([{'text' : text}])
    print("")
    if st.button("Predict"):
        output = predict(model=model, input_df=input_df)
        output = 'Positive' if int(output)>0 else "Negative"

    st.success(f'Your emotions about covid-19 vaccines are {output}')
