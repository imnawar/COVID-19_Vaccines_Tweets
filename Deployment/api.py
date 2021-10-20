import numpy as np
import pandas as pd
import streamlit as st
from joblib import load 
import matplotlib.pyplot as plt
from clean_data import clean_text

model = load('rf_pipeline.pkl')

def predict(model, input_df):
    return model.predict([clean_text(str(input_df.text))])


def load_data():
    data = pd.read_csv('../Data/VaccinesData.csv')
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    return data

st.title('Covid-19 Vaccines Analysis')



# to use the pre-trained model in predecting 
st.header("Use the pre-trained model to analyze your emotions about COVID-19 Vaccines ...!")
text = st.text_input('text')

output="WRITE EMOTIONS TO ANALYSIS"

input_df = pd.DataFrame([{'text' : text}])
print("")
if st.button("Predict"):
    output = predict(model=model, input_df=input_df)
    output = 'Positive' if int(output)>0 else "Negative"

st.success(f'Your emotions about covid-19 vaccines are {output}')

# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
data = load_data()
# Notify the reader that the data was successfully loaded.
data_load_state.text('Loading data...done!')


st.subheader('Number of positive/negative tweets')
hist_values = np.histogram(data['blob_label'],  bins=2)[0]
st.bar_chart(hist_values)


fig = plt.figure(figsize=(45,40))
for i, group in data.groupby('date'):
    plt.bar(group['date'].iloc[0].month,group['clean_text'].count())
plt.ylabel('number of tweets')
plt.xlabel('date')
plt.xticks(rotation=90)
plt.title('Number of tweets/Day'); 

st.pyplot(fig)