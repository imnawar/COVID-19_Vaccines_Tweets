import pandas as pd
import streamlit as st
from joblib import load 
import matplotlib.pyplot as plt
from clean_data import clean_text

model = load('Deployment/rf_pipeline.pkl')

def predict(model, input_df):
    return model.predict([clean_text(str(input_df.text))])


def load_data():
    data = pd.read_csv('Data/VaccinesData.csv')
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
fig = plt.figure()
plt.hist(data['blob_label'])
plt.title('The positive vs negative tweets')
plt.xlabel('label')
plt.xticks([0, 1],['Neg', 'Pos'])
plt.ylabel('number of tweets')
st.pyplot(fig)


st.subheader('Number of tweets per month')
data['date'] = pd.to_datetime(data['date'])
data['month'] = pd.DatetimeIndex(data['date']).month
fig = plt.figure(figsize=(25,20))
for i, group in data.groupby('month'):
    plt.bar(group['month'].iloc[0],group['clean_text'].count())
plt.ylabel('number of tweets')
plt.xlabel('date')
plt.xticks(range(12), ['January', 'February', 'March', 'April', 'May', 'June', 'July',
          'August', 'September', 'October', 'November', 'December'])
plt.title('Number of tweets/Day'); 
st.pyplot(fig)