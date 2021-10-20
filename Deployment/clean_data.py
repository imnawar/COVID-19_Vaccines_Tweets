import re
import pandas as pd 
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sent = SentimentIntensityAnalyzer()
lemmatizer = WordNetLemmatizer()
def clean_text(text):
    text = re.sub(r'@\w+', '', text) # remove mintions 
    text = re.sub(r'#', '', text) # remove hashtages 
    text = re.sub(r'RT[\s]+', '', text) # remove RT notaitions 
    text = re.sub(r'https?:\/\/\S+', '', text) # remove URLs
    text = text.encode('ascii', 'ignore').decode('ascii') # remove emoji
    text = text.lower() #convert the text to lower case 
    text = text.split() #splitting the text 
    text = [word for word in text if word not in stopwords.words('english')] # remove stop words 
    text = [lemmatizer.lemmatize(word) for word in text] 
    text = ' '.join(text)
    return text

def refer(tweet, refs):
  flag =0
  for ref in refs:
    if tweet.find(ref) != -1:
      flag =1
  return flag

def generate_label(x):
    return 1 if TextBlob(x).polarity>0 else 0


def clean_data():
    df = pd.read_csv("Data/vaccination_all_tweets.csv")
    print("="*80)
    print("Satuts: strat cleaning the data...")
    print("Status: clean tweets ....")
    df['clean_text'] = df['text'].apply(clean_text)
    print("Status: finsh the tweets clean...")
    print("Status: adding vaccines type feautre...")
    pfizer_refs = ["Pfizer","pfizer","Pfizer–BioNTech","pfizer-bioNtech","BioNTech","biontech"]
    bbiotech_refs = ["covax","covaxin","Covax","Covaxin","Bharat Biotech","bharat biotech","BharatBiotech","bharatbiotech"]
    sputnik_refs = ["russia","sputnik","Sputnik","V"]
    astra_refs = ['sii','SII','adar poonawalla','Covishield','covishield','astra','zenca','Oxford–AstraZeneca','astrazenca','oxford-astrazenca','serum institiute']
    moderna_refs = ['moderna','Moderna','mRNA-1273','Spikevax']

    df['pfizer'] = df['clean_text'].apply(lambda x : refer(x, pfizer_refs))
    df['bbiotech'] = df['clean_text'].apply(lambda x : refer(x, bbiotech_refs))
    df['sputnik'] = df['clean_text'].apply(lambda x : refer(x, sputnik_refs))
    df['astra'] = df['clean_text'].apply(lambda x : refer(x, astra_refs))
    df['moderna'] = df['clean_text'].apply(lambda x : refer(x, moderna_refs))
    print("Status: finsh adding vaccines type feautre...")
    print("Status: adding country name feature (extracted from user_location)...")
    df['country_name']=df['user_location'].str.split(',').str[-1]
    print("Status: change date feature type to datetime type...")
    df['date'] = pd.to_datetime(df['date']).dt.date
    print("Status: finsh adding the new features...")
    print("Status: labelling the text tweets using -SentimentIntensityAnalyzer...")
    df['label_sent'] = df['clean_text'].apply(lambda x: 1 if sent.polarity_scores(x)['compound']>0 else 0) 
    print("Status: labelling the text tweets using -TextBlob...")
    df['blob_label'] = df['clean_text'].apply(lambda x: generate_label(x))
    print("Status: drop unused features...")
    (df.drop(columns={'user_name', 'user_location', 'user_description', 'user_created',
                    'user_followers', 'user_friends', 'user_favourites', 'user_verified',
                    'text', 'source', 'retweets', 'favorites','is_retweet'}, inplace=True))
    print("Status: FINSH THE DATA CLEANING PHASE")
    print("="*80)
    return df
