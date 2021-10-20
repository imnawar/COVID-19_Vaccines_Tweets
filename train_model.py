import pickle 
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer




def train(df): 
    print("="*80)
    print("Status: split the data into train/test...")
    X_train, X_test, y_train_sent, y_test_sent = train_test_split(df.clean_text, df.label_sent, test_size=0.25, random_state=123)
    y_train_blob, y_test_blob = train_test_split(df.blob_label, test_size=0.25, random_state=123)
    print("Status: vectorize the text...")
    tfv = TfidfVectorizer()
    X_train = tfv.fit_transform(X_train.values.astype('U'))
    X_test = tfv.transform(X_test.values.astype('U'))
    print("Status: init the RF model, SENT labels...")
    rf_sent = RandomForestClassifier(random_state = 123)
    rf_sent.fit(X_train, y_train_sent)
    print("Status: testing the RF model SENT labels...")
    y_pred = rf_sent.predict(X_test)
    print("Status: RF ACCURACY ON SENT labels:", accuracy_score(y_test_sent,y_pred))
    print("Status: init the RF model, TextBlob labels...")
    rf_blob = RandomForestClassifier(random_state = 123)
    rf_blob.fit(X_train, y_train_blob)
    print("Status: testing the RF model TextBlob labels...")
    y_pred = rf_blob.predict(X_test)
    print("Status: RF ACCURACY ON TextBlob labels:", accuracy_score(y_test_blob,y_pred))
    print("Status: Saving the trained models at Models/...")
    with open("Models/rf_blob.pkl", "wb") as f:
        pickle.dump(rf_blob, f)
    with open("Models/rf_sent.pkl", "wb") as f:
        pickle.dump(rf_sent, f)
    print("Status: Models saved...")
    print("="*80)
    if accuracy_score(y_test_blob,y_pred)>accuracy_score(y_test_sent,y_pred): return rf_blob, tfv
    else: return rf_sent, tfv
