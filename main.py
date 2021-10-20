from operator import mod
import os
import argparse
import pandas as pd
from joblib import load
from train_model import train
from Deployment.app import run
from Deployment.clean_data import clean_data, clean_text

parser = argparse.ArgumentParser(description='Covid-19 Vaccines Tweets Analysis')
parser.add_argument('-d', '--cleanedDataset', action='store_false', help='Use the clean dataset phase')
parser.add_argument('-t', '--train', action='store_false', help='Use a training model phase')
parser.add_argument('-api', '--api', action='store_false', help='use api')
args = parser.parse_args()

def read_dataset(cleaning_flag):
    if cleaning_flag:
        print("USE THE CLEANED DATA")
        return pd.read_csv("Data/VaccinesData.csv")
    else:
        print("START CLEAN DATA PHASE")
        return clean_data()

def prepare_model(training_flag, df):
    if training_flag:
        print("USE A PRETRAINED RF")
        return load('rf_pipeline.pkl')
    else: 
        print("START MODELS TARINING PHASE")
        return train(df)


def main():
    df = read_dataset(args.cleanedDataset)
    model = prepare_model(args.train, df)
    if args.api:
        os.system('streamlit run api.py')
if __name__ == '__main__':main()
