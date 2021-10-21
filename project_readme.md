# COVID-19_Vaccines_Tweets
This project is one of the T5 Data Science BootCamp requirements.

## Abstract
This project aimed to understand people's emotions about the covid-19 vaccines by analyzing the tweets using machine learning models to help the safety of society, and know the most covid-19 vaccine that is comfortable for the people. The used data in this project is provided by Kaggle, the data is labeled using Sentiment Intensity Analyzer, with sklearn library random forest was trained and get 96.86% accuracy. The streamlit is used to build an interactive dashboard to visualize and communicate the final results. 

<!-- The data has been explored, cleaned, and new features such as labeling the tweets 0 for negative tweets and 1 for positive tweets have been added, as well as on-hot-encoding for the vaccine type has been added.  -->

## Data 

The dataset is provided in ```.csv``` format. It contains 19,3272 tweets, each tweet has 15 features. The most relevant feature to this project is the text which contains the tweet text. Some other features are extracted from other features such as country name is extracted from location, where it contains the user location at the time of the Tweet. Another important feature of this project is the label of the tweet where it extracted from tweet text using Sentiment Analysis tools. The pre-processed data are provided in ```csv``` file. 

# Usage 

To run this project you can run ```python main.py``` from your command line after installing the requirements listed at req.txt. This command will run the project with the developed API to show the results and use the pre-trained model in predicting sentiment. In addition this command will use the pre-processed data and the pre-trained model. 

run ```python main.py -api``` to do the experiment without the developed API using the pre-processed data and the pre-trained model. 

run ```python main.py -c``` to do all data cleaning phase rather than use the pre-processed data. 

run ```python main.py -t``` this command will train a model rather than use the pre-trained model. 

If you are interested in doing all experiments with all phases you can run ```python main.py -c -t``` this command will show you the pre-processing steps then training model steps. 

***Explore the steps***

If you're interested in showing the steps and exploring the data, then the provided [notebook]() shows all details. 

## Tools

- Pandas for data manipulation
- Scikit-learn for modeling
- re for clean data
- nltk for natural language processing
- Matplotlib for plotting
- streamlit for interactive visualizations

