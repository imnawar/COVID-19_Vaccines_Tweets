# Project Proposal 

This repo is one of the T5 Bootcamp requirements. 


# How do people feel about the covid-19 vaccine?

In a short period of time and under pandemic conditions, many vaccines have been suggested. This vaccine is necessary for the safety of society, and people should take the most covid-19 vaccine that is comfortable for them. 

The study aims to find out which covid-19 vaccination is most successful, by analyzing the tweets of people.

## Dataset
To achieve the goal of this study the dataset **COVID-19 All Vaccines Tweets** will be used. 
This dataset can be found [here](https://www.kaggle.com/gpreda/all-covid19-vaccines-tweets) at Kaggle.


This dataset contains tweets for the follwing vaccines:

- Pfizer/BioNTech
- Sinopharm
- Sinovac
- Moderna
- Oxford/AstraZeneca
- Covaxin
- Sputnik V

The dataset is available as the ```.csv``` file. Each row contains the following features: 

<table width="100%">
 <tr>
  <th>id</th><th>user_name</th><th>user_location</th><th>user_description</th><th>user_created</th><th>user_followers</th><th>user_friends</th><th>user_favourites</th><th>user_verified</th><th>date</th><th>text</th><th>hashtags</th><th>source</th><th>retweets</th><th>favorites</th><th>is_retweet</th>
 </tr>
 <tr>
  <th>id</th><th>user_name</th><th>user_location</th><th>user_description</th><th>user_created</th><th>user_followers</th><th>user_friends</th><th>user_favourites</th><th>user_verified</th><th>date</th><th>text</th><th>hashtags</th><th>source</th><th>retweets</th><th>favorites</th><th>is_retweet</th>
 </tr>
</table>
<!-- ```
       id, user_name, user_location, user_description, user_created,
       user_followers, user_friends, user_favourites, user_verified,
       date, text, hashtags, source, retweets, favorites,
       is_retweet
``` -->

Features such as **text**, which contained the tweet content, **date**, which included the tweet date, and **user_location** are used to identify the most important features for this study. 

Due to the success of Neural Networks (NN) such as RNN and LSTM in Natural Language Processing (NLP), I will fit a model on the tweets to discover which covid-19 vaccine has positive feedbacks from people. 


## Tools

There are tools that will be used to achieve the goal of this study, such as: ```TensorFlow, matplotlib, pandas, nltk``` for discovering the data and train a model. The work will be done through Jupyter notebook.

Furthermore, the Sentiment Analysis from ```nltk``` will be used to determine the target of the data in order to train the model in supervised manner. 

## **TO DO**: 
- Explore the data and come up with EDA phases then use a model to fit the data.  
- **NOTE:** the used features may be increased or changed and the model as well. 