# MVP of COVID-19_Vaccines_Tweets

In this folder, the Minimum Viable Product (MVP) is provided. 

The initial steps in detail are provided in the Jupyter notebook. 
The data explored in addition to the EDA phase is done. 

## The MVP notebook summarizes the following: 
- Explore the features 
- Drop any unnecessary features
- Clean the text of the tweets
- Create one-hot-encoding for the vaccine type
- Label the data using ```SentimentIntensityAnalyzer``` from ```nltk```
- Apply the ```TfidfVectorizer```
- Train and test a naive bayes model to classify the data and got 88.75%


The following figure illustrates the ratio of positive to negative tweets:

![pos_neg_sent](https://user-images.githubusercontent.com/36853625/137148801-a3d41640-e188-47a9-8974-f36fc52add5c.png)


## There are a few more steps to finish the project: 
- Use different models to classify the text. 
- Try TextBlob in data labelling. 
