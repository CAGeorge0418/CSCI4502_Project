import snscrape.modules.twitter as sntwitter
import pandas as pd
import numpy as np
import re
from nltk.sentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()
stopwords = ["for", "on", "an", "a", "of", "and", "in", "the", "to", "from"]

#https://www.youtube.com/watch?v=jtIMnmbnOFo
def scrap(query, limit):
    df = None
    tweets = []
    for tweet in sntwitter.TwitterSearchScraper(query).get_items():
        if len(tweets) == limit:
            break
        tweets.append([tweet.id, tweet.date, tweet.user.username, tweet.content, tweet.hashtags, tweet.likeCount, tweet.retweetCount, tweet.mentionedUsers, tweet.coordinates])
    df = pd.DataFrame(tweets, columns=['ID', 'Date','User','Tweet','HashTags', 'Likes','Retweets','MentionUsers', 'LocationCoordinates'])
    return df

#https://catriscode.com/2021/05/01/tweets-cleaning-with-python/
#regex code for clean tweet comes from the above source
def clean_tweet(tweet):
    if type(tweet) == np.float:
        return ""
    temp = tweet.lower()
    temp = re.sub("'", "", temp) # to avoid removing contractions in english
    temp = re.sub("@[A-Za-z0-9_]+","", temp)
    temp = re.sub("#[A-Za-z0-9_]+","", temp)
    temp = re.sub(r'http\S+', '', temp)
    temp = re.sub('[()!?]', ' ', temp)
    temp = re.sub('\[.*?\]',' ', temp)
    temp = re.sub("[^a-z0-9]"," ", temp)
    temp = temp.split()
    temp = [w for w in temp if not w in stopwords]
    temp = " ".join(word for word in temp)
    return temp

def clean_tweet_feature(dataframe):
    if 'Tweet' in dataframe.columns:
        dataframe['clean_Tweet'] = dataframe['Tweet'].apply(clean_tweet)
        return dataframe
    else:
        print('here1')
        return 'need column of tweets named "Tweet"'
  
def sentiment_analysis_col(tweet):
    tweet = str(tweet)
    return sia.polarity_scores(tweet)

def create_col_of_Sentiment_Analysis(dataframe):
    if 'clean_Tweet' in dataframe.columns:
        dataframe['Sentiment_Analysis'] = dataframe['clean_Tweet'].apply(sentiment_analysis_col)
        return dataframe
    else:
        return 'need column of cleaned tweets named "clean_Tweets"'

def is_positive(analysis) -> bool:
    return analysis["compound"] > 0

def create_col_of_labels_for_tweet_sentiment(dataframe):
    if 'Sentiment_Analysis' in dataframe.columns:
        dataframe['Positive_Sentiment'] = dataframe['Sentiment_Analysis'].apply(is_positive)
        return dataframe
    else:
        return 'need column named "Sentiment_Analysis" with compound : score map'

def compound_col(dic):
    return dic['compound']

def create_col_of_compound_tweet_sentiment_score(dataframe):
    if 'Sentiment_Analysis' in dataframe.columns:
        dataframe['Compound_Sentiment Score'] = dataframe['Sentiment_Analysis'].apply(compound_col)
        return dataframe
    else:
        return 'need column named "Sentiment_Analysis" with compound : score map'

def make_csv_from_df(df, file_name):
    df.to_csv(file_name)