import snscrape.modules.twitter as sntwitter
import pandas as pd
import numpy as np
import re
#sentiment anlysis tools (https://www.unite.ai/10-best-python-libraries-for-sentiment-analysis/)
from nltk.sentiment import SentimentIntensityAnalyzer as nltk_sia
    #pip install nltk
from textblob import TextBlob as sia_textblob
    #pip install textblob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as vader_sia
    #pip install vaderSentiment
    #better at negative sentiment analaysis
    #https://www.analyticsvidhya.com/blog/2021/10/sentiment-analysis-with-textblob-and-vader/
from pattern.en import sentiment as sia_Pattern
    #pip install pattern
    #https://digiasset.org/html/pattern-en.html#sentiment
    #https://www.analyticsvidhya.com/blog/2021/11/pattern-library-for-natural-language-processing-in-python/

from nltk.corpus import stopwords

sia_nltk = nltk_sia()
#sia_vader = vader_sia()
stopwords = set(stopwords.words('english'))

def create_full_data_frame(query, limit):
    df = scrape(query, limit)
    df = clean_tweet_feature(df)
    df = omit_short_tweets(df)
    df = create_col_of_NLTK_Sentiment_Analysis(df)
    df = create_col_of_TextBlob_Sentiment_Analysis(df)
    df = create_col_of_compound_sentiment_scores(df)
    df = create_col_of_labels_for_tweet_sentiment(df)
    df = drop_neutral_tweets(df)
    return df

#https://www.youtube.com/watch?v=jtIMnmbnOFo
def scrape(query, limit):
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
    temp = tweet.lower() #lowercase 
    temp = re.sub("'", "", temp) # to avoid removing contractions in english
    temp = re.sub("@[A-Za-z0-9_]+","", temp)#remove mention of other users
    temp = re.sub("#[A-Za-z0-9_]+","", temp)#remove hashtags
    temp = re.sub(r'http\S+', '', temp)#remove linkes
    temp = re.sub('[()!?]', ' ', temp)#remove unessery punctuation ancharacters
    temp = re.sub('\[.*?\]',' ', temp)
    temp = re.sub("[^a-z0-9]"," ", temp)
    temp = temp.split()
    temp = [w for w in temp if not w in stopwords]#rmove stop words
    temp = " ".join(word for word in temp)
    return temp

def clean_tweet_feature(dataframe):
    if 'Tweet' in dataframe.columns:
        dataframe['clean_Tweet'] = dataframe['Tweet'].apply(clean_tweet)
        return dataframe
    else:
        return 'need column of tweets named "Tweet"'
    
def drop_neutral_tweets(dataframe):
    assert("Compound_Sentiment_Score" in dataframe.columns)
    idxs = dataframe.loc[(dataframe['Compound_Sentiment_Score'] == 0) | (dataframe['Compound_Sentiment_Score'].isnull())].index
    dataframe = dataframe.drop(idxs)
    return dataframe
#--------------omit rows with short tweets------------------------------------
def omit_short_tweets(dataframe):
    assert('clean_Tweet' in dataframe.columns)
    idx_to_drop = []
    for index, row in dataframe.iterrows():
        t = row['clean_Tweet']
        t = t.split()
        if len(t) <= 3:
            idx_to_drop.append(index)
    dataframe = dataframe.drop(idx_to_drop)
    return dataframe
#--------------------------------------------------------------------------
#----------------sentiment analysis columns------------------------------- 
def NLTK_sentiment_analysis_col(tweet):
    tweet = str(tweet)
    return sia_nltk.polarity_scores(tweet)["compound"]

def create_col_of_NLTK_Sentiment_Analysis(dataframe):
    if 'clean_Tweet' in dataframe.columns:
        dataframe['NLTK_Sentiment_Analysis'] = dataframe['clean_Tweet'].apply(NLTK_sentiment_analysis_col)
        return dataframe
    else:
        return 'need column of cleaned tweets named "clean_Tweets"'
    
def TextBlob_sentiment_analysis_col(tweet):
    tweet = str(tweet)
    return sia_textblob(tweet).sentiment.polarity

def create_col_of_TextBlob_Sentiment_Analysis(dataframe):
    if 'clean_Tweet' in dataframe.columns:
        dataframe['TextBlob_Sentiment_Analysis'] = dataframe['clean_Tweet'].apply(TextBlob_sentiment_analysis_col)
        return dataframe
    else:
        return 'need column of cleaned tweets named "clean_Tweets"'
'''
#same results as nltk
def Vader_sentiment_analysis_col(tweet):
    tweet = str(tweet)
    return sia_vader.polarity_scores(tweet)['compound']

def create_col_of_Vader_Sentiment_Analysis(dataframe):
    if 'clean_Tweet' in dataframe.columns:
        dataframe['Vader_Sentiment_Analysis'] = dataframe['clean_Tweet'].apply(Vader_sentiment_analysis_col)
        return dataframe
    else:
        return 'need column of cleaned tweets named "clean_Tweets"'
#same results as textblob
def pattern_sentiment_analysis_col(tweet):
    tweet = str(tweet)
    return sia_Pattern(tweet)[0]

def create_col_of_Pattern_Sentiment_Analysis(dataframe):
    if 'clean_Tweet' in dataframe.columns:
        dataframe['Pattern_Sentiment_Analysis'] = dataframe['clean_Tweet'].apply(pattern_sentiment_analysis_col)
        return dataframe
    else:
        return 'need column of cleaned tweets named "clean_Tweets"'
'''
#----------------------------------------------------------------------------------------------------------
#-------------------------create ensemble compound score------------------------------------------------------
def create_col_of_compound_sentiment_scores(dataframe):
    dataframe['Compound_Sentiment_Score'] = dataframe['TextBlob_Sentiment_Analysis']+dataframe['NLTK_Sentiment_Analysis']#ser
    return dataframe
#----------------------------------------------------------------------------------------------------------
#--------------create stentment label based off ensemble compund score-------------------------------------
def is_positive(compound_score) -> bool:
    return compound_score > 0

def create_col_of_labels_for_tweet_sentiment(dataframe):
    if 'Compound_Sentiment_Score' in dataframe.columns:
        dataframe['Positive_Sentiment'] = dataframe['Compound_Sentiment_Score'].apply(is_positive)
        return dataframe
    else:
        return 'need column named "Sentiment_Analysis" with compound : score map'
#----------------------------------------------------------------------------------------------------------
#def compound_col(dic):
#    return dic['compound']
#
#def create_col_of_compound_tweet_sentiment_score(dataframe):
#    if 'Sentiment_Analysis' in dataframe.columns:
#       dataframe['Compound_Sentiment Score'] = dataframe['Sentiment_Analysis'].apply(compound_col)
#        return dataframe
#    else:
#        return 'need column named "Sentiment_Analysis" with compound : score map'

def make_csv_from_df(df, file_name):
    df.to_csv(file_name)