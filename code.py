#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''twitter bot which can automatically find the trending hashtag for perticaular location
   and download csv file for that hashtag tweets and perform sentiment analysis on it'''
#21MCA085 jimit patel python case study
#required modules import
import tweepy
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import string
nltk.download('stopwords')
nltk.download('vader_lexicon')


# In[2]:


#twitter account credentials for apply operations
access_token="1513574046671446017-bR7skNpLv7CMddNQrIn2YQbhgCsWHg"
access_token_secret="hjPFUyqzRas0YwMHTroRvNdlLodCygGfgKZdxfUR69bt8"
api_key="rdIJnfOCt2e8yzuzDDr9Q4ih6"
api_key_secret="9PciOOKeBlCe5ya2eXBi1kSX2vcfz6RZcRjD9nESW9ik8BOoVM"

#create auth variable for perform authentication of twitter account
auth=tweepy.OAuthHandler(consumer_key=api_key,consumer_secret=api_key_secret)
auth.set_access_token(access_token,access_token_secret)
api=tweepy.API(auth)
print(api)


# In[3]:


#where on earth id this is for india
woeid = 23424848
 
# fetching the trends
trends = api.get_place_trends(id = woeid)
 
# printing the information
print("The top trends for the location are :")

try:
    
    for value in trends:
         for trend in value['trends']:
             print('-----------------------------------------------')
             print(trend['name'])
             print('No. of Tweets Using It.',trend['tweet_volume'])
             
        
except:
    print("somthing going wrong...")



#fetching the top 100 tweets of any hashtag
'''
choice=input("enter the  trending hashtag : ")

#top_tweets=tweepy.cursor(api.search_tweets,q=choice).items(100)

for tweet in api.search_tweets(q=choice,lang='en'):
#for tweet in top_tweets:
    print('--------------------------------------------')
    print(tweet.text)
    print('/n')
'''


# In[14]:


#now code write for the find tweets for perticular hashtag

# function to display data of each tweet

def printtweetdata(n, ith_tweet):
		print()
		print(f"Tweet {n}:")
		print(f"Username:{ith_tweet[0]}")
		print(f"Description:{ith_tweet[1]}")
		print(f"Location:{ith_tweet[2]}")
		print(f"Following Count:{ith_tweet[3]}")
		print(f"Follower Count:{ith_tweet[4]}")
		print(f"Total Tweets:{ith_tweet[5]}")
		print(f"Retweet Count:{ith_tweet[6]}")
		print(f"Tweet Text:{ith_tweet[7]}")
		print(f"Hashtags Used:{ith_tweet[8]}")


# function to perform data extraction
def scrape(words, date_since, numtweet):
    

		# Creating DataFrame using pandas
		db = pd.DataFrame(columns=['username',
								'description',
								'location',
								'following',
								'followers',
								'totaltweets',
								'retweetcount',
								'text',
								'hashtags'])

		# We are using .Cursor() to search
		# through twitter for the required tweets.
		# The number of tweets can be
		# restricted using .items(number of tweets)
		tweets = tweepy.Cursor(api.search_tweets,
							words, lang="en",
							since_id=date_since,
							tweet_mode='extended').items(numtweet)


		# .Cursor() returns an iterable object. Each item in
		# the iterator has various attributes
		# that you can access to
		# get information about each tweet
		list_tweets = [tweet for tweet in tweets]

		# Counter to maintain Tweet Count
		i = 1

		# we will iterate over each tweet in the
		# list for extracting information about each tweet
		for tweet in list_tweets:
				username = tweet.user.screen_name
				description = tweet.user.description
				location = tweet.user.location
				following = tweet.user.friends_count
				followers = tweet.user.followers_count
				totaltweets = tweet.user.statuses_count
				retweetcount = tweet.retweet_count
				hashtags = tweet.entities['hashtags']

				# Retweets can be distinguished by
				# a retweeted_status attribute,
				# in case it is an invalid reference,
				# except block will be executed
				try:
						text = tweet.retweeted_status.full_text
				except AttributeError:
						text = tweet.full_text
				hashtext = list()
				for j in range(0, len(hashtags)):
						hashtext.append(hashtags[j]['text'])

				# Here we are appending all the
				# extracted information in the DataFrame
				ith_tweet = [username, description,
							location, following,
							followers, totaltweets,
							retweetcount, text, hashtext]
				db.loc[len(db)] = ith_tweet

				# Function call to print tweet data on screen
				printtweetdata(i, ith_tweet)
				i = i+1
		filename = '#'+words+'_'+date_since+'.csv'

		# we will save our database as a CSV file.
		db.to_csv(filename)
        #print(filename)
        #return filename


# In[15]:

print('------------------------------------------------------------------------')
print("Enter Twitter HashTag to search for")
words = input()
print("Enter Date since The Tweets are required in yyyy-mm--dd")
date_since = input()
    # number of tweets you want to extract in one run for analysis
print("Enter number of tweets data you want required for sentimet analysis")
num = input()
numtweet=int(num)
#numtweet=100
print(numtweet)
#numtweet=trends['tweet_volume']
scrape(words, date_since, numtweet)
print('file downloaded ')


# In[19]:

print('--------------------------------------------------------------------------')
print("Enter .csv file name for perform TWEETS sentiment analysis :-")
filename = input()
data = pd.read_csv("C:/Users/jimit/Downloads/"+filename+".csv")
#print(data.head())


# In[20]:



stemmer = nltk.SnowballStemmer("english")


stopword=set(stopwords.words('english'))

def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text

data["tweet"] = data["text"].apply(clean)



sentiments = SentimentIntensityAnalyzer()
data["Positive"] = [sentiments.polarity_scores(i)["pos"] for i in data["tweet"]]
data["Negative"] = [sentiments.polarity_scores(i)["neg"] for i in data["tweet"]]
data["Neutral"] = [sentiments.polarity_scores(i)["neu"] for i in data["tweet"]]

data = data[["tweet", "Positive", 
             "Negative", "Neutral"]]
#print(data.head())

x = sum(data["Positive"])
y = sum(data["Negative"])
z = sum(data["Neutral"])

def sentiment_score(a, b, c):
    if (a>b) and (a>c):
        print("sentiment is Positive 😊 for this trend")
    elif (b>a) and (b>c):
        print("sentiment is going Negative 😠 for this trend")
    else:
        print("sentiment is Neutral 🙂  for this trend")
sentiment_score(x, y, z)

print("Positive tweets: ", x)
print("Negative tweets: ", y)
print("Neutral tweets: ", z)







