# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 12:26:14 2018

@author: rsudhakar
"""

import tweepy
from textblob import TextBlob
import matplotlib.pyplot as plt
import time

# Step 1 - Authenticate
consumer_key= 'FCZO2Yhl8PtxSXJajKkyZqgcZ'
consumer_secret= 'h1UcupJQA7HCyVSbSpL4Zz7iKeCbysud3VNqnWAzNnZmEpDfb5'

access_token='1476917616-MChc02uO3pNHOehfhzu8E6bMDvFUp5HfkL9YQeq'
access_token_secret='ZOklhmIsjjGW0nik3YHUbCJd1iIkbVgUR5jwq55fmDGBd'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

#Step 3 - Retrieve Tweets
public_tweets = api.search('Aptean', count=10000,result_type="recent", lang='en')







def calctime(a):
    return time.time()-a

initime=time.time()
t=int(calctime(initime))
R=0
T=0
S=0
#CHALLENGE - Instead of printing out each tweet, save each Tweet to a CSV file
#and label each one as either 'positive' or 'negative', depending on the sentiment 
#You can decide the sentiment polarity threshold yourself


for tweet in public_tweets:
    print(tweet.text)
    
    #Step 4 Perform Sentiment Analysis on Tweets
    analysis = TextBlob(tweet.text)
    print(analysis.sentiment)
    print(analysis.sentiment.polarity)
    print("")
    if analysis.sentiment.polarity>=0:
        R=R+analysis.sentiment.polarity
    else:
        T=T+analysis.sentiment.polarity    
    
plt.axis([ -20, 70, -20,20])
plt.xlabel('Time')
plt.ylabel('Sentiment' )
plt.plot([t],[R],'go',[t] ,[T],'ro', markersize='50')

plt.pause(0.0001)
plt.show()      