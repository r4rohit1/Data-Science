# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 22:50:12 2018

@author: rsudhakar
"""

#TWITTER NEW SENTIMENT


import json 
import pandas as pd
import matplotlib.pyplot as plt
import tweepy as tp
import re
from textblob import TextBlob
#Authentication 
consumer_key= '3DNWh0PnMbDN083TWpTnl02Rs'
consumer_secret= 'NjobiyRO5kDJfeZMBg0pc1LjhhgDAvoEwMQwOJW41JTBgzw84U'

access_token='1476917616-MChc02uO3pNHOehfhzu8E6bMDvFUp5HfkL9YQeq'
access_token_secret='ZOklhmIsjjGW0nik3YHUbCJd1iIkbVgUR5jwq55fmDGBd'


class Listerner(tp.StreamListener):
    def on_data(self, data):
            with open('rawtweets.csv','a') as txtfile:
                txtfile.write(data)
            print(data)
    
    def on_error(self,status):
        print(status)

help(tp.StreamListener)
auth = tp.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
St=tp.Stream(auth, Listerner
St.filter(track=['India'])
twee=[]
twee_lang=[]
twee_country=[]
tweet={}      
with open('rawtweets.txt','r') as raw_tweets:
    for line in raw_tweets:
        try:    
        
            tweets=json.loads(line)
            tweet.update(tweets)
            twee.append(tweet['text'])
            twee_country.append(tweet['place']['country'])
            twee_lang.append(tweet['lang'])
        except:
            continue
   
cleantweets=pd.DataFrame()
places=pd.DataFrame()
lang=pd.DataFrame()


    
   

cleantweets['tweet']=twee
lang['lang']=twee_lang
places['country']=twee_country
places['country']=places['country'].replace(to_replace='भारत', value='India',inplace =True)
tweets_by_lang=lang['lang'].value_counts()
tweets_by_country=places['country'].value_counts()


from sklearn.feature_extraction.text import CountVectorizer
cs=CountVectorizer( stop_words='english')
X=cs.fit_transform(twee)


fig, ax = plt.subplots()
ax.tick_params(axis='x',labelsize=15)
ax.tick_params(axis='y',labelsize=15)
ax.set_xlabel('Languages',fontsize=15)
ax.set_ylabel('Number of tweets',fontsize=15)
ax.set_title('Languages', fontsize=15 ,fontweight='bold')
tweets_by_lang[:10].plot(ax=ax, kind='bar', color='blue')
plt.show()

fig, ax = plt.subplots()
ax.tick_params(axis='x',labelsize=15)
ax.tick_params(axis='y',labelsize=15)
ax.set_xlabel('Places',fontsize=15)
ax.set_ylabel('Number of tweets',fontsize=15)
ax.set_title('Places', fontsize=15 ,fontweight='bold')
tweets_by_country[:10].plot(ax=ax, kind='bar', color='blue')
plt.show()

clean_tweets=[]
for i in range(0,2004):
      clean_tweets_var=re.sub('\W+',' ',cleantweets['tweet'][i])
      clean_tweets_var.lower()
      clean_tweets.append(clean_tweets_var)
    
text=pd.DataFrame()
text['text']=str(clean_tweets)
for i in range(0,2004):
    bloba=TextBlob(text['text'][i])