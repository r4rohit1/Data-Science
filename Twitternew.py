# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 17:01:46 2018

@author: rsudhakar
"""
import tweepy as tp
import json


cosumer_key='FCZO2Yhl8PtxSXJajKkyZqgcZ'
comsumer_secret='h1UcupJQA7HCyVSbSpL4Zz7iKeCbysud3VNqnWAzNnZmEpDfb5'
access_token_secret='ZOklhmIsjjGW0nik3YHUbCJd1iIkbVgUR5jwq55fmDGBd'
access_token='1476917616-MChc02uO3pNHOehfhzu8E6bMDvFUp5HfkL9YQeq'

class Listener(tp.StreamListener):
    def on_data(self,data):
                tweet=data.split('","text":')[1].split(',"source')[0]
                print(tweet.text)
       
    def on_error(self,status):
        print(status)


   
auth=tp.OAuthHandler(cosumer_key,comsumer_secret)      
auth.set_access_token(access_token,access_token_secret)

stlisten=tp.streaming.Stream(auth,Listener())
stlisten.filter(track=['Arsenal', 'Bayern Munich','Chelsea'])


