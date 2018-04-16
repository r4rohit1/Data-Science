# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 23:17:30 2018

@author: rsudhakar
"""

import pandas as pd
import matplotlib.pyplot as plt
import json
tweet3=[]
with open('twitterdata_IT1.txt', 'r') as txt_file:
    for text in txt_file:
        tweet=json.loads(line)
        tweet3.append(tweet)
    

tweets=pd.DataFrame  

tweets['tweet']=map(lambda tweets:tweets['tweet'],tweet3)    
    
  

