#!/usr/bin/env python
# coding: utf-8

# In[41]:


import nltk
nltk.download('punkt')
  


# In[1]:


import pandas as pd


# In[2]:


#reading the dataset
tweets = pd.read_csv("tweets.csv", encoding = "ISO-8859-1")
tweets.head()


# In[3]:


#Since we are only interested in the actual content of the tweet,
#the only relevant column for us is the text column. let us have a
#closer look at it
tweets.text


# In[19]:


#Now we need to preprocess the data to get rid of the text noise
#we do that by creating a function
import re
import string
punctuation = string.punctuation

from nltk.corpus import stopwords
stopwords = stopwords.words("english")

def cleanTweet(tweet):
    tweet = tweet.lower()
    #maybe we want to remove the twitter handle as well
    #removing the punctuations
    tweet = "".join(x for x in tweet if x not in punctuation)
    #removing the stop words
    individualWords = tweet.split()
    individualWords = [w for w in individualWords if w not in stopwords]
    tweet = " ".join(individualWords)
    #removing certain common occuring words
    tweet = re.sub('rt', '', tweet)
    return tweet
cleanTweet("This is A sample,,,, text!")


# In[20]:


tweets["clean tweets"] = tweets["text"].apply(cleanTweet)


# In[21]:


tweets[["text", "clean tweets"]]


# KEYWORD ANALYSIS

# In[35]:


#now that the preprocessing has been done, it is time to analyse the data

#somethings which can look for the following things
#certain keywords, phrases, people mentioned, hashtags,urls, emails

#Keywords: let us have a look at the most commmon used words
from collections import Counter
complete_set = " ".join(tweets["clean tweets"])
words = complete_set.split()
Counter(words).most_common(50)


# In[26]:


#if you want to get the top 50 mentioned twitter handles
rawText = " ".join(tweets["text"])
twitterHandles = [w for w in rawText.split() if w.startswith('@')]
Counter(twitterHandles).most_common(50)


# In[28]:


#if you want to extract the top HashTags
rawText = " ".join(tweets["text"])
hashTags = [w for w in rawText.split() if w.startswith('#')]
#we can possibly get rid of the common Hashtags, since this data set
#is related to demonetization, we can get rid of words starting
# with demo
hashTags = [w for w in hashTags if "demo" not in w.lower()]
Counter(hashTags).most_common(50)


# In[31]:


#if you want to extract the top links
rawText = " ".join(tweets["text"])
links = [w for w in rawText.split() if w.startswith('http')]
links = [w for w in links if "demo" not in w.lower()]
Counter(links).most_common(50)


# In[37]:


#let us now look at ngrams to get an understanding of phrases being used
from nltk import ngrams
bigrams = ngrams(complete_set.split(), 2)
Counter(bigrams).most_common(20)


# In[48]:


#let us look at the top named entities now
import nltk
from nltk import word_tokenize, pos_tag, ne_chunk

#iterating one by one through the raw data as cleaned text has data missing
for text in tweets["text"]:
    entities = ne_chunk(pos_tag(word_tokenize(text)))
    for entity in entities:
        if hasattr(entity, "label"):
            print(entity)
    


# In[51]:


# let us see if we can analyze the sentiment of the tweet.
# we can use the textblob library

#!pip install textblob
from textblob import TextBlob

TextBlob("EVeryone hated demonetization").sentiment


# In[52]:


TextBlob("EVeryone loved demonetization").sentiment


# In[55]:


sentiment = []
for i in range(0, len(tweets["text"])):
    sentiment.append(TextBlob(tweets["text"][i]).sentiment)
tweets["Sentiment"] = sentiment
tweets


# In[ ]:




