#!/usr/bin/env python
# coding: utf-8

# # **Applied Machine Learning Homework 5**
# # **UNI**-rk3165
# # **Name**- Rakshith Kamath
# **Due 2 May,2022 (Monday) 11:59PM EST**

# ### Natural Language Processing
# We will train a supervised training model to predict if a tweet has a positive or negative sentiment.

# In[1]:


import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import classification_report


# ####  **Dataset loading & dev/test splits**

# **1.1) Load the twitter dataset from NLTK library**

# In[2]:


import nltk
nltk.download('twitter_samples')
from nltk.corpus import twitter_samples 


# In[3]:


import nltk
nltk.download("stopwords")
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


# **1.2) Load the positive & negative tweets**

# In[4]:


all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')


# **1.3) Create a development & test split (80/20 ratio):**

# In[5]:


#code here
pos_label = ['pos']*len(all_positive_tweets)
neg_label = ['neg']*len(all_negative_tweets)

tweets=all_positive_tweets+all_negative_tweets
labels=pos_label+neg_label

df=pd.DataFrame({'tweets':tweets,'sentiment':labels})
df.head(10)


# In[6]:


y = df.sentiment
df.drop(['sentiment'], axis=1, inplace=True)
df.head()


# In[7]:


X_dev,X_test, y_dev, y_test = train_test_split(df, y, test_size=.2, random_state=42)
print(f"The amount of positive and negative sentiment tweets in dev")
print(y_dev.value_counts())
print(f"The amount of positive and negative sentiment tweets in test")
print(y_test.value_counts())


# #### **Data preprocessing**
# We will do some data preprocessing before we tokenize the data. We will remove `#` symbol, hyperlinks, stop words & punctuations from the data. You can use the `re` package in python to find and replace these strings. 

# **1.4) Replace the `#` symbol with '' in every tweet**

# In[8]:


#code here
X_dev[['tweets']]=X_dev.apply({'tweets':lambda x:re.sub(r'#','',x)})
X_dev.head(10)


# In[9]:


#code here
X_test[['tweets']]=X_test.apply({'tweets':lambda x:re.sub(r'#','',x)})
X_test.head(10)


# **1.5) Replace hyperlinks with '' in every tweet**

# In[10]:


#code here
X_dev[['tweets']]=X_dev.apply({'tweets':lambda x:re.sub(r'@\w*','',x)})
X_dev[['tweets']]=X_dev.apply({'tweets':lambda x:re.sub(r'http\S+','',x)})
X_dev.head(10)


# In[11]:


#code here
X_test[['tweets']]=X_test.apply({'tweets':lambda x:re.sub(r'@\w*','',x)})
X_test[['tweets']]=X_test.apply({'tweets':lambda x:re.sub(r'http\S+','',x)})
X_test.head(10)


# **1.6) Remove all stop words**

# In[12]:


#code here
stop_words = stopwords.words('english')

def remove_stop_words(sent):
    token_words = word_tokenize(sent)
    stopwords_removed = [word for word in token_words if word not in stop_words]
    return ' '.join(stopwords_removed)


# In[13]:


X_dev[['tweets']]=X_dev.apply({'tweets':lambda x:remove_stop_words(x)})
X_dev.head(10)


# In[14]:


X_test[['tweets']]=X_test.apply({'tweets':lambda x:remove_stop_words(x)})
X_test.head(10)


# **1.7) Remove all punctuations**

# In[15]:


#code here
X_dev[['tweets']]=X_dev.apply({'tweets':lambda x:re.sub(r'[^\w\s]', '',x)})
X_dev[['tweets']]=X_dev.apply({'tweets':lambda x:re.sub(r'_', '',x)})
X_dev.head(10)


# In[16]:


#code here
X_test[['tweets']]=X_test.apply({'tweets':lambda x:re.sub(r'[^\w\s]', '',x)})
X_test[['tweets']]=X_test.apply({'tweets':lambda x:re.sub(r'_', '',x)})
X_test.head(10)


# **1.8) Apply stemming on the development & test datasets using Porter algorithm**

# In[17]:


#code here
porter = PorterStemmer()
def stem(sent):
    token_words = word_tokenize(sent)
    stem_sent = [porter.stem(word) for word in token_words]
    return ' '.join(stem_sent)


# In[18]:


X_dev[['tweets']]=X_dev.apply({'tweets':lambda x:stem(x)})
X_dev.head(10)


# In[19]:


X_test[['tweets']]=X_test.apply({'tweets':lambda x:stem(x)})
X_test.head(10)


# #### **Model training**

# **1.9) Create bag of words features for each tweet in the development dataset**

# In[20]:


#code here
bag_of_words = CountVectorizer()
X_dev_bag=bag_of_words.fit_transform(X_dev.tweets)


# **1.10) Train a supervised learning model of choice on the development dataset**

# In[21]:


#code here
lr_bag = LogisticRegressionCV(cv=10, max_iter=10000)
lr_bag.fit(X_dev_bag, y_dev)


# **1.11) Create TF-IDF features for each tweet in the development dataset**

# In[22]:


#code here
tf_idf = TfidfVectorizer()
X_dev_tf_idf=tf_idf.fit_transform(X_dev.tweets)


# **1.12) Train the same supervised learning algorithm on the development dataset with TF-IDF features**

# In[23]:


#code here
lr_tf_idf = LogisticRegressionCV(cv=10, max_iter=10000)
lr_tf_idf.fit(X_dev_tf_idf, y_dev)


# **1.13) Compare the performance of the two models on the test dataset**

# In[24]:


#code here
X_test_bag = bag_of_words.transform(X_test.tweets)
print(f"Performance of bag of words on test dataset-{lr_bag.score(X_test_bag, y_test)}")
print(classification_report(y_test,lr_bag.predict(X_test_bag)))


# In[25]:


X_test_tf_idf = tf_idf.transform(X_test.tweets)
print(f"Performance of tf idf on test dataset-{lr_tf_idf.score(X_test_tf_idf, y_test)}")
print(classification_report(y_test,lr_tf_idf.predict(X_test_tf_idf)))


# **Answer-**
# Bag of Words model constructs a vocabulary extracting the unique words from the documents and keeps the vector with the term frequency of the particular word in the corresponding document.In TF-IDF, apart from the term frequencies we also take inverse of number of documents that a particular term appears or the inverse of document frequency.
# 
# Hence,in this study, Term ordering is not considered and Rareness of a term is not considered in BOW hence TF-IDF is better approach.
