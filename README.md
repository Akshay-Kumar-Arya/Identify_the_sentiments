Identify_the_sentiments
=======================

Contest at Analytics Vidhya. [Link](https://datahack.analyticsvidhya.com/contest/linguipedia-codefest-natural-language-processing-1/#About) for the contest.

## About the Problem:
Sentiment analysis is contextual mining of text which identifies and extracts subjective information in source material, and helping a business to understand the 
social sentiment of their brand, product or service while monitoring online conversations. Brands can use this data to measure the success of their products in 
an objective manner. In this challenge,tweet data is provided to predict sentiment on electronic products of netizens.

## Problem Statement
Sentiment analysis remains one of the key problems that has seen extensive application of natural language processing. 
This time around, given the tweets from customers about various tech firms who manufacture and sell mobiles, computers, laptops, etc, 
the task is to identify if the tweets have a negative sentiment towards such companies or products.


## Data Science Resources
* Get started with NLP and text classification with [Natural Language Processing (NLP) using Python](https://trainings.analyticsvidhya.com/courses/course-v1:AnalyticsVidhya+NLP101+2018_T1/about?utm_source=practice_problem_Identify_The_Sentiments&utm_medium=Datahack) 
 course
* Refer this [comprehensive guide](https://www.analyticsvidhya.com/blog/2018/11/tutorial-text-classification-ulmfit-fastai-library/) 
  that exhaustively covers text classification techniques using different libraries and its implementation in python.
* You can also refer this [guide](https://www.analyticsvidhya.com/blog/2018/07/hands-on-sentiment-analysis-dataset-python/) 
  that covers multiple techniques including TF-IDF, Word2Vec etc. to tackle problems related to Sentiment Analysis.
  
## Dataset
[`train.csv`](https://datahack.analyticsvidhya.com/contest/linguipedia-codefest-natural-language-processing-1/download/train-file) -  The train data file 
contains 7920 tweets. The dataset is provided with each line storing a tweet id, its label and the tweet.

[`test.csv`](https://datahack.analyticsvidhya.com/contest/linguipedia-codefest-natural-language-processing-1/download/test-file) - The test data file 
contains 1953 tweets. The test data file contains only tweet ids and the tweet text with each tweet in a new line.

Most profane and vulgar terms in the tweets have been replaced with “$&@*#”. 
However, please note that the dataset still might contain text that may be considered profane, vulgar, or offensive.


## Implementation Approach

### Text Cleaning and Preprocessing
Apply the below text preposessing on the training and testing tweets sets:

* URLs removal: We have used Regular Expressions (or RegEx) to remove the URLs.
* Remove the Twitter user handle with @ mentions.
* Punctuation marks removal: remove any punction marks from the text.
* Numbers removal: replace any digits in the tweets with space.
* Whitespaces removal
* Convert the text to lowercase.
* Convert text into embedding using pretrained Model from Tensorflow Hub.

### Tweets to BERT vectors
We import and use the pretrained google BERT model, where we extract BERT vectors for the cleaned tweets in the train and test datasets. Each tweet is represented by an BERT vector of length 768 in terms of the tweet's words/tokens.

### Tweets to nnlm vectors
We import and used the pretrained google nnlm model, where we extract nnlm vectors for the cleaned tweets in the train and test datasets. Each tweet is represented by an nnlm vector of length 128 in terms of the tweet's words/tokens.

### Classification Model building and evaluation
Use preprocessed dataset to for training classification models. Use `f1 score` metric for evaluation as it is the official evaluation metric in contest. Models trained and their evaluation score is provided:
* Logistic regression model using BERT vectors, evaluation score is `0.8715878799554232`
* Logistic regression model using nnlm vectors, evaluation score is `0.85817439707144`
* Logistic regression model using nnlm and BERT vectors, evaluation score is `0.877712596054773`
* MLP model using BERT vectors, evaluation score is `0.8792921986900165`
* GRU sequence model using BERT vectors, evaluation score is `0.8989105276853336`
