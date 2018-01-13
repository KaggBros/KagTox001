# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import nltk
import csv
import pandas as pd
import matplotlib
import numpy as np
from numpy import where

import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
import string
from nltk.classify import NaiveBayesClassifier
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
# nltk.download()
# string.punctuation           punctuations
# stopwords.words('english')   stop words


dfTrain = pd.read_csv('C:/Users/Pedram/Documents/GitManemuneee/KagTox001/train.csv')

#dfTrain['toxic', ] = dfTrain['toxic'].astype('int')
#dfTrain['severe_toxic', ] = dfTrain['severe_toxic'].astype('int')
#dfTrain['obscene', ] = dfTrain['obscene'].astype('int')
#dfTrain['threat', ] = dfTrain['threat'].astype('int')
#dfTrain['insult', ] = dfTrain['insult'].astype('int')
#dfTrain['identity_hate', ] = dfTrain['identity_hate'].astype('int') 

selCols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
dfTrain['overallToxicity'] = dfTrain[selCols].sum(axis = 1)

dfTrain['overallToxicity'].plot(kind='hist')
dfTrain[dfTrain['overallToxicity'] != 0]['overallToxicity'].plot(kind='hist') 
##############################################
##############################################
######## add a new column of tokenized words #
def stop_word_remover(sentence):
    word_list=word_tokenize(sentence)
    word_list=[word.lower() for word in word_list]
    return [word for word in word_list if word not in stopwords.words('english') and word not in string.punctuation]
    
# dfTrain['tokenized']=dfTrain.apply(lambda row: word_tokenize(row['comment_text']), axis=1)

import time
start=time.time()
dfTrain['tokenized']=dfTrain.apply(lambda row: stop_word_remover(row['comment_text']), axis=1)
t=time.time() - start
print(t)
##############################################
##############################################
################### wordCloud ################
allText = "".join(dfTrain['comment_text'])
wordcloud = WordCloud().generate(allText)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
##############################################
##############################################
############ toxic Text wordcloud ############
dfTrain_toxic=dfTrain[dfTrain['toxic']==1]
toxic_text = "".join(dfTrain_toxic['comment_text'])
wordcloud = WordCloud(collocations =False).generate(toxic_text)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

dfTrain_toxic=dfTrain[(dfTrain['toxic']==1) & (dfTrain['overallToxicity']==1)]
toxic_text = "".join(dfTrain_toxic['comment_text'])
wordcloud = WordCloud(collocations =False).generate(toxic_text)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
##############################################
##############################################
############ toxic Text wordcloud ############
dfTrain_severe_toxic=dfTrain[dfTrain['severe_toxic']==1]
severe_toxic_text = "".join(dfTrain_severe_toxic['comment_text'])
wordcloud = WordCloud(collocations =False).generate(severe_toxic_text)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

dfTrain_severe_toxic=dfTrain[(dfTrain['severe_toxic']==1) & (dfTrain['overallToxicity']==2)]
severe_toxic_text = "".join(dfTrain_severe_toxic['comment_text'])
wordcloud = WordCloud(collocations =False).generate(severe_toxic_text)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

## all the severe_toxic messages are at least toxic. 
dfTrain_severe_toxic=dfTrain[(dfTrain['severe_toxic']==1) & (dfTrain['overallToxicity']==3)]
severe_toxic_text = "".join(dfTrain_severe_toxic['comment_text'])
wordcloud = WordCloud(collocations =False).generate(severe_toxic_text)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
##############################################
##############################################
############ toxic Text wordcloud ############
dfTrain_obscene=dfTrain[dfTrain['obscene']==1]
obscene_text = "".join(dfTrain_obscene['comment_text'])
wordcloud = WordCloud(collocations =False).generate(obscene_text)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

dfTrain_obscene=dfTrain[(dfTrain['obscene']==1) & (dfTrain['overallToxicity']==1)]
obscene_text = "".join(dfTrain_obscene['comment_text'])
wordcloud = WordCloud(collocations =False).generate(obscene_text)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
##############################################
##############################################
############ toxic Text wordcloud ############
dfTrain_threat=dfTrain[dfTrain['threat']==1]
threat_text = "".join(dfTrain_threat['comment_text'])
wordcloud = WordCloud(collocations =False).generate(threat_text)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


## purely threatning dataset is developed with natural words. they are really hard to predict,
## there are barely curse words in them. and some of them are not even threatning
dfTrain_threat=dfTrain[(dfTrain['threat']==1) & (dfTrain['overallToxicity']==1)]
threat_text = "".join(dfTrain_threat['comment_text'])
wordcloud = WordCloud(collocations =False).generate(threat_text)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
##############################################
##############################################
############ toxic Text wordcloud ############
dfTrain_insult=dfTrain[dfTrain['insult']==1]
insult_text = "".join(dfTrain_insult['comment_text'])
wordcloud = WordCloud(collocations =False).generate(insult_text)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

dfTrain_insult=dfTrain[(dfTrain['insult']==1) & (dfTrain['overallToxicity']==1)]
insult_text = "".join(dfTrain_insult['comment_text'])
wordcloud = WordCloud(collocations =False).generate(insult_text)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
##############################################
##############################################
############ toxic Text wordcloud ############
dfTrain_hate=dfTrain[dfTrain['identity_hate']==1]
hate_text = "".join(dfTrain_hate['comment_text'])
wordcloud = WordCloud(collocations =False).generate(hate_text)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

dfTrain_hate=dfTrain[(dfTrain['identity_hate']==1) & (dfTrain['overallToxicity']==1)]
hate_text = "".join(dfTrain_hate['comment_text'])
wordcloud = WordCloud(collocations =False).generate(hate_text)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
##############################################
##############################################
## this is find the comments for any number of offenses and common words in them.
dfTrain_multiple=dfTrain[dfTrain['overallToxicity']==5]
multiple_text = "".join(dfTrain_multiple['comment_text'])
wordcloud = WordCloud(collocations =False).generate(multiple_text)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
##############################################
##############################################

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

df = pd.read_csv('C:/Users/Pedram/Documents/GitManemuneee/KagTox001/train.csv')

df['class']=where(dfTrain.identity_hate==1, "identity_hate", 
       where(dfTrain.threat==1, "threat",
             where(dfTrain.insult==1, "insult",
                   where(dfTrain.obscene==1, "obscene",
                         where(dfTrain.severe_toxic==1, "severe_toxic",
                               where(dfTrain.toxic==1, "toxic", 0))))))

y = df['class']
x_train, x_test, y_train, y_test = train_test_split(df['comment_text'],
                                                    y, test_size=0.33,
                                                    random_state=53)


count_vectorizer = CountVectorizer(stop_words='english')
count_train = count_vectorizer.fit_transform(x_train.values)
count_test = count_vectorizer.transform(x_test.values)

nb_classifier = MultinomialNB()
nb_classifier.fit(count_train, y_train)
pred = nb_classifier.predict(count_test)
metrics.accuracy_score(y_test, pred)
metrics.confusion_matrix(y_test, pred)
### We have the model and we just have to apply it on the test set




dfTest = pd.read_csv('C:/Users/Pedram/Documents/GitManemuneee/KagTox001/test.csv')
dfTest = dfTest.apply(lambda row: punctuation_remover)
dfTest = dfTest['comment_text']
pd.DataFrame.dropna(dfTest, axis = 1, how = 'all')


count_test = count_vectorizer.transform(dfTest.values)
predTestSet = nb_classifier.predict(dfTest)
##############################################
##############################################
dfTrain.loc[1, 'tokenized']
dfTrain1=dfTrain.loc[0:1,]
dfTrain1.loc['tokenized_clean']=dfTrain1.apply(lambda row: stop_word_remover(row['tokenized']), axis=1)

sampleSent=dfTrain.loc[0, 'comment_text']
tokenized=nltk.word_tokenize(sampleSent)


word_list=dfTrain.loc[0, 'tokenized']

print(stopwords)


