# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import nltk
import csv
import pandas as pd
import matplotlib



dfTrain = pd.read_csv('C:/Users/Pedram/Documents/GitManemuneee/KagTox001/train.csv')

dfTrain['toxic', ] = dfTrain['toxic'].astype('int')
dfTrain['severe_toxic', ] = dfTrain['severe_toxic'].astype('int')
dfTrain['obscene', ] = dfTrain['obscene'].astype('int')
dfTrain['threat', ] = dfTrain['threat'].astype('int')
dfTrain['insult', ] = dfTrain['insult'].astype('int')
dfTrain['identity_hate', ] = dfTrain['identity_hate'].astype('int') 

selCols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
dfTrain['overallToxicity'] = dfTrain[selCols].sum(axis = 1)

dfTrain['overallToxicity'].plot(kind='hist')
dfTrain[dfTrain['overallToxicity'] != 0]['overallToxicity'].plot(kind='hist') 






