import re
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
import pickle
from bs4 import BeautifulSoup
import numpy as np                                                                
import nltk                                      
nltk.download('stopwords')
from nltk.corpus import stopwords                                
from gensim.models import KeyedVectors

from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, Flatten, Conv1D, MaxPooling1D, Embedding, merge, Dropout, GlobalMaxPooling1D
from keras.models import Model
from sklearn.model_selection import train_test_split
import itertools


"""### Loading the data"""

url="http://qim.fs.quoracdn.net/quora_duplicate_questions.tsv"
original_data=pd.read_csv(url,error_bad_lines=False,sep='\t')

test_list = [['what is the best country in the world. USA, england or india?', 'which engineering has the best salary in india?']]
test_data = pd.DataFrame(test_list, columns = ['question1', 'question2'])
test = {'question1': test_data.question1, 'question2': test_data.question2}


stop = set(stopwords.words('english')) 

"""Next, I defined `sentence_to_wordlist` which takes each sentence from one of the two columns `question1` and `question2`. On each sentence it applies beautifulsoup for removing html tags if any. Then on that string I applied `regular expression` for some text preprocessing like include upper and lower case alphabets, 0-9 numbers, correct short forms and so on.
Then applied `.lower()` to bring everything to lowercase to maintain regularity in every word and `.split()` method to convert the sentence into list of words. Finally, on these list of words I iterated one by one to check if that word is not a `stopword` if not then it is returned.
"""
def sentence_to_wordlist(sentence):
    
    sentence = BeautifulSoup(sentence, "html.parser")  
    sentence = sentence.get_text()

    sentence = re.sub(r"[^A-Za-z%@#$&*]", " ", sentence)
    sentence = re.sub(r"what's", "what is ", sentence)
    sentence = re.sub(r"\'s", " ", sentence)
    sentence = re.sub(r"\'ve", " have ", sentence)
    sentence = re.sub(r"can't", "cannot ", sentence)
    sentence = re.sub(r"n't", " not ", sentence)
    sentence = re.sub(r"i'm", "i am ", sentence)
    sentence = re.sub(r"\'re", " are ", sentence)
    sentence = re.sub(r"\'d", " would ", sentence)
    sentence = re.sub(r"\'ll", " will ", sentence)
    sentence = re.sub(r",", " ", sentence)
    sentence = re.sub(r"\.", " ", sentence)
    sentence = re.sub(r"\/", " ", sentence)
    sentence = re.sub(r"\^", " ^ ", sentence)
    sentence = re.sub(r"\+", " + ", sentence)
    sentence = re.sub(r"\=", " = ", sentence)
    sentence = re.sub(r"'", " ", sentence)
    sentence = re.sub(r"(\d+)(k)", r"\g<1>000", sentence)
    sentence = re.sub(r":", " : ", sentence)
    sentence = re.sub(r" e g ", " eg ", sentence)
    sentence = re.sub(r" b g ", " bg ", sentence)
    sentence = re.sub(r" u s ", " american ", sentence)
    sentence = re.sub(r"\0s", "0", sentence)
    sentence = re.sub(r"e - mail", "email", sentence)
    sentence = re.sub(r"j k", "jk", sentence)
    sentence = re.sub(r"\s{2,}", " ", sentence)
    
    sentence = sentence.lower().split()


    stops = set(stopwords.words("english"))
    sentence = [w for w in sentence if not w in stops]

    return(sentence)

"""Here I defined the two columns on which I applied the preprocessing."""

columns = ['question1', 'question2']

"""Next, I iterated over each row by using dataframes function called `iterrows()` which iterates over each record or row of the dataframe. For each row I iterated over the two columns namely `question1`, `question2` and then for each record I called the `sentence_to_wordlist()` function passing in the row and one of the two columns. Using pandas `at()` function I updated for each row both columns `question1` and `question2`."""

for indices, record in test_data.iterrows():
        # Iterate through the text of both questions of the row
        for column in columns:
            test_data.at[indices, column] =  sentence_to_wordlist(record[column])


"""After I preprocessed the sentences into list of words, next I assigned each unique word in the whole corpuse a number, so that I could pass this as an input to the model and also create a word2vec representation of these vocabularies (numbers).

For doing this, I initialised a dictionary of variable `vocabulary()` which stored each word as a key and a number as a value respectively. Another variable called `inverse_vocabulary()` which is a list that holds the value or number for each unique word. It was initialised which an `<unk>` token since we want to zero pad the words with a number zero I did not want to assign any word a number 0. Hence, initialised with a `<unk>` token which holds the value zero.

Similar to above, I again iterate over the dataframe using `iterrows()` function, for each question in a row I iterate over all the words one by one. First I check whether the word is already in the dictionary `vocabulary()` if the word is not there then a value based on the length of the `inverse_vocabulary` is assigned to that new word (key), the inverse_vocabulary is updated with the new value along with it.

To update the dataframe `data` with numbers, I have a list named `sentence_to_numbers` which will append a value (number) for each word. Then using `at()` function the dataframe for each question of the particular row was updated with the list of word indices.
"""

vocabulary = dict()
inverse_vocabulary = ['<unk>']  

for indices, record in test_data.iterrows():
         for column in columns:

            sentence_to_numbers = []  
            for word in record[column]:

               
                if word not in vocabulary:
                    vocabulary[word] = len(inverse_vocabulary)
                    sentence_to_numbers.append(len(inverse_vocabulary))
                    inverse_vocabulary.append(word)
                else:
                    sentence_to_numbers.append(vocabulary[word])

            test_data.at[indices, column] =  sentence_to_numbers




"""Next, I created the embedding matrix for my vocabulary. For which I used `gensim` library and google's pre-trained word2vec model. Google's pre-trained word2vec model gives a 300 dimensional vector for each word which will be fed to the `Embedding layer` of my model. Since, I will pad my sentences with a zero, I initialise the embedding matrix's zeroth element as zero. The size of the embedding matrix will be `(Size of Vocabulary + 1 (for zero) X 300 (embedding dim))`.

I iterated over vocabulary and for each word corresponding to its index I store the 300 dimensional vector in the `embeddings` numpy array.

If the word is not there in the Google's pretrained model then that word will be randomly initialised, since the `embeddings` array is initialised randomly beforehand.
"""

word2vec = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin',binary=True)

embedding_dim = 300
embeddings = 1 * np.random.randn(len(vocabulary) + 1, embedding_dim)  # Initialising the embedding matrix randomly
embeddings[0] = 0  
# Build the embedding matrix
for word, index in vocabulary.items():
    if word in word2vec.vocab:
        embeddings[index] = word2vec.word_vec(word)





"""Since, my network will have two inputs, the data was divided as `question1` and `question2` in a dictionary fashion."""

#test = {'question1': X_train.question1, 'question2': X_train.question2}

"""Next, using `itertools()` function on both training and validation data, I padded each sentence with zeros to make each sequence of same size i.e. `103`. By default, Keras will pad zeros in a `pre-order` i.e. before the sequence."""

#for dataset, side in itertools.product([train, val], ['question1', 'question2']):
#    dataset[side] = pad_sequences(dataset[side], maxlen=max_seq_length)



import keras
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Lambda,Dropout,merge,Lambda,Reshape
from keras.layers import BatchNormalization, Bidirectional, GlobalMaxPool1D
import keras.backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import regularizers
import numpy.random as rng
from keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
from keras import models
import tensorflow as tf


"""Next, using `itertools()` function on both training and validation data, I padded each sentence with zeros to make each sequence of same size i.e. `103`. By default, Keras will pad zeros in a `pre-order` i.e. before the sequence."""
max_seq_length = 97

for dataset, side in itertools.product([test], ['question1', 'question2']):
    dataset[side] = pad_sequences(dataset[side], maxlen=max_seq_length)


X_train, X_validation, Y_train, Y_validation = train_test_split(original_data,original_data.is_duplicate, random_state=13,test_size=0.2)

train_data = X_train.question1.append(X_train.question2)

train_data = train_data.reset_index()

train_data = train_data.drop(['index'], axis=1)

train_data.columns = ['questions']

"""### Breaking the network to extract output of 128 feature maps from the middle( ) function by creating a new model

First I load the model again which has both weights and model, then just save the weights.
"""

print('test_data', test_data)
print('test', test)
print('test[question1]', test['question1'])

quora1 = models.load_model('quora_lstm_max10_dense.h5')

test_prediction = quora1.predict([test['question1'],test['question2']]) #test predictions
test_prediction = np.array(test_prediction) #converting the list of predictions into numpy array

test_pred = np.reshape(test_prediction,(-1,128))

query_pred = test_pred

train_prediction = np.load('train_predictions.npy')
train_pred = np.reshape(train_prediction,(-1,128))

data_pred = train_pred

query_questions = test_data

data_questions = train_data




"""## Brute-Force Method for Finding the Top-3 Closest from the Training data for a given Input Query"""

import heapq

no_questions = 1
main_array = np.zeros((no_questions,3)) #(no of query questions, 3)

print('data_pred.shape[0]', data_pred.shape[0])

def comparison(query):
    arr = []
    for i in range(data_pred.shape[0]):
            predict = np.linalg.norm(query - data_pred[i])
            arr.append(predict)
    hp = np.array(heapq.nsmallest(3, range(len(arr)), arr.__getitem__))
    print(hp)
    return hp

"""I took only 100 query questions since this method takes 313 seconds to output top-3 suggestions for each input query time!"""

import time
start = time.clock()
for i in range(no_questions):
    main_array[i,:] = comparison(query_pred[i])
print (time.clock() - start)

#main_array.shape

np.set_printoptions(suppress=True)

main_array = main_array.astype(np.int64)

pd.set_option('display.max_colwidth', -1)


import csv
import os
filename = 'test_brute_force.csv'
a = open(filename, 'a')

headers = ['Query', 'Closest-1','Closest-2','Closest-3']
writer = csv.DictWriter(a, delimiter='\t', lineterminator='\n',fieldnames=headers)
fileEmpty = os.stat(filename).st_size == 0
writer.writeheader()
for i in range(len(main_array)):
    a.write((str(query_questions.iloc[i])).split('\n')[0].split('    ')[1]+'\t'+str(data_questions.iloc[main_array[i][0]]).split('\n')[0].split('    ')[1] +'\t'+ str(data_questions.iloc[main_array[i][1]]).split('\n')[0].split('    ')[1] +'\t'+ str(data_questions.iloc[main_array[i][2]]).split('\n')[0].split('    ')[1] + "\n")
    #a.write(((query_questions.iloc[i]).encode('utf-8')).split('\n')[0].split('    ')[1]+'\t'+(data_questions.iloc[main_array[i][0]]).encode('utf-8').split('\n')[0].split('    ')[1] +'\t'+ (data_questions.iloc[main_array[i][1]]).encode('utf-8').split('\n')[0].split('    ')[1] +'\t'+ (data_questions.iloc[main_array[i][2]]).encode('utf-8').split('\n')[0].split('    ')[1] + "\n")
a.close()

data_brute = pd.read_csv('test_brute_force.csv',sep='\t')
print('data_brute', data_brute)

