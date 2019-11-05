"""
Load saved data of model(s) trained from different platform (eg. Google Colab)
"""

import tensorflow as tf
import gensim

from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
import pickle

from gensim.test.utils import datapath, get_tmpfile
from gensim.models import Word2Vec, KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

from lib.nlplstm_class import (TFModelLSTMWordToken, TFModelLSTMWord2vec) 

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# load
in_filename = './data/pathfinder_token_sequences.txt'
doc = load_doc(in_filename)
lines = doc.split('\n')

#
# Word tokenization with word embedding model
#

# Load saved tokenized text
# Derive vocab_size and seq_length
tokenizer = pickle.load(open('./model/pathfinder_token_tokenizer.pkl', 'rb'))
sequences = tokenizer.texts_to_sequences(lines)
vocab_size = len(tokenizer.word_index) + 1

# separate into input and output
sequences = array(sequences)
X, y = sequences[:,:-1], sequences[:,-1]
y = to_categorical(y, num_classes=vocab_size)
seq_length = X.shape[1]

my_nlp_model = TFModelLSTMWordToken(use_gpu=False)

print(my_nlp_model.model_name)
print(my_nlp_model.have_gpu)
print(my_nlp_model.use_cudadnn)

my_nlp_model.define_LSTM(vocab_size=vocab_size, 
                         embedding_size=100, 
                         seq_length=seq_length)
print(my_nlp_model.model.summary())
print()

#
# Word2vec pre-trained model
#

# Load saved weights pre-trained using Word2vec from Gemsim
#word_model = pickle.load(open('./model/pathfinder_token_w2v300_word_model.pkl', 'rb'))
pretrained_weights = pickle.load(open('./model/pathfinder_token_w2v300_weights.pkl', 'rb'))

# Set vocab_size and embedding_size
#pretrained_weights = word_model.wv.syn0
vocab_size, emdedding_size = pretrained_weights.shape

# Load saved tokenized text
# Derive vocab_size and seq_length
tokenizer = pickle.load(open('./model/pathfinder_token_tokenizer.pkl', 'rb'))
sequences = tokenizer.texts_to_sequences(lines)
input_size=vocab_size+1

# separate into input and output
sequences = array(sequences)
X, y = sequences[:,:-1], sequences[:,-1]
y = to_categorical(y, num_classes=(input_size))
seq_length = X.shape[1]

my_2nd_nlp_model = TFModelLSTMWord2vec(use_gpu=False)

print(my_2nd_nlp_model.model_name)
print(my_2nd_nlp_model.have_gpu)
print(my_2nd_nlp_model.use_cudadnn)

my_2nd_nlp_model.define_LSTM(vocab_size=vocab_size, 
                             embedding_size=emdedding_size, 
                             pretrained_weights=pretrained_weights)

print(my_2nd_nlp_model.model.summary())
