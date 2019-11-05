"""
Train and save model data (tokens, weights, etc) to be used on different platforms
"""

import tensorflow as tf
import gensim

import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
import pickle

from gensim.test.utils import datapath, get_tmpfile
from gensim.models import Word2Vec, KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

import string
import textwrap

from lib.nlplstm_class import (TFModelLSTMWordToken, TFModelLSTMWord2vec) 

pathfinder_textfile = './data/textgen_pathfinder.txt'
fixed_length_token_textfile = 'pathfinder_fixed-length_tokens.txt'

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# save tokens to file, one dialog per line
def save_doc(lines, filename):
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()

# turn a doc into clean tokens
def clean_doc(doc):
	# replace '--' with a space ' '
	doc = doc.replace('--', ' ')
    # replace '-' with a space ' '
	doc = doc.replace('-', ' ')
    # split into tokens by white space
	tokens = doc.split()
	# remove punctuation from each token
	table = str.maketrans('', '', string.punctuation)
	tokens = [w.translate(table) for w in tokens]
	# remove remaining tokens that are not alphabetic
	tokens = [word for word in tokens if word.isalpha()]
	# make lower case
	tokens = [word.lower() for word in tokens]
	return tokens

# organize into fixed-length lines of tokens
def build_token_lines(tokens, length=50):
	length += 1
	lines = list()
	for i in range(length, len(tokens)):
		# select sequence of tokens
		seq = tokens[i-length:i]
		# convert into a line
		line = ' '.join(seq)
		# store
		lines.append(line)
	return lines

# prepare text tokens into format ready for LSTM training
def prepare_text_tokens(lines):
	# integer encode sequences of words
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	sequences = tokenizer.texts_to_sequences(lines)

	# vocabulary size
	vocab_size = len(tokenizer.word_index)
	#print(tokenizer.word_index)

	# split into X and y
	npsequences = np.array(sequences)
	X, y = npsequences[:,:-1], npsequences[:,-1]
	y = to_categorical(y, num_classes=vocab_size+1)
	seq_length = X.shape[1]
	
	return X, y, seq_length, vocab_size, tokenizer


#
# Word tokenization with word embedding model
#

# load document
docs = load_doc(pathfinder_textfile)
#print(docs[:200])
print(textwrap.fill('%s' % (docs[:200]), 80))

# pre-process and tokenize document
tokens = clean_doc(docs)
print(tokens[:20])
print('Total Tokens: %d' % len(tokens))
print('Unique Tokens: %d' % len(set(tokens)))

# organize into fixed-length lines of tokens
lines = build_token_lines(tokens, length=50)
print('Total lines: %d' % len(lines))

# save fixed-length lines to file
save_doc(lines, fixed_length_token_textfile)

# tokenize and separate into input and output
X, y, seq_length, vocab_size, tokenizer = prepare_text_tokens(lines)
print(X.shape)

# split tokens up per line for Gensim Word2vec consumption
sentences = [line.split() for line in lines]

# create new object that is an LSTM model using word tokenization
# and word embedding to generate text
textgen_model_2 = TFModelLSTMWordToken(use_gpu=False)
	
# sanity check
print(textgen_model_2.model_name)
print(textgen_model_2.have_gpu)
print(textgen_model_2.use_cudadnn)

# define and compile the model parameters
textgen_model_2.define(vocab_size=vocab_size, 
                         embedding_size=100, 
                         seq_length=seq_length)
print(textgen_model_2.model.summary())

# compile model
textgen_model_2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit model
#history = textgen_model_2.fit(X, y, batch_size=128, epochs=200)
history = textgen_model_2.fit(X, y, batch_size=128, epochs=2)

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
sequences = np.array(sequences)
X, y = sequences[:,:-1], sequences[:,-1]
y = to_categorical(y, num_classes=(input_size))
seq_length = X.shape[1]

my_2nd_nlp_model = TFModelLSTMWord2vec(use_gpu=False)

print(my_2nd_nlp_model.model_name)
print(my_2nd_nlp_model.have_gpu)
print(my_2nd_nlp_model.use_cudadnn)

my_2nd_nlp_model.define(vocab_size=vocab_size, 
                             embedding_size=emdedding_size, 
                             pretrained_weights=pretrained_weights)
print(my_2nd_nlp_model.model.summary())

# compile model
my_2nd_nlp_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit model
#history = model.fit(X, y, batch_size=128, epochs=100)
#history = my_2nd_nlp_model.fit(X, y, batch_size=128, epochs=50)
history = my_2nd_nlp_model.fit(X, y, batch_size=128, epochs=5)
