"""
Data loading, saving and pre-processing functions
"""

import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

from gensim.models import Word2Vec
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import Word2Vec, KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

import numpy as np
import pickle

import string
import textwrap


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

# tokenize character data and separate into features and target for LSTM training
def prepare_char_tokens(text, maxlen, step):
    print('corpus length:', len(text))

    # get list of unique characters from text
    characters = sorted(list(set(text)))
    num_unique_char = len(characters)
    print('total chars:', num_unique_char)

    # store mappings of character to index and vice versa
    char2indices = dict((c, i) for i, c in enumerate(characters))
    indices2char = dict((i, c) for i, c in enumerate(characters))

    # cut the text in semi-redundant sequences of maxlen characters
    sentences = []
    next_chars = []
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])
    print('number of sequences:', len(sentences))
    print('number of next_chars:', len(next_chars))

    # Converting indices into vectorized format
    X = np.zeros((len(sentences), maxlen, num_unique_char), dtype=np.bool)
    y = np.zeros((len(sentences), num_unique_char), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, char2indices[char]] = 1
        y[i, char2indices[next_chars[i]]] = 1

    return X, y, char2indices, indices2char, num_unique_char

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

def load_word2vec(lines):
    # split tokens up per line for Gensim Word2vec consumption
    sentences = [line.split() for line in lines]

    print('\nTraining word2vec...')
    # workers=1 will ensure a fully deterministrically-reproducible run, per Gensim docs
    word_model = Word2Vec(sentences, size=300, min_count=1, window=5, iter=100, workers=1)
    pretrained_weights = word_model.wv.syn0
    vocab_size, emdedding_size = pretrained_weights.shape
    print('Result embedding shape:', pretrained_weights.shape)

    return vocab_size, emdedding_size, pretrained_weights


if __name__ == '__main__':
    pass
