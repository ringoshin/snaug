#
# Data loading, saving and pre-processing functions
#

import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

from gensim.models import Word2Vec
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import Word2Vec, KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

import numpy as np
import pickle

import string
import textwrap

pathfinder_textfile = './data/textgen_pathfinder.txt'
fixed_length_token_textfile = './data/pathfinder_fixed-length_tokens.txt'


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

# load Gensim Word2vec model, train it and keep the pretrained weights
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

# helper function to predict using random sampling, 
# to return an index from a probability array
def sample_predict(preds, temperature=1.0):
    if temperature <= 0:
        return np.argmax(preds)
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
#    print(preds)
    exp_preds = np.exp(preds)
#    print(exp_preds)
    preds = exp_preds / np.sum(exp_preds)
#    print(preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# generate a sequence of characters with a language model
def generate_seq_of_chars(model, num_unique_char, char2indices, indices2char, 
                          seed_text, maxlen=40, n_chars=400, temperature=1.0):
    generated = ''
    #generated += seed_text
    in_text = seed_text
#    sys.stdout.write(generated)

    for _ in range(n_chars):
        x = np.zeros((1, maxlen, num_unique_char))
        for t, char in enumerate(in_text):
            x[0, t, char2indices[char]] = 1.

        preds = model.predict(x, verbose=0)[0]
        next_index = sample_predict(preds, temperature)
        pred_char = indices2char[next_index]

        generated += pred_char
        in_text = in_text[1:] + pred_char

 #       sys.stdout.write(pred_char)
  #      sys.stdout.flush()
    return generated

# generate a sequence of words from a language model
def generate_seq_of_words(model, tokenizer, seq_length, seed_text, n_words, temperature=0):
    result = list()
    in_text = seed_text
    # generate a fixed number of words
    for _ in range(n_words):
        # encode the text as integer
        #   [0] coz shape is (num_lines, num_tokens_per_line)
        #   and there is only one line right now
        #   also num_tokens is not a const, hence trunc will be required
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        # truncate sequences to a fixed lenght
        encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
        # predict probabilities for each word
        yhat = sample_predict(model.predict(encoded,verbose=0)[-1], temperature)
        # map predicted word index to word
        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index == yhat:
                out_word = word
                break
        # append to input
        in_text += ' ' + out_word
        result.append(out_word)
    return ' '.join(result)


if __name__ == '__main__':
	#preds=[0.05, 0.1, 0.35, 0.5]
	#np.random.multinomial(1, preds, 1)
	#print([sample_predict(preds,0.05) for _ in range(10)])
	#print([sample_predict(preds,1) for _ in range(10)])
	#print([sample_predict(preds,50) for _ in range(10)])
	#sample_predict(preds,1.2)
    pass
