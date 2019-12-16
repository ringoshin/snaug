#
# Train models on cloud platforms using GPU, then save only the model weights 
# to be reloaded later on machines without powerful GPU.
#

from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

import string
import textwrap
import pickle

from lib.nlplstm_class import (TFModelLSTMCharToken, TFModelLSTMWordToken, 
                               TFModelLSTMWord2vec) 
from lib.data_common import (load_doc, save_doc, clean_doc, prepare_char_tokens)
from lib.data_common import (build_token_lines, prepare_text_tokens, load_word2vec)
from lib.data_common import pathfinder_textfile, fixed_length_token_textfile

# 
# Loading, saving and pre-processing of the text data source for 
# character tokenisation
#
# load document
text = load_doc(pathfinder_textfile).lower()

# tokenize character data and separate into features and target for LSTM training
maxlen = 40
step = 3
X, y, char2indices, indices2char, num_unique_char = prepare_char_tokens(text, maxlen, step)

# save the mappings
pickle.dump(char2indices, open('./model/pathfinder_chartoken_char2indices.pkl', 'wb'))
pickle.dump(indices2char, open('./model/pathfinder_chartoken_indices2char.pkl', 'wb'))

# create new object that is an LSTM model using character tokenization
# to generate text
textgen_model_1 = TFModelLSTMCharToken(use_gpu=False)

# define and compile the model parameters
textgen_model_1.define(maxlen, num_unique_char)

# LSTM object sanity check
print(textgen_model_1.model_name)
print(textgen_model_1.have_gpu)
print(textgen_model_1.use_cudadnn)
print(textgen_model_1.model.summary())

# compile model
textgen_model_1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit model
#history = textgen_model_1.fit(X, y, batch_size=128, epochs=50)
history = textgen_model_1.fit(X, y, batch_size=128, epochs=2)

# plot accuracy vs error for training
textgen_model_1.plot_training()

# serialize model weights to HDF5 and save model training history
textgen_model_1.save_weights_and_history(fname_prefix="./model/pathfinder_chartoken_model_50_epoch")

print()


# 
# Loading, saving and pre-processing of the text data source for
# word tokenisation
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

# tokenize and separate into features and target
X, y, seq_length, vocab_size, tokenizer = prepare_text_tokens(lines)
print(X.shape)


#
# Word tokenization with word embedding model
#

# create new object that is an LSTM model using word tokenization
# and word embedding to generate text
textgen_model_2 = TFModelLSTMWordToken(use_gpu=False)
	
# define and compile the model parameters
textgen_model_2.define(vocab_size=vocab_size, 
                       embedding_size=300, 
                       seq_length=seq_length)

# LSTM object sanity check
print(textgen_model_2.model_name)
print(textgen_model_2.have_gpu)
print(textgen_model_2.use_cudadnn)
print(textgen_model_2.model.summary())

# compile model
textgen_model_2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit model
#history = textgen_model_2.fit(X, y, batch_size=128, epochs=200)
history = textgen_model_2.fit(X, y, batch_size=128, epochs=2)

# plot accuracy vs error for training
textgen_model_2.plot_training()

# serialize model weights to HDF5 and save model training history
textgen_model_2.save_weights_and_history(fname_prefix="./model/pathfinder_wordtoken_model_200_epoch")

print()

#
# Word2vec pre-trained model
#

# get pretrained weights for LSTM model's word embedding using Gensim Word2vec
vocab_size, emdedding_size, pretrained_weights = load_word2vec(lines)

# save gensim Word2Vec word model's pretrained weights
pickle.dump(pretrained_weights, open('./model/pathfinder_wordtoken_w2v_word_model_weights.pkl', 'wb'))

# create new object that is an LSTM model using word tokenization
# and pre-trained Word2vec model from Gensim to generate text
textgen_model_3 = TFModelLSTMWord2vec(use_gpu=False)

textgen_model_3.define(vocab_size=vocab_size, 
                       embedding_size=emdedding_size, 
                       pretrained_weights=pretrained_weights)

# LSTM object sanity check
print(textgen_model_3.model_name)
print(textgen_model_3.have_gpu)
print(textgen_model_3.use_cudadnn)
print(textgen_model_3.model.summary())

# compile model
textgen_model_3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit model
#history = model.fit(X, y, batch_size=128, epochs=100)
#history = textgen_model_3.fit(X, y, batch_size=128, epochs=50)
history = textgen_model_3.fit(X, y, batch_size=128, epochs=2)

# plot accuracy vs error for training
textgen_model_3.plot_training()

# serialize model weights to HDF5 and save model training history
textgen_model_3.save_weights_and_history(fname_prefix="./model/pathfinder_wordtoken_w2v_model_50_epoch")
