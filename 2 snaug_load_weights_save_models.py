#
# Load saved weights of models that were previously trained on cloud platforms 
# using GPU (eg. Google Colab)
# Save instances of trained models for future prediction tasks on platforms without 
# using GPU (eg. my wimpy lappy)
#

import string
import textwrap
import pickle

from lib.nlplstm_class import (TFModelLSTMCharToken, TFModelLSTMWordToken, 
                               TFModelLSTMWord2vec) 
from lib.data_common import (load_doc, save_doc, clean_doc, prepare_char_tokens)
from lib.data_common import (build_token_lines, prepare_text_tokens, load_word2vec)
from lib.data_common import pathfinder_textfile, fixed_length_token_textfile

# 
# LSTM model that uses character tokenisation  
#
# load document
text = load_doc(pathfinder_textfile).lower()

# tokenize character data and separate into features and target for LSTM training
maxlen = 40
step = 3
X, y, char2indices, indices2char, num_unique_char = prepare_char_tokens(text, maxlen, step)

# create new object that is an LSTM model using character tokenization
# to generate text
# this model does not use CudaDNN library
#
textgen_model_1 = TFModelLSTMCharToken(use_gpu=False)

# define the model parameters
textgen_model_1.define(maxlen, num_unique_char)
print(textgen_model_1.model.summary())

# load model weights trained on platform using GPU
textgen_model_1.load_weights("./model/pathfinder_chartoken_model_50_epoch")

# save model updated with previously trained model weights
textgen_model_1.save("./model/pathfinder_chartoken_model_50_epoch_noncuda")

print()



# 
# Loading text data that uses word tokenisation
#

# load fixed-length lines of tokens
doc = load_doc(fixed_length_token_textfile)
lines = doc.split('\n')
#print('Total lines: %d' % len(lines))

# tokenize and separate into features and target
X, y, seq_length, vocab_size, tokenizer = prepare_text_tokens(lines)
#print(X.shape)

#
# Word tokenization with word embedding model
#

# create new object that is an LSTM model using word tokenization
# and word embedding to generate text
# this model does not use CudaDNN library
#
textgen_model_2 = TFModelLSTMWordToken(use_gpu=False)

# define the model parameters
textgen_model_2.define(vocab_size=vocab_size, 
                       embedding_size=300, 
                       seq_length=seq_length)
print(textgen_model_2.model.summary())

# load model weights trained on platform using GPU
textgen_model_2.load_weights("./model/pathfinder_wordtoken_model_200_epoch")

# save model updated with previously trained model weights
textgen_model_2.save("./model/pathfinder_wordtoken_model_200_epoch_noncuda")

print()

#
# Word2vec pre-trained model
#

# load gensim Word2Vec word model's pretrained weights
pretrained_weights = pickle.load(open('./model/pathfinder_wordtoken_w2v_word_model_weights.pkl', 'rb'))
vocab_size, emdedding_size = pretrained_weights.shape

# create new object that is an LSTM model using word tokenization
# and pre-trained Word2vec model form Gensim to generate text
# this model does not use CudaDNN library
#
textgen_model_3 = TFModelLSTMWord2vec(use_gpu=False)

# define the model parameters
textgen_model_3.define(vocab_size=vocab_size, 
                       embedding_size=emdedding_size, 
                       pretrained_weights=pretrained_weights)
print(textgen_model_3.model.summary())

# load model weights trained on platform using GPU
textgen_model_3.load_weights("./model/pathfinder_wordtoken_w2v_model_50_epoch")

# save model updated with previously trained model weights
textgen_model_3.save("./model/pathfinder_wordtoken_w2v_model_50_epoch_noncuda")
