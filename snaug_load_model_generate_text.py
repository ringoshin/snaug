#
# Load trained models to generate new text
#

import string
import textwrap
import pickle
import sys
from random import randint

from keras.models import load_model

from lib.nlplstm_class import (TFModelLSTMCharToken, TFModelLSTMWordToken, 
                               TFModelLSTMWord2vec) 
from lib.data_common import (load_doc, save_doc, clean_doc, prepare_char_tokens,
                             build_token_lines, prepare_text_tokens, load_word2vec,
                             sample_predict, generate_seq_of_chars, generate_seq_of_words)

pathfinder_textfile = './data/textgen_pathfinder.txt'
fixed_length_token_textfile = './data/pathfinder_fixed-length_tokens.txt'

# parameters for character tokenisation model
maxlen = 40
step = 3

# load fixed-length lines of tokens
doc = load_doc(fixed_length_token_textfile)
lines = doc.split('\n')

# generate word_seed_text for word tokenisation models
# generate separate char_seed_text for char tokenisation model, using
# maxlen rightmost characters of word seed text
word_seed_text = lines[randint(0,len(lines))]
char_seed_text = word_seed_text[-maxlen:]

# 
# LSTM model that uses character tokenisation  
#
# load document
text = load_doc(pathfinder_textfile).lower()

# tokenize character data and separate into features and target for LSTM training
X, y, char2indices, indices2char, num_unique_char = prepare_char_tokens(text, maxlen, step)

# load trained model
textgen_model_1 = load_model('./model/pathfinder_chartoken_model_50_epoch_noncuda_model.h5')

# select seed text
#start_index = randint(0, len(text) - maxlen - 1)
#seed_text = text[start_index: start_index + maxlen]
print()
print('> using character tokenisation')
print('> seed text:')
print(textwrap.fill('%s' % (char_seed_text), 80) + '\n')

# generate new text
#temperature_table = [0.2, 0.7,1.2] 
#temperature_table = [0.07, 0.1, 0.5, 0.7] 
temperature_table = [0.1, 1.0] 

#seed_text = "mine".rjust(maxlen)

for temperature in temperature_table:
    generated = generate_seq_of_chars(textgen_model_1, num_unique_char, char2indices, 
                    indices2char, char_seed_text, maxlen, 300, temperature)    
    print(">> generated text (temperature: {})".format(temperature))
    print(textwrap.fill('%s' % (generated), 80))
    print()

print()



# 
# Loading text data that uses word tokenisation
#

# load fixed-length lines of tokens
#doc = load_doc(fixed_length_token_textfile)
#lines = doc.split('\n')
#print('Total lines: %d' % len(lines))

# tokenize and separate into features and target
X, y, seq_length, vocab_size, tokenizer = prepare_text_tokens(lines)
#print(X.shape)

#
# Word tokenization with word embedding model
#

# load trained model
textgen_model_2 = load_model('./model/pathfinder_wordtoken_model_200_epoch_noncuda_model.h5')


# select a seed text
#word_seed_text = lines[randint(0,len(lines))]
print('> using word tokenisation')
print('> seed text:')
print(textwrap.fill('%s' % (word_seed_text), 80) + '\n')

# generate new text
temperature_table = [0.1, 1.0]

for temperature in temperature_table:
    generated = generate_seq_of_words(textgen_model_2, tokenizer, seq_length, 
                    word_seed_text, 100, 0.2)
    print(">> generated text (temperature: {})".format(temperature))
    print(textwrap.fill('%s' % (generated), 80))
    print()

print()

#
# Word2vec pre-trained model
#

# load trained model
textgen_model_3 = load_model('./model/pathfinder_wordtoken_w2v_model_50_epoch_noncuda_model.h5')

# select a seed text
#word_seed_text = lines[randint(0,len(lines))]
print('> using word tokenisation and pre-trained model')
print('> seed text:')
print(textwrap.fill('%s' % (word_seed_text), 80) + '\n')

# generate new text
temperature_table = [0.1, 1.0]

for temperature in temperature_table:
    generated = generate_seq_of_words(textgen_model_3, tokenizer, seq_length, 
                    word_seed_text, 100, 0.2)
    print(">> generated text (temperature: {})".format(temperature))
    print(textwrap.fill('%s' % (generated), 80))
    print()
