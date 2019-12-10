#
# Load trained preferred model to generate new text and visualize it
#

import string
import textwrap
import pickle
import sys
from random import randint
from collections import defaultdict

from keras.models import load_model

from lib.nlplstm_class import TFModelLSTMWord2vec 
from lib.data_common import (load_doc, prepare_text_tokens, generate_seq_of_words)
from lib.text_viz_common import (Init_Text_Viz_Params, Find_Entity, 
                                Modify_For_Full_Names, List_Notables, Viz_Generated_Text)

#
# Initialization
#

# input text files
pathfinder_textfile = './data/textgen_pathfinder.txt'
fixed_length_token_textfile = './data/pathfinder_fixed-length_tokens.txt'

# load fixed-length lines of tokens
doc = load_doc(fixed_length_token_textfile)
lines = doc.split('\n')
#print('Total lines: %d' % len(lines))

# tokenize and separate into features and target
X, y, seq_length, vocab_size, tokenizer = prepare_text_tokens(lines)

# load Word2vec pre-trained model
textgen_model = load_model('./model/pathfinder_wordtoken_w2v_model_50_epoch_noncuda_model.h5')

# initialize text visualization parameters
nlp, matcher, entity_names, entity_labels = Init_Text_Viz_Params()


#
# Text generation and visualization
#

# select a seed text
seed_text = lines[randint(0,len(lines))]
# print('> using word tokenisation and pre-trained model')
# print('> seed text:')
# print(textwrap.fill('%s' % (seed_text), 80) + '\n')

# generate new text using selected seed text with a temperature of 1.1
# for higher degree of randomness
generated = generate_seq_of_words(textgen_model, tokenizer, seq_length, 
                    seed_text, 75, 1.1)
full_generated_text = ' '.join([seed_text, generated])

# output visualized text via spaCy library functions, that highlight named entities
# pertaining to input files of Pathfinder tales
Viz_Generated_Text(full_generated_text, nlp, matcher, entity_names, entity_labels)
