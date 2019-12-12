#
# Load trained preferred model to generate new text and visualize it
#

import string
import textwrap
import pickle
import sys
from random import randint

from keras.models import load_model

from lib.data_common import (load_doc, prepare_text_tokens, generate_seq_of_words)
from lib.data_common import pathfinder_textfile, fixed_length_token_textfile
from lib.text_viz_common import init_text_viz_params, visualize_gen_text

#
# Initialization
#

# load fixed-length lines of tokens
doc = load_doc(fixed_length_token_textfile)
lines = doc.split('\n')
#print('Total lines: %d' % len(lines))

# tokenize and separate into features and target
X, y, seq_length, vocab_size, tokenizer = prepare_text_tokens(lines)

# load Word2vec pre-trained model
textgen_model = load_model('./model/pathfinder_wordtoken_w2v_model_50_epoch_noncuda_model.h5')

# initialize text visualization parameters
nlp, matcher, entity_names, entity_labels = init_text_viz_params()


#
# Text generation and visualization
#

def generate_and_visualize(lines, textgen_model, tokenizer, seq_length,
                            nlp, matcher, entity_names, entity_labels,
                            seed_text='random'):
    """
    generate new text using seed text based one:
        1. user input
        2. randomly selected from source text, when user enter 'random' instead
    """
    n_words = 75
    temperature = 1.1

    # select a seed text
    seed_text = lines[randint(0,len(lines))] if seed_text=='random' else seed_text

    # print('> using word tokenisation and pre-trained model')
    # print('> seed text:')
    # print(textwrap.fill('%s' % (seed_text), 80) + '\n')

    # generate new text using selected seed text with a temperature of 1.1
    # for higher degree of randomness
    generated = generate_seq_of_words(textgen_model, tokenizer, seq_length, 
                        seed_text, n_words, temperature)
    full_generated_text = ' '.join([seed_text, generated])

    # output visualized text via spaCy library functions, that highlight named entities
    # pertaining to input files of Pathfinder tales
    visualize_gen_text(full_generated_text, nlp, matcher, entity_names, entity_labels)


seed_text = input("Enter text to generate a story ('random' to automate, 'quit' to quit): ")

generate_and_visualize(lines, textgen_model, tokenizer, seq_length,
                        nlp, matcher, entity_names, entity_labels,
                        seed_text='random')