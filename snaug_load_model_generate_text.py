#
# Load trained models to generate new text
#

import string
import textwrap
import pickle
import sys

import numpy as np
from random import randint

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

from lib.nlplstm_class import (TFModelLSTMCharToken, TFModelLSTMWordToken, 
                               TFModelLSTMWord2vec) 
from lib.data_common import (load_doc, save_doc, clean_doc, prepare_char_tokens)
from lib.data_common import (build_token_lines, prepare_text_tokens, load_word2vec)

pathfinder_textfile = './data/textgen_pathfinder.txt'
fixed_length_token_textfile = './data/pathfinder_fixed-length_tokens.txt'

# 
# LSTM model that uses character tokenisation  
#
# load document
text = load_doc(pathfinder_textfile).lower()

# tokenize character data and separate into features and target for LSTM training
maxlen = 40
step = 3
X, y, char2indices, indices2char, num_unique_char = prepare_char_tokens(text, maxlen, step)

# load trained model
textgen_model_1 = load_model('./model/pathfinder_chartoken_model_50_epoch_noncuda_model.h5')

# Function to convert prediction into index
def pred_indices(preds, metric=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / metric
#    print(preds)
    exp_preds = np.exp(preds)
#    print(exp_preds)
    preds = exp_preds/np.sum(exp_preds)
#    print(preds)
    probs = np.random.multinomial(1, preds, 1)    
    return np.argmax(probs)

#preds=[0.05, 0.1, 0.35, 0.5]
#np.random.multinomial(1, preds, 1)
#print([pred_indices(preds,0.05) for _ in range(10)])
#print([pred_indices(preds,1) for _ in range(10)])
#print([pred_indices(preds,50) for _ in range(10)])
#pred_indices(preds,1.2)

# generate a sequence of characters with a language model
def generate_seq_of_chars(model, num_unique_char, char2indices, indices2char, sentence, maxlen=40, n_chars=400, diversity=1.0):

    print('\n----- diversity:', diversity)

    generated = ''

    generated += sentence
    print('----- Generating with seed: "' + sentence + '"')
    print("\n--------------------------------------------------------\n")
    sys.stdout.write(generated)

    for _ in range(n_chars):
        x = np.zeros((1, maxlen, num_unique_char))
        for t, char in enumerate(sentence):
            x[0, t, char2indices[char]] = 1.

        preds = model.predict(x, verbose=0)[0]
        next_index = pred_indices(preds, diversity)
        pred_char = indices2char[next_index]

        generated += pred_char
        sentence = sentence[1:] + pred_char

        sys.stdout.write(pred_char)
        sys.stdout.flush()
    print("\n--------------------------------------------------------\n")
    
    return generated

start_index = randint(0, len(text) - maxlen - 1)
sentence = text[start_index: start_index + maxlen]

#diversity_table = [0.2, 0.7,1.2] 
diversity_table = [0.07, 0.1, 0.5, 0.7] 

#sentence = "mine".rjust(maxlen)

for diversity in diversity_table:
    generate_seq_of_chars(textgen_model_1, num_unique_char, char2indices, indices2char, sentence, maxlen, 300, diversity)    

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

# load trained model
textgen_model_2 = load_model('./model/pathfinder_wordtoken_model_200_epoch_noncuda_model.h5')

def sample_word(preds, temperature=1.0):
    if temperature <= 0:
        return np.argmax(preds)
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# generate a sequence of words from a language model
def generate_seq_of_words(model, tokenizer, seq_length, seed_text, n_words, temperature=0):
    result = list()
    in_text = seed_text
    # generate a fixed number of words
    for _ in range(n_words):
        # encode the text as integer
        #   [0] coz shape is (num_lines, num_tokens_per_line)
        #   and tehre is only one line right now
        #   also num_tokens is not a const, hence trunc will be required
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        # truncate sequences to a fixed lenght
        encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
        # predict probabilities for each word
        #yhat = model.predict_classes(encoded, verbose=0)
        yhat = sample_word(model.predict(encoded,verbose=0)[-1], temperature)
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

# select a seed text
seed_text = lines[randint(0,len(lines))]
print(textwrap.fill('%s' % (seed_text), 80) + '\n')

# generate new text
generated = generate_seq_of_words(textgen_model_2, tokenizer, seq_length, seed_text, 100, 0.2)
print(textwrap.fill('%s' % (generated), 80))

print()

#
# Word2vec pre-trained model
#

# load trained model
textgen_model_3 = load_model('./model/pathfinder_wordtoken_w2v_model_50_epoch_noncuda_model.h5')

# select a seed text
seed_text = lines[randint(0,len(lines))]
print(textwrap.fill('%s' % (seed_text), 80) + '\n')

# generate new text
generated = generate_seq_of_words(textgen_model_3, tokenizer, seq_length, seed_text, 100, 0.2)
print(textwrap.fill('%s' % (generated), 80))
