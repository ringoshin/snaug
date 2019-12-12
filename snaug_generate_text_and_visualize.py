#
# Load trained preferred model to generate new text and visualize it
#

from keras.models import load_model

from lib.data_common import load_doc, prepare_text_tokens
from lib.data_common import pathfinder_textfile, fixed_length_token_textfile
from lib.text_viz_common import init_text_viz_params, generate_and_visualize

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

#seed_text = input("Enter any text to generate a story, random' to automate or 'quit' to quit: ")
generate_and_visualize(lines, textgen_model, tokenizer, seq_length,
                        nlp, matcher, entity_names, entity_labels,
                        seed_text='random')