#
# Load trained preferred model to generate new text and visualize it
#

import string
import textwrap
import pickle
import sys
from random import randint

from keras.models import load_model

from lib.nlplstm_class import TFModelLSTMWord2vec 
from lib.data_common import (load_doc, save_doc, clean_doc, prepare_char_tokens,
                             build_token_lines, prepare_text_tokens, load_word2vec,
                             sample_predict, generate_seq_of_chars, generate_seq_of_words)

pathfinder_textfile = './data/textgen_pathfinder.txt'
fixed_length_token_textfile = './data/pathfinder_fixed-length_tokens.txt'

# 
# Loading text data that uses word tokenisation
#

# load fixed-length lines of tokens
doc = load_doc(fixed_length_token_textfile)
lines = doc.split('\n')
print('Total lines: %d' % len(lines))

# tokenize and separate into features and target
X, y, seq_length, vocab_size, tokenizer = prepare_text_tokens(lines)
#print(X.shape)

#
# Word2vec pre-trained model
#

# load trained model
textgen_model = load_model('./model/pathfinder_wordtoken_w2v_model_50_epoch_noncuda_model.h5')

# select a seed text
word_seed_text = lines[randint(0,len(lines))]
print('> using word tokenisation and pre-trained model')
print('> seed text:')
print(textwrap.fill('%s' % (word_seed_text), 80) + '\n')

# generate new text
temperature_table = [0, 1.1, 1.7]

# for temperature in temperature_table:
#     generated = generate_seq_of_words(textgen_model, tokenizer, seq_length, 
#                     word_seed_text, 75, temperature)
#     print(">> generated text (temperature: {})".format(temperature))
#     print(textwrap.fill('%s' % (generated), 80))
#     print()

generated = generate_seq_of_words(textgen_model, tokenizer, seq_length, 
                    word_seed_text, 75, 1.1)


import spacy
from spacy import displacy
from spacy.tokens import Span
from spacy.matcher import Matcher, PhraseMatcher

from collections import Counter
#import en_core_web_sm
#nlp = en_core_web_sm.load()

#!python -m spacy download en_core_web_lg
import en_core_web_lg
nlp = en_core_web_lg.load()

sentence_nlp = nlp(generated)

# print named entities in article
print([(word, word.ent_type_) for word in sentence_nlp if word.ent_type_])

# visualize named entities
displacy.render(sentence_nlp, style='ent')

god_list = ['Erastil', 'Aroden', 'Desna', 'Sarenrae']
race_list = ['Azlanti', 'Varisian', 'Thassilonian', 'Korvosan', 'Magnimarian']
org_list = ['Runelord', 'Runelords', 'Rockwhelp', 'Academae', 'Versade', 
            'Pathfinder', 'Society']
monster_list = ['wererat', 'wererats', 'werewolf', 'Yarg', 'goblin', 'goblins', 
                'orc', 'orcs', 'spider', 'spiders', 'minotaur', 'elemental', 
                'wolf', 'lycanthropes', 'lycanthrope', 'Cheh', 'Kavoos', 'Shala',
                'undead', 'ghouls']
person_list = ['Eberius', 'Tauranor', 'Xanderghul', 'Sheila', 'Heidmarch',
               'Hork', 'Shalesmash', 'Amelon', 'Birm', 'Grald', 'Hazelindra', 
               'Linna', 'Montrovale', 'Gorbic', 'Salmore', 'Mena', 'Zamola', 'Duso', 'Galino',
               'Piotr', 'Oja', 'Ollie', 'Lenstraand', 'Tibaldo', 'Russo', 'Joachim',
               'Aldemar', 'Lenstra', 'Jargie', 'Krzysztof', 'Szabo', 'Shabana', 'Neergah', 
               'Elliana', 'Silva', 'Savasha', 'Versade', 'Nindrik', 'Versade', 'Hobart', 'Deverin',
               'Kendra', 'Deverin', 'Gradon', 'Scarnetti', 'Zimandi', 'Kaddren', 'Cheiskaia', 'Nirodin',
               'Gradon', 'Scarnetti', 'Tauk', 'Yordan', 'Zorakov', 'Lucas', 'Gustavo', 'Kantaro',
               'Das', 'Korvut', 'Zanthus', 'Belor', 'Durn', 'Belor', 'Hemlock', 'Pavlina']
location_list = ['Golarion', 'Varisia', 'Lurkwood', 'Riddleport', 'Galduria', 
                 'Acadamae', 'Korvosa', 'Thassilon',  'Magnimar', 'Irespan', 
                 'Dockway', 'Sandpoint', 'Brinestump', 'Marsh', 'Soggy', 'River', 
                 'Windsong', 'Abbey', 'Rusty', 'Dragon', 'Necropolis']
spell_list = ['Burning', 'Hands']
long_name_list = god_list + race_list + org_list + monster_list + person_list + location_list + spell_list


god_labels = ['Erastil', 'Aroden', 'Desna', 'Sarenrae']
race_labels = ['Azlanti', 'Varisian', 'Thassilonian', 'Korvosan', 'Magnimarian']
org_labels = ['Runelord', 'Runelords', 'Aockwhelp', 'Academae', 'Versade',
             'Pathfinder Society']
monster_labels = ['wererat', 'wererats', 'werewolf', 'Yarg', 'goblin', 'goblins', 
                'orc', 'orcs', 'spider', 'spiders', 'minotaur', 'elemental', 
                'wolf', 'lycanthropes', 'lycanthrope', 'Cheh', 'Kavoos', 'Shala',
                'undead', 'ghouls']
person_labels = ['Eberius Tauranor', 'Xanderghul', 'Sheila Heidmarch',
#               'Hork Shalesmash', 'Amelon Birm', 'Grald', 'Hazelindra', 
               'Hork', 'Amelon Birm', 'Grald', 'Hazelindra', 
               'Linna Montrovale', 'Gorbic Salmore', 'Mena Zamola', 'Duso Galino',
               'Piotr Oja', 'Ollie Lenstraand', 'Tibaldo Russo', 'Joachim',
               'Aldemar Lenstra', 'Jargie', 'Krzysztof Szabo', 'Shabana Neergah', 
               'Elliana Silva', 'Savasha Versade', 'Nindrik Versade', 'Hobart Deverin',
               'Kendra Deverin', 'Gradon Scarnetti', 'Zimandi Kaddren', 'Cheiskaia Nirodin',
               'Gradon Scarnetti', 'Tauk', 'Yordan Zorakov', 'Lucas', 'Gustavo', 'Kantaro',
               'Das Korvut', 'Zanthus', 'Belor', 'Durn', 'Belor Hemlock', 'Pavlina']
location_labels = ['Golarion', 'Varisia', 'Lurkwood', 'Riddleport', 'Galduria', 
                 'Acadamae', 'Korvosa', 'Thassilon',  'Magnimar', 'Irespan', 
                 'Dockway', 'Sandpoint', 'Brinestump Marsh', 'Soggy River', 'Windsong Abbey', 
                  'Rusty Dragon', 'Necropolis']
spell_labels = ['Burning Hands']

def Entity_Type(word):
    word_lower = word
    word = word[0].upper() + word[1:]
    if word in god_list:
        return 'GOD', word
    elif word in race_list:
        return 'RACE', word
    elif word in org_list:
        return 'ORG', word
    elif word in monster_list:
        return 'MOB', word
    elif word_lower in monster_list:
        return 'MOB', word_lower 
    elif word in person_list:
        return 'PER', word
    elif word in location_list:
        return 'LOC', word
    elif word in spell_list:
        return 'SP', word
    else:
        return 'UNK', word_lower

    
full_generated_text = word_seed_text + generated
tmp_generated = generated
generated = full_generated_text


g_new = ''
g_god = []
g_mob = []
g_per = []
g_loc = []
g_race = []
g_org = []
g_sp = []
for g_word in generated.split():
    g_type, g_word = Entity_Type(g_word)
    if g_type=='GOD':
        g_god.append(g_word)
    elif g_type=='MOB':
        g_mob.append(g_word)
    elif g_type=='PER':
        g_per.append(g_word)
    elif g_type=='LOC':
        g_loc.append(g_word)
    elif g_type=='RACE':
        g_race.append(g_word)
    elif g_type=='ORG':
        g_org.append(g_word)
    elif g_type=='SP':
        g_sp.append(g_word)
    g_new  = ' '.join([g_new, g_word]) 
g_new = g_new.strip()


new_g_per=[]
if len(g_per)>1:
    x = iter(g_per)
    first_name = next(x)
    prev_fullname = ''
    for _ in range(len(g_per)-1):
        second_name = next(x)
        full_name = '{} {}'.format(first_name, second_name)
        if full_name in person_labels:
            new_g_per.append(full_name)
            prev_fullname = full_name
        else:
            if first_name in prev_fullname:
                prev_fullname = ''
            else:
                new_g_per.append(first_name)
        first_name = second_name
    g_per = new_g_per
    if first_name not in prev_fullname:
        g_per.append(first_name)

    
new_g_loc=[]
if len(g_loc)>1:
    x = iter(g_loc)
    first_name = next(x)
    prev_fullname = ''
    for _ in range(len(g_loc)-1):
        second_name = next(x)
        full_name = '{} {}'.format(first_name, second_name)
        if full_name in location_labels:
            new_g_loc.append(full_name)
            prev_fullname = full_name
        else:
            if first_name in prev_fullname:
                prev_fullname = ''
            else:
                new_g_loc.append(first_name)
        first_name = second_name
    g_loc = new_g_loc
    if first_name not in prev_fullname:
        g_loc.append(first_name)


new_g_sp=[]
if len(g_sp)>1:
    x = iter(g_sp)
    first_name = next(x)
    prev_fullname = ''
    for _ in range(len(g_sp)-1):
        second_name = next(x)
        full_name = '{} {}'.format(first_name, second_name)
        if full_name in spell_labels:
            new_g_sp.append(full_name)
            prev_fullname = full_name
        else:
            if first_name in prev_fullname:
                prev_fullname = ''
            else:
                new_g_sp.append(first_name)
        first_name = second_name
    g_sp = new_g_sp
    if first_name not in prev_fullname:
        g_sp.append(first_name)


new_g_org=[]
if len(g_org)>1:
    x = iter(g_org)
    first_name = next(x)
    prev_fullname = ''
    for _ in range(len(g_org)-1):
        second_name = next(x)
        full_name = '{} {}'.format(first_name, second_name)
        if full_name in org_labels:
            new_g_org.append(full_name)
            prev_fullname = full_name
        else:
            if first_name in prev_fullname:
                prev_fullname = ''
            else:
                new_g_org.append(first_name)
        first_name = second_name
    g_org = new_g_org
    if first_name not in prev_fullname:
        g_org.append(first_name)


def List_Notables():
    g_all = g_god + g_mob + g_per + g_sp + g_loc + g_race + g_org
    if g_all:
        print('Notable(s) found:')
        if g_god:
            print("   God(s)     : {}".format(', '.join(g_god)))
        if g_mob:
            print("   Monster(s) : {}".format(', '.join(g_mob)))
        if g_per:
            print("   Person(s)  : {}".format(', '.join(g_per)))
        if g_sp:
            print("   Spell(s)   : {}".format(', '.join(g_sp)))
        if g_loc:
            print("   Location(s): {}".format(', '.join(g_loc)))
        if g_race:
            print("   Race(s)    : {}".format(', '.join(g_race)))
        if g_org:
            print("   Organisation(s): {}".format(', '.join(g_org)))
    else:
        print(">>> No notables found.")

    
god_patterns = [nlp(text) for text in god_labels]
mob_patterns = [nlp(text) for text in monster_labels]
per_patterns = list(nlp.pipe(person_labels))
loc_patterns = list(nlp.pipe(location_labels))
race_patterns = [nlp(text) for text in race_labels]
org_patterns = [nlp(text) for text in org_labels]
sp_patterns = [nlp(text) for text in spell_labels]

matcher = PhraseMatcher(nlp.vocab)
matcher.add('GOD', None, *god_patterns)
matcher.add('MOB', None, *mob_patterns)
matcher.add('PER', None, *per_patterns)
matcher.add('LOC', None, *loc_patterns)
matcher.add('RACE', None, *race_patterns)
matcher.add('ORG', None, *org_patterns)
matcher.add('SP', None, *sp_patterns)


doc = nlp(g_new)
matches = matcher(doc)
spans = []
for match_id, start, end in matches:
    rule_id = nlp.vocab.strings[match_id]  # get the unicode ID, i.e. 'COLOR'
    span = doc[start : end]  # get the matched slice of the doc
    print(rule_id, span.text)
    spans.append(Span(doc, start, end, label=rule_id))
    doc.ents = spans


print()
print()
print('You: {}'.format('Tell me about a tome penned by an apprentice.'))
print('-'*95)
options = {"ents": ['GOD','MOB','PER','LOC','RACE','ORG','SP'],
           "colors": {'GOD':'#f2865e','MOB':'#58f549','PER':'#aef5ef',
                      'LOC':'pink','RACE':'#edcb45','ORG':'#d88fff', 'SP':'pink'}}
print('Snaug_bot:') 
# displacy.render(doc, style='ent', jupyter=True, options=options)
displacy.render(doc, style='ent', options=options)
print()
List_Notables()
print('-'*95)
print()
print()
