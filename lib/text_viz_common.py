#
# Visualize generated text using spaCy library
#

from collections import defaultdict, Counter

from spacy import displacy
from spacy.tokens import Span
from spacy.matcher import Matcher, PhraseMatcher


def get_Pathfinder_entities():
    """
    Quick and dirty list of recognized entity actual names and labels used
    in imported story text files
    """
    god_name_list = ['Erastil', 'Aroden', 'Desna', 'Sarenrae']
    race_name_list = ['Azlanti', 'Varisian', 'Thassilonian', 'Korvosan', 'Magnimarian']
    org_name_list = ['Runelord', 'Runelords', 'Rockwhelp', 'Academae', 
                    'Pathfinder', 'Society']
    mob_name_list = ['wererat', 'wererats', 'werewolf', 'Yarg', 'goblin', 'goblins', 
                    'orc', 'orcs', 'spider', 'spiders', 'minotaur', 'elemental', 
                    'wolf', 'lycanthropes', 'lycanthrope', 'Cheh', 'Kavoos', 'Shala',
                    'undead', 'ghouls']
    per_name_list = ['Eberius', 'Tauranor', 'Xanderghul', 'Sheila', 'Heidmarch',
                'Hork', 'Shalesmash', 'Amelon', 'Birm', 'Grald', 'Hazelindra', 
                'Linna', 'Montrovale', 'Gorbic', 'Salmore', 'Mena', 'Zamola', 'Duso', 'Galino',
                'Piotr', 'Oja', 'Ollie', 'Lenstraand', 'Tibaldo', 'Russo', 'Joachim',
                'Aldemar', 'Lenstra', 'Jargie', 'Krzysztof', 'Szabo', 'Shabana', 'Neergah', 
                'Elliana', 'Silva', 'Savasha', 'Versade', 'Nindrik', 'Versade', 'Hobart', 'Deverin',
                'Kendra', 'Deverin', 'Gradon', 'Scarnetti', 'Zimandi', 'Kaddren', 'Cheiskaia', 'Nirodin',
                'Gradon', 'Scarnetti', 'Tauk', 'Yordan', 'Zorakov', 'Lucas', 'Gustavo', 'Kantaro',
                'Das', 'Korvut', 'Zanthus', 'Belor', 'Durn', 'Belor', 'Hemlock', 'Pavlina']
    loc_name_list = ['Golarion', 'Varisia', 'Lurkwood', 'Riddleport', 'Galduria', 
                    'Acadamae', 'Korvosa', 'Thassilon',  'Magnimar', 'Irespan', 
                    'Dockway', 'Sandpoint', 'Brinestump', 'Marsh', 'Soggy', 'River', 
                    'Windsong', 'Abbey', 'Rusty', 'Dragon', 'Necropolis']
    sp_name_list = ['Burning', 'Hands']
    #long_name_list = god_name_list + race_name_list + org_name_list + mob_name_list + per_name_list + loc_name_list + sp_name_list

    entity_names = {'GOD': god_name_list, 'RACE': race_name_list, 'ORG': org_name_list,
                    'MOB': mob_name_list, 'PER': per_name_list, 'LOC': loc_name_list,
                    'SP': sp_name_list}

    god_labels = ['Erastil', 'Aroden', 'Desna', 'Sarenrae']
    race_labels = ['Azlanti', 'Varisian', 'Thassilonian', 'Korvosan', 'Magnimarian']
    org_labels = ['Runelord', 'Runelords', 'Aockwhelp', 'Academae', 
                'Pathfinder Society']
    mob_labels = ['wererat', 'wererats', 'werewolf', 'Yarg', 'goblin', 'goblins', 
                'orc', 'orcs', 'spider', 'spiders', 'minotaur', 'elemental', 
                'wolf', 'lycanthropes', 'lycanthrope', 'Cheh', 'Kavoos', 'Shala',
                'undead', 'ghouls']
    per_labels = ['Eberius Tauranor', 'Xanderghul', 'Sheila Heidmarch',
    #             'Hork Shalesmash', 'Amelon Birm', 'Grald', 'Hazelindra', 
                'Hork', 'Amelon Birm', 'Grald', 'Hazelindra', 
                'Linna Montrovale', 'Gorbic Salmore', 'Mena Zamola', 'Duso Galino',
                'Piotr Oja', 'Ollie Lenstraand', 'Tibaldo Russo', 'Joachim',
                'Aldemar Lenstra', 'Jargie', 'Krzysztof Szabo', 'Shabana Neergah', 
                'Elliana Silva', 'Savasha Versade', 'Nindrik Versade', 'Hobart Deverin',
                'Kendra Deverin', 'Gradon Scarnetti', 'Zimandi Kaddren', 'Cheiskaia Nirodin',
                'Gradon Scarnetti', 'Tauk', 'Yordan Zorakov', 'Lucas', 'Gustavo', 'Kantaro',
                'Das Korvut', 'Zanthus', 'Belor', 'Durn', 'Belor Hemlock', 'Pavlina']
    loc_labels = ['Golarion', 'Varisia', 'Lurkwood', 'Riddleport', 'Galduria', 
                'Acadamae', 'Korvosa', 'Thassilon',  'Magnimar', 'Irespan', 
                'Dockway', 'Sandpoint', 'Brinestump Marsh', 'Soggy River', 'Windsong Abbey', 
                'Rusty Dragon', 'Necropolis']
    sp_lables = ['Burning Hands']

    entity_labels = {'GOD': god_labels, 'RACE': race_labels, 'ORG': org_labels, 'MOB': mob_labels,
                    'PER': per_labels, 'LOC': loc_labels, 'SP': sp_lables}

    return entity_names, entity_labels


def get_matcher(nlp, entity_labels):
    """
    Generate new Spacy matchers for customised entities
    """

    matcher = PhraseMatcher(nlp.vocab)
    for entity, label_list in entity_labels.items():
        entity_patterns = [nlp(text) for text in label_list]
        matcher.add(entity, None, *entity_patterns)

    # god_labels = entity_labels['GOD']
    # mob_labels = entity_labels['MOB']
    # per_labels = entity_labels['PER']
    # loc_labels = entity_labels['LOC']
    # race_labels = entity_labels['RACE']
    # org_labels = entity_labels['ORG']
    # sp_labels = entity_labels['SP']

    # god_patterns = [nlp(text) for text in god_labels]
    # mob_patterns = [nlp(text) for text in mob_labels]
    # per_patterns = list(nlp.pipe(per_labels))
    # loc_patterns = list(nlp.pipe(loc_labels))
    # race_patterns = [nlp(text) for text in race_labels]
    # org_patterns = [nlp(text) for text in org_labels]
    # sp_patterns = [nlp(text) for text in sp_labels]

    # matcher = PhraseMatcher(nlp.vocab)
    # matcher.add('GOD', None, *god_patterns)
    # matcher.add('MOB', None, *mob_patterns)
    # matcher.add('PER', None, *per_patterns)
    # matcher.add('LOC', None, *loc_patterns)
    # matcher.add('RACE', None, *race_patterns)
    # matcher.add('ORG', None, *org_patterns)
    # matcher.add('SP', None, *sp_patterns)

    return matcher


def init_text_viz_params():
    """
    Initialize parameters before utilizing spaCy library to visualize
    generated text data
    """
    #import en_core_web_sm
    #nlp = en_core_web_sm.load()

    #!python -m spacy download en_core_web_lg
    import en_core_web_lg
    nlp = en_core_web_lg.load()

    #sentence_nlp = nlp(generated)

    # print named entities in article
    #print([(word, word.ent_type_) for word in sentence_nlp if word.ent_type_])

    # visualize named entities
    #displacy.render(sentence_nlp, style='ent')

    entity_names, entity_labels = get_Pathfinder_entities()
    matcher = get_matcher(nlp, entity_labels)

    return nlp, matcher, entity_names, entity_labels


def identify_entity_type(word, entity_names):
    """
    Identity whether the word is an entity and return the entity type
    """
    word_lower = word
    word = word[0].upper() + word[1:]
    for entity, name_list  in entity_names.items():
        if entity=='MOB':
            if word_lower in name_list:
                return 'MOB', word_lower
            elif word in name_list:
                return 'MOB', word
        elif word in name_list:
            return entity, word
    return 'UNK', word_lower


def find_entity(text, entity_names):
    """
    Create list of entities found in provided text
    """
    new_text = ''
    found_entity = defaultdict(list)

    for each_word in text.split():
        entity_type, entity_name = identify_entity_type(each_word, entity_names)
        if entity_type!='UNK' and entity_name not in found_entity[entity_type]:
            found_entity[entity_type].append(entity_name)

        new_text  = ' '.join([new_text, entity_name]) 
    return found_entity, new_text.strip()


def modify_for_full_names(found_entity, entity_labels):
    """
    Modify entity labels for entity with names that has more than one word
    """

    for entity, name_list  in found_entity.items():
        new_name_list = []
        if len(name_list)>1:
            x = iter(name_list)
            first_name = next(x)
            prev_fullname = ''
            for _ in range(len(name_list)-1):
                second_name = next(x)
                full_name = '{} {}'.format(first_name, second_name)
                if full_name in entity_labels[entity]:
                    new_name_list.append(full_name)
                    prev_fullname = full_name
                else:
                    if first_name in prev_fullname:
                        prev_fullname = ''
                    else:
                        new_name_list.append(first_name)
                first_name = second_name
            found_entity[entity] = new_name_list
            if first_name not in prev_fullname:
                found_entity[entity].append(first_name)
    return found_entity


def list_notables(found_entity):
    """
    Display in a list identified named entities grouped by entity type
    """
    notables_found = False
    print('Notables:')
    for entity, name_list in found_entity.items():
        if name_list:
            notables_found = True
            print("   {}(s)\t: {}".format(entity, ', '.join(name_list)))
    if not notables_found:
        print(">>> None found.")


def visualize_gen_text(generated, nlp, matcher, entity_names, entity_labels, using_notebook=False):
    """
    Visualize customized named entities in generated text using spaCy library
    """
    found_entity, revised_text = find_entity(generated, entity_names)
    found_entity = modify_for_full_names(found_entity, entity_labels)

    doc = nlp(revised_text)
    matches = matcher(doc)
    spans = []
    for match_id, start, end in matches:
        rule_id = nlp.vocab.strings[match_id]  # get the unicode ID, i.e. 'COLOR'
        span = doc[start : end]                # get the matched slice of the doc
        # print(rule_id, span.text)
        spans.append(Span(doc, start, end, label=rule_id))
        doc.ents = spans

    print()
    print('-'*95)
    options = {"ents": ['GOD','MOB','PER','LOC','RACE','ORG','SP'],
            "colors": {'GOD':'#f2865e','MOB':'#58f549','PER':'#aef5ef',
                        'LOC':'pink','RACE':'#edcb45','ORG':'#d88fff', 'SP':'pink'}}
    print('Snaug_bot:') 
    
    if using_notebook:
        displacy.render(doc, style='ent', jupyter=True, options=options)
    else:
        displacy.render(doc, style='ent', options=options)

    print()
    list_notables(found_entity)
    print('-'*95)
    print()
    print()


if __name__ == '__main__':
    pass