#! /usr/bin/python3

import sys
from os import listdir
from xml.dom.minidom import parse

from deptree import *
#import patterns


## ------------------- 
## -- Convert a pair of drugs and their context in a feature vector

def extract_features(tree, entities, e1, e2) :
   feats = set()

   # get head token for each gold entity
   tkE1 = tree.get_fragment_head(entities[e1]['start'],entities[e1]['end'])
   tkE2 = tree.get_fragment_head(entities[e2]['start'],entities[e2]['end'])

   if tkE1 is not None and tkE2 is not None:
      # features for tokens in between E1 and E2
      #for tk in range(tkE1+1, tkE2) :
      tk=tkE1+1
      try:
        while (tree.is_stopword(tk)):
          tk += 1
      except:
        return set()
      word  = tree.get_word(tk)
      lemma = tree.get_lemma(tk).lower()
      tag = tree.get_tag(tk)
      feats.add("lib=" + lemma)
      feats.add("wib=" + word)
      feats.add("lpib=" + lemma + "_" + tag)
      
      eib = False
      for tk in range(tkE1+1, tkE2) :
         if tree.is_entity(tk, entities):
            eib = True 
      
	  # feature indicating the presence of an entity in between E1 and E2
      feats.add('eib='+ str(eib))

      # features about paths in the tree
      lcs = tree.get_LCS(tkE1,tkE2)
      
      path1 = tree.get_up_path(tkE1,lcs)
      path1 = "<".join([tree.get_lemma(x)+"_"+tree.get_rel(x) for x in path1])
      feats.add("path1="+path1)

      path2 = tree.get_down_path(lcs,tkE2)
      path2 = ">".join([tree.get_lemma(x)+"_"+tree.get_rel(x) for x in path2])
      feats.add("path2="+path2)

      path = path1+"<"+tree.get_lemma(lcs)+"_"+tree.get_rel(lcs)+">"+path2      
      feats.add("path="+path)
      
   return feats

def extract_ngram_features(tree, entities, e1, e2, n=2):
    feats = set()

    # get head token for each gold entity
    tkE1 = tree.get_fragment_head(entities[e1]['start'], entities[e1]['end'])
    tkE2 = tree.get_fragment_head(entities[e2]['start'], entities[e2]['end'])

    if tkE1 is not None and tkE2 is not None:
        # tokens between E1 and E2
        tokens_between = []
        for tk in range(tkE1 + 1, tkE2):
            try:
                if not tree.is_stopword(tk):
                    word = tree.get_word(tk)
                    lemma = tree.get_lemma(tk).lower()
                    tokens_between.append((lemma, word))
            except:
                return set()

        # n-grams extraction
        for i in range(len(tokens_between) - n + 1):
            ngram = " ".join([tokens_between[j][0] for j in range(i, i + n)])
            feats.add("ngram=" + ngram)

        # features about paths in the tree
        lcs = tree.get_LCS(tkE1, tkE2)

        path1 = tree.get_up_path(tkE1, lcs)
        path1 = "<".join([tree.get_lemma(x) + "_" + tree.get_rel(x) for x in path1])
        feats.add("path1=" + path1)

        path2 = tree.get_down_path(lcs, tkE2)
        path2 = ">".join([tree.get_lemma(x) + "_" + tree.get_rel(x) for x in path2])
        feats.add("path2=" + path2)

        path = path1 + "<" + tree.get_lemma(lcs) + "_" + tree.get_rel(lcs) + ">" + path2
        feats.add("path=" + path)


    return feats

def extract_features_distance(tree, entities, e1, e2):
    feats = set()

    # get head token for each gold entity
    tkE1 = tree.get_fragment_head(entities[e1]['start'], entities[e1]['end'])
    tkE2 = tree.get_fragment_head(entities[e2]['start'], entities[e2]['end'])

    if tkE1 is not None and tkE2 is not None:
        # features for tokens in between E1 and E2
        tk = tkE1 + 1
        try:
            while tree.is_stopword(tk):
                tk += 1
        except:
            return set()

        word = tree.get_word(tk)
        lemma = tree.get_lemma(tk).lower()
        tag = tree.get_tag(tk)
        feats.add("lib=" + lemma)
        feats.add("wib=" + word)
        feats.add("lpib=" + lemma + "_" + tag)

        eib = False
        for tk in range(tkE1 + 1, tkE2):
            if tree.is_entity(tk, entities):
                eib = True

        # feature indicating the presence of an entity in between E1 and E2
        feats.add('eib=' + str(eib))

        # features about paths in the tree
        lcs = tree.get_LCS(tkE1, tkE2)

        path1 = tree.get_up_path(tkE1, lcs)
        path1 = "<".join([tree.get_lemma(x) + "_" + tree.get_rel(x) for x in path1])
        feats.add("path1=" + path1)

        path2 = tree.get_down_path(lcs, tkE2)
        path2 = ">".join([tree.get_lemma(x) + "_" + tree.get_rel(x) for x in path2])
        feats.add("path2=" + path2)

        path = path1 + "<" + tree.get_lemma(lcs) + "_" + tree.get_rel(lcs) + ">" + path2
        feats.add("path=" + path)

        # Distance between Entities feature
        distance = abs(tkE2 - tkE1)
        feats.add("distance=" + str(distance))

        # Entity Neighbors feature
        neighbor1 = tree.get_word(tree.get_parent(tkE1)) if tree.get_parent(tkE1) is not None else 'None'
        neighbor2 = tree.get_word(tree.get_parent(tkE2)) if tree.get_parent(tkE2) is not None else 'None'
        feats.add("neighbor1=" + neighbor1)
        feats.add("neighbor2=" + neighbor2)

    return feats

def extract_features_neighbors(tree, entities, e1, e2):
    feats = set()

    # get head token for each gold entity
    tkE1 = tree.get_fragment_head(entities[e1]['start'], entities[e1]['end'])
    tkE2 = tree.get_fragment_head(entities[e2]['start'], entities[e2]['end'])

    if tkE1 is not None and tkE2 is not None:
        # features for tokens in between E1 and E2
        tk = tkE1 + 1
        try:
            while tree.is_stopword(tk):
                tk += 1
        except:
            return set()

        word = tree.get_word(tk)
        lemma = tree.get_lemma(tk).lower()
        tag = tree.get_tag(tk)
        feats.add("lib=" + lemma)
        feats.add("wib=" + word)
        feats.add("lpib=" + lemma + "_" + tag)

        eib = False
        for tk in range(tkE1 + 1, tkE2):
            if tree.is_entity(tk, entities):
                eib = True

        # feature indicating the presence of an entity in between E1 and E2
        feats.add('eib=' + str(eib))

        # features about paths in the tree
        lcs = tree.get_LCS(tkE1, tkE2)

        path1 = tree.get_up_path(tkE1, lcs)
        path1 = "<".join([tree.get_lemma(x) + "_" + tree.get_rel(x) for x in path1])
        feats.add("path1=" + path1)

        path2 = tree.get_down_path(lcs, tkE2)
        path2 = ">".join([tree.get_lemma(x) + "_" + tree.get_rel(x) for x in path2])
        feats.add("path2=" + path2)

        path = path1 + "<" + tree.get_lemma(lcs) + "_" + tree.get_rel(lcs) + ">" + path2
        feats.add("path=" + path)

        # Entity Neighbors feature
        neighbor1 = tree.get_word(tree.get_parent(tkE1)) if tree.get_parent(tkE1) is not None else 'None'
        neighbor2 = tree.get_word(tree.get_parent(tkE2)) if tree.get_parent(tkE2) is not None else 'None'
        feats.add("neighbor1=" + neighbor1)
        feats.add("neighbor2=" + neighbor2)

        # Determine prepositions and verbs based on PoS tag
        prepositions = [tree.get_word(tk).lower() for tk in range(tkE1 + 1, tkE2) if tree.get_tag(tk).startswith('IN')]
        verbs = [tree.get_word(tk).lower() for tk in range(tkE1 + 1, tkE2) if tree.get_tag(tk).startswith('V')]
        
        feats.add("prepositions=" + "_".join(prepositions))
        feats.add("verbs=" + "_".join(verbs))
        
        # Add entity types
        e1_type = entities[e1].get('type', '<none>')
        e2_type = entities[e2].get('type', '<none>')
        feats.add("e1_type=" + e1_type)
        feats.add("e2_type=" + e2_type)

        # Add next entity types for tkE1 and tkE2
        for i, entity in enumerate([tkE1, tkE2]):
            next_entity_types = []
            for tk in range(entity + 1, min(tree.get_n_nodes(), entity + 3)):  # up to two nodes ahead
                if tk in entities and tree.is_entity(tk, entities):
                    next_entity_type = entities[tk].get('type', '<none>')
                    next_entity_types.append(next_entity_type)
            feats.add("next_entity{}_types=".format(i + 1) + "_".join(next_entity_types))

    return feats

def extract_features_prep(tree, entities, e1, e2):
    feats = set()
    # get head token for each gold entity
    tkE1 = tree.get_fragment_head(entities[e1]['start'], entities[e1]['end'])
    tkE2 = tree.get_fragment_head(entities[e2]['start'], entities[e2]['end'])
    if tkE1 is not None and tkE2 is not None:
        # features for tokens in between E1 and E2
        for tk in range(tkE1 + 1, tkE2):
            try:
                while tree.is_stopword(tk):
                    tk += 1
            except:
                return set()
            word = tree.get_word(tk)
            lemma = tree.get_lemma(tk).lower()
            tag = tree.get_tag(tk)
            feats.add("lib=" + lemma)
            feats.add("wib=" + word)
            feats.add("lpib=" + lemma + "_" + tag)
        
        # Determine prepositions based on PoS tag
        prepositions = [tree.get_word(tk).lower() for tk in range(tkE1 + 1, tkE2) if tree.get_tag(tk).startswith('IN')]
        feats.add("prepositions=" + "_".join(prepositions))
    return feats

def extract_features_verb(tree, entities, e1, e2):
    feats = set()
    # get head token for each gold entity
    tkE1 = tree.get_fragment_head(entities[e1]['start'], entities[e1]['end'])
    tkE2 = tree.get_fragment_head(entities[e2]['start'], entities[e2]['end'])
    if tkE1 is not None and tkE2 is not None:
        # features for tokens in between E1 and E2
        for tk in range(tkE1 + 1, tkE2):
            try:
                while tree.is_stopword(tk):
                    tk += 1
            except:
                return set()
            word = tree.get_word(tk)
            lemma = tree.get_lemma(tk).lower()
            tag = tree.get_tag(tk)
            feats.add("lib=" + lemma)
            feats.add("wib=" + word)
            feats.add("lpib=" + lemma + "_" + tag)
        
        # Determine verbs based on PoS tag
        verbs = [tree.get_word(tk).lower() for tk in range(tkE1 + 1, tkE2) if tree.get_tag(tk).startswith('V')]
        feats.add("verbs=" + "_".join(verbs))
    return feats

def entity_types(tree, entities, e1, e2):
    feats = set()
    # get head token for each gold entity
    tkE1 = tree.get_fragment_head(entities[e1]['start'], entities[e1]['end'])
    tkE2 = tree.get_fragment_head(entities[e2]['start'], entities[e2]['end'])
    if tkE1 is not None and tkE2 is not None:
        # features for tokens in between E1 and E2
        for tk in range(tkE1 + 1, tkE2):
            try:
                while tree.is_stopword(tk):
                    tk += 1
            except:
                return set()
            word = tree.get_word(tk)
            lemma = tree.get_lemma(tk).lower()
            tag = tree.get_tag(tk)
            feats.add("lib=" + lemma)
            feats.add("wib=" + word)
            feats.add("lpib=" + lemma + "_" + tag)
        

        prepositions = [tree.get_word(tk).lower() for tk in range(tkE1 + 1, tkE2) if tree.get_tag(tk).startswith('IN')]
        verbs = [tree.get_word(tk).lower() for tk in range(tkE1 + 1, tkE2) if tree.get_tag(tk).startswith('V')]
        
        feats.add("prepositions=" + "_".join(prepositions))
        feats.add("verbs=" + "_".join(verbs))
        
        # Add entity types
        e1_type = entities[e1].get('type', '<none>')
        e2_type = entities[e2].get('type', '<none>')
        feats.add("e1_type=" + e1_type)
        feats.add("e2_type=" + e2_type)
    return feats

def entity_types_next(tree, entities, e1, e2):
    feats = set()
    # get head token for each gold entity
    tkE1 = tree.get_fragment_head(entities[e1]['start'], entities[e1]['end'])
    tkE2 = tree.get_fragment_head(entities[e2]['start'], entities[e2]['end'])
    
    if tkE1 is not None and tkE2 is not None:
        # features for tokens in between E1 and E2
        for tk in range(tkE1 + 1, tkE2):
            try:
                while tree.is_stopword(tk):
                    tk += 1
            except:
                return set()
            word = tree.get_word(tk)
            lemma = tree.get_lemma(tk).lower()
            tag = tree.get_tag(tk)
            feats.add("lib=" + lemma)
            feats.add("wib=" + word)
            feats.add("lpib=" + lemma + "_" + tag)
        
        # Determine prepositions and verbs based on PoS tag
        prepositions = [tree.get_word(tk).lower() for tk in range(tkE1 + 1, tkE2) if tree.get_tag(tk).startswith('IN')]
        verbs = [tree.get_word(tk).lower() for tk in range(tkE1 + 1, tkE2) if tree.get_tag(tk).startswith('V')]
        
        feats.add("prepositions=" + "_".join(prepositions))
        feats.add("verbs=" + "_".join(verbs))
        
  
        e1_type = entities[e1].get('type', '<none>')
        e2_type = entities[e2].get('type', '<none>')
        feats.add("e1_type=" + e1_type)
        feats.add("e2_type=" + e2_type)
        

        for i, entity in enumerate([tkE1, tkE2]):
            next_entity_types = []
            for tk in range(entity + 1, min(tree.get_n_nodes(), entity + 3)):
                if tk in entities and tree.is_entity(tk, entities):
                    next_entity_type = entities[tk].get('type', '<none>')
                    next_entity_types.append(next_entity_type)
            feats.add("next_entity{}_types=".format(i + 1) + "_".join(next_entity_types))
    
    return feats

######NEW FUNCTION INCLUDES ADVERBS FOR PREVIOUS
def entity_types_next(tree, entities, e1, e2):
    feats = set()
    # get head token for each gold entity
    tkE1 = tree.get_fragment_head(entities[e1]['start'], entities[e1]['end'])
    tkE2 = tree.get_fragment_head(entities[e2]['start'], entities[e2]['end'])
    
    if tkE1 is not None and tkE2 is not None:
        # features for tokens in between E1 and E2
        for tk in range(tkE1 + 1, tkE2):
            try:
                while tree.is_stopword(tk):
                    tk += 1
            except:
                return set()
            word = tree.get_word(tk)
            lemma = tree.get_lemma(tk).lower()
            tag = tree.get_tag(tk)
            feats.add("lib=" + lemma)
            feats.add("wib=" + word)
            feats.add("lpib=" + lemma + "_" + tag)
        
        prepositions = [tree.get_word(tk).lower() for tk in range(tkE1 + 1, tkE2) if tree.get_tag(tk).startswith('IN')]
        verbs = [tree.get_word(tk).lower() for tk in range(tkE1 + 1, tkE2) if tree.get_tag(tk).startswith('V')]
        adverbs = [tree.get_word(tk).lower() for tk in range(tkE1 + 1, tkE2) if tree.get_tag(tk).startswith('RB')]
        
        feats.add("prepositions=" + "_".join(prepositions))
        feats.add("verbs=" + "_".join(verbs))
        feats.add("adverbs=" + "_".join(adverbs))
        e1_type = entities[e1].get('type', '<none>')
        e2_type = entities[e2].get('type', '<none>')
        feats.add("e1_type=" + e1_type)
        feats.add("e2_type=" + e2_type)
        

        for i, entity in enumerate([tkE1, tkE2]):
            next_entity_types = []
            for tk in range(entity + 1, min(tree.get_n_nodes(), entity + 3)):
                if tk in entities and tree.is_entity(tk, entities):
                    next_entity_type = entities[tk].get('type', '<none>')
                    next_entity_types.append(next_entity_type)
            feats.add("next_entity{}_types=".format(i + 1) + "_".join(next_entity_types))
    
    return feats
## --------- MAIN PROGRAM ----------- 
## --
## -- Usage:  extract_features targetdir
## --
## -- Extracts feature vectors for DD interaction pairs from all XML files in target-dir
## --

# directory with files to process
datadir = sys.argv[1]

# process each file in directory
for f in listdir(datadir) :

    # parse XML file, obtaining a DOM tree
    tree = parse(datadir+"/"+f)

    # process each sentence in the file
    sentences = tree.getElementsByTagName("sentence")
    for s in sentences :
        sid = s.attributes["id"].value   # get sentence id
        stext = s.attributes["text"].value   # get sentence text
        # load sentence entities
        entities = {}
        ents = s.getElementsByTagName("entity")
        for e in ents :
           id = e.attributes["id"].value
           offs = e.attributes["charOffset"].value.split("-")           
           entities[id] = {'start': int(offs[0]), 'end': int(offs[-1])}

        # there are no entity pairs, skip sentence
        if len(entities) <= 1 : continue

        # analyze sentence
        analysis = deptree(stext)

        # for each pair in the sentence, decide whether it is DDI and its type
        pairs = s.getElementsByTagName("pair")
        for p in pairs:
            # ground truth
            ddi = p.attributes["ddi"].value
            if (ddi=="true") : dditype = p.attributes["type"].value
            else : dditype = "null"
            # target entities
            id_e1 = p.attributes["e1"].value
            id_e2 = p.attributes["e2"].value
            # feature extraction

            feats = extract_features_neighbors(analysis,entities,id_e1,id_e2) 
            # resulting vector
            if len(feats) != 0:
              print(sid, id_e1, id_e2, dditype, "\t".join(feats), sep="\t")

