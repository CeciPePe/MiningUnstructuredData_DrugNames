#! /usr/bin/python3

import sys
import re
from os import listdir

from xml.dom.minidom import parse
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec

   
## --------- tokenize sentence ----------- 
## -- Tokenize sentence, returning tokens and span offsets

def tokenize(txt):
    offset = 0
    tks = []
    ## word_tokenize splits words, taking into account punctuations, numbers, etc.
    for t in word_tokenize(txt):
        ## keep track of the position where each token should appear, and
        ## store that information with the token
        offset = txt.find(t, offset)
        tks.append((t, offset, offset+len(t)-1))
        offset += len(t)

    ## tks is a list of triples (word,start,end)
    return tks


## --------- get tag ----------- 
##  Find out whether given token is marked as part of an entity in the XML

def get_tag(token, spans) :
   (form,start,end) = token
   for (spanS,spanE,spanT) in spans :
      if start==spanS and end<=spanE : return "B-"+spanT
      elif start>=spanS and end<=spanE : return "I-"+spanT

   return "O"
 
## --------- Feature extractor ----------- 
## -- Extract features for each token in given sentence

def extract_features(tokens) :
   #Contextual features
   # for each token, generate list of features and add it to the result
   result = []
   for k in range(len(tokens)):
      token_features = []
      t = tokens[k][0]

      # Basic features for the current token
      token_features.append("form=" + t)
      token_features.append("suf3=" + t[-3:])

      # Features for previous tokens within the window size
      for i in range(1, 1 + 1):
         if k - i >= 0:
               t_prev = tokens[k - i][0]
               token_features.append(f"formPrev{i}=" + t_prev)
               token_features.append(f"suf3Prev{i}=" + t_prev[-3:])
         else:
               token_features.append(f"BoS{i}")

      # Features for next tokens within the window size
      for i in range(1, 1 + 1):
         if k + i < len(tokens):
               t_next = tokens[k + i][0]
               token_features.append(f"formNext{i}=" + t_next)
               token_features.append(f"suf3Next{i}=" + t_next[-3:])
         else:
               token_features.append(f"EoS{i}")

      result.append(token_features)

   return result

   #Capitalization and brand names
   #drug_brand_names = []
   #with open('/Users/bielcave/Documents/MDS/4th_Semester/MUD/lab_resources/DDI/resources/DrugBank.txt', 'r') as file:
   #   for line in file:
   #      parts = line.strip().split('|')
   #      if parts[1] == 'brand':
   #            drug_brand_names.append(parts[0])
#
   #result = []
   #for k in range(0, len(tokens)):
   #   tokenFeatures = []
   #   t = tokens[k][0]
#
   #   tokenFeatures.append("form=" + t)
   #   tokenFeatures.append("suf3=" + t[-3:])
#
   #   if k > 0:
   #      tPrev = tokens[k - 1][0]
   #      tokenFeatures.append("formPrev=" + tPrev)
   #      tokenFeatures.append("suf3Prev=" + tPrev[-3:])
   #   else:
   #      tokenFeatures.append("BoS")
#
   #   if k < len(tokens) - 1:
   #      tNext = tokens[k + 1][0]
   #      tokenFeatures.append("formNext=" + tNext)
   #      tokenFeatures.append("suf3Next=" + tNext[-3:])
   #   else:
   #      tokenFeatures.append("EoS")
#
   #   # Check for capitalization
   #   if t.isupper():
   #      tokenFeatures.append("capitalization=true")
   #   else:
   #      tokenFeatures.append("capitalization=false")
#
   #   # Check for drug brand names
   #   if t in drug_brand_names:
   #      tokenFeatures.append("drugBrandName=true")
   #   else:
   #      tokenFeatures.append("drugBrandName=false")
#
   #   result.append(tokenFeatures)
#
   #return result

## --------- MAIN PROGRAM ----------- 
## --
## -- Usage:  baseline-NER.py target-dir
## --
## -- Extracts Drug NE from all XML files in target-dir, and writes
## -- them in the output format requested by the evalution programs.
## --


# directory with files to process
datadir = sys.argv[1]

# process each file in directory
for f in listdir(datadir) :
   
   # parse XML file, obtaining a DOM tree
   tree = parse(datadir+"/"+f)
   
   # process each sentence in the file
   sentences = tree.getElementsByTagName("sentence")
   for s in sentences:
      sid = s.attributes["id"].value   # get sentence id
      spans = []
      stext = s.attributes["text"].value   # get sentence text
      entities = s.getElementsByTagName("entity")
      for e in entities :
         # for discontinuous entities, we only get the first span
         # (will not work, but there are few of them)
         (start,end) = e.attributes["charOffset"].value.split(";")[0].split("-")
         typ =  e.attributes["type"].value
         spans.append((int(start),int(end),typ))
         

      # convert the sentence to a list of tokens
      tokens = tokenize(stext)
      # extract sentence features
      features = extract_features(tokens)

      # print features in format expected by crfsuite trainer
      for i in range (0,len(tokens)):
         # see if the token is part of an entity
         tag = get_tag(tokens[i], spans) 
         print (sid, tokens[i][0], tokens[i][1], tokens[i][2], tag, "\t".join(features[i]), sep='\t')

      # blank line to separate sentences
      print()
