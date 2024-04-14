#! /usr/bin/python3
## ---------------------------------------------------------------------------- Libraries --------------------------------------------------------------------------------- 
import sys
import re
from os import listdir
from xml.dom.minidom import parse
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
import nltk
#nltk.download('averaged_perceptron_tagger')
#nltk.download('wordnet')
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
## ---------------------------------------------------------------------------- Default functions from lab --------------------------------------------------------------------------------- 
## --------- tokenize sentence ----------- 
## -- Tokenize sentence, returning tokens and span offsets
#It iterates over each token generated with word_tokenize and by finding the position of the token being checked it appends the token wit its start offset and end
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
#get_tag function gets information from token and it iterates over each span checking if start offset of the token checked matches the start offset of the span and if end offset is less or equal to the end ffset of span. 
#This indicates the beginning of the entity and if it's not the case, and the start offsett of token is  is within the span and the end offset is less than or equal to the end offset of the span, it will indicate it's in the inside.
#Other cases will indicate outside of entity.
 
def get_tag(token, spans) :
   (form,start,end) = token
   for (spanS,spanE,spanT) in spans :
      if start==spanS and end<=spanE : return "B-"+spanT
      elif start>=spanS and end<=spanE : return "I-"+spanT

   return "O"
 
## --------- Feature extractor ----------- 
## -- Extract features for each token in given sentence
#iterates through each token in the sentence and adds it's features based on form (form start offset and end offset of token). The features are added based on the form, the last 3 characters of the token, iif token not the first
#in the sentence then previous token is added, and viceversa, if it's not last token, the features are based on next token. If the features are special then they are added according to BoS or EoS (beginning or end).
def extract_features(tokens) :
   result = []
   for k in range(0,len(tokens)):
      tokenFeatures = [];
      t = tokens[k][0]

      tokenFeatures.append("form="+t)
      tokenFeatures.append("suf3="+t[-3:])

      if k>0 :
         tPrev = tokens[k-1][0]
         tokenFeatures.append("formPrev="+tPrev)
         tokenFeatures.append("suf3Prev="+tPrev[-3:])
      else :
         tokenFeatures.append("BoS")

      if k<len(tokens)-1 :
         tNext = tokens[k+1][0]
         tokenFeatures.append("formNext="+tNext)
         tokenFeatures.append("suf3Next="+tNext[-3:])
      else:
         tokenFeatures.append("EoS")
    
      result.append(tokenFeatures)
    
   return result
## ---------------------------------------------------------------------------- Used Extract-features functions --------------------------------------------------------------------------------- 
## --------- DrugBank ----------
def drugbank_extraction(tokens):
    drug_brand_names = []
    drug_names = []
    group_names = []
    #read the drug information that there is in the DrugBank.txt file provided by the lab
    with open('/Users/bielcave/Documents/MDS/4th_Semester/MUD/lab_resources/DDI/resources/DrugBank.txt', 'r') as file:
        for line in file:
            parts = line.strip().split('|')
            if parts[1] == 'brand':
                drug_brand_names.append(parts[0])
            elif parts[1] == 'drug':
                drug_names.append(parts[0])
            elif parts[1] == 'group':
                group_names.append(parts[0])

    result = []
    #iterate through each token in the list and add features based on previous and next token. Finally it checks if the tokens match the drug names 
    for k in range(0, len(tokens)):
        tokenFeatures = []
        t = tokens[k][0]

        tokenFeatures.append("form=" + t)
        tokenFeatures.append("suf3=" + t[-3:])

        if k > 0:
            tPrev = tokens[k - 1][0]
            tokenFeatures.append("formPrev=" + tPrev)
            tokenFeatures.append("suf3Prev=" + tPrev[-3:])
        else:
            tokenFeatures.append("BoS")

        if k < len(tokens) - 1:
            tNext = tokens[k + 1][0]
            tokenFeatures.append("formNext=" + tNext)
            tokenFeatures.append("suf3Next=" + tNext[-3:])
        else:
            tokenFeatures.append("EoS")

        # Check for drug, brand, and group names
        if t in drug_brand_names:
            tokenFeatures.append("drugBank=brand")
        elif t in drug_names:
            tokenFeatures.append("drugBank=drug")
        elif t in group_names:
            tokenFeatures.append("drugBank=group")
        else:
            tokenFeatures.append("drugBank=other")

        result.append(tokenFeatures)

    return result
   
## --------- capitalization and brand names -----------
def cap_brand_names(tokens):
    drug_brand_names = []
    #read drugbank file
    with open('/Users/bielcave/Documents/MDS/4th_Semester/MUD/lab_resources/DDI/resources/DrugBank.txt', 'r') as file:
        for line in file:
            parts = line.strip().split('|')
            if parts[1] == 'brand':
                drug_brand_names.append(parts[0])

    result = []
    #iterte through each token in the list of tokens and add features based on the form and suffix pf the tken and based on previous and next tokens
    for k in range(0, len(tokens)):
        tokenFeatures = []
        t = tokens[k][0]

        tokenFeatures.append("form=" + t)
        tokenFeatures.append("suf3=" + t[-3:])

        if k > 0:
            tPrev = tokens[k - 1][0]
            tokenFeatures.append("formPrev=" + tPrev)
            tokenFeatures.append("suf3Prev=" + tPrev[-3:])
        else:
            tokenFeatures.append("BoS")

        if k < len(tokens) - 1:
            tNext = tokens[k + 1][0]
            tokenFeatures.append("formNext=" + tNext)
            tokenFeatures.append("suf3Next=" + tNext[-3:])
        else:
            tokenFeatures.append("EoS")

        # Check for capitalization (isupper)
        if t.isupper():
            tokenFeatures.append("capitalization=true")
        else:
            tokenFeatures.append("capitalization=false")

        # Check for drug brand names
        if t in drug_brand_names:
            tokenFeatures.append("drugBrandName=true")
        else:
            tokenFeatures.append("drugBrandName=false")

        result.append(tokenFeatures)

    return result

## --------- contextual features -----------
def contextual_features(tokens):
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
                token_features.append(f"BoS{i}")#Feature: Beginning of Sentence (BoS)

        # Features for next tokens within the window size
        for i in range(1, 1 + 1):
            if k + i < len(tokens):
                t_next = tokens[k + i][0]
                token_features.append(f"formNext{i}=" + t_next)
                token_features.append(f"suf3Next{i}=" + t_next[-3:])
            else:
                token_features.append(f"EoS{i}")#Feature: End of Sentence (EoS)

        result.append(token_features)

    return result

## --------- prefixes-sufixes ----------- 
def extract_features_prefix_suffix(tokens):
    result = []
    token_strings = []
    prefix_length=3
    suffix_length=3
    for k in range(len(tokens)):
        tokenFeatures = []
        t = tokens[k][0]

        tokenFeatures.append("form=" + t)
        tokenFeatures.append("prefix=" + t[:prefix_length])
        tokenFeatures.append("suffix=" + t[-suffix_length:])

        if k > 0:
            tPrev = tokens[k-1][0]
            tokenFeatures.append("formPrev=" + tPrev)
            tokenFeatures.append("suf3Prev=" + tPrev[-suffix_length:])
        else:
            tokenFeatures.append("BoS")

        if k < len(tokens) - 1:
            tNext = tokens[k+1][0]
            tokenFeatures.append("formNext=" + tNext)
            tokenFeatures.append("suf3Next=" + tNext[-suffix_length:])
        else:
            tokenFeatures.append("EoS")
        
        result.append(tokenFeatures)
    return result


#-----------------------------final combination of extract feature methods-----------------------
def final_extract_features(tokens):
    #Combination of getting prefixes, suffixes of 3 letters, Dash detection, Numeric detection, Capitalizatio, The name itself compared in DrugBank, Post-tags
    result = []
    prefix_length=3
    suffix_length=3
    stop_words = set(stopwords.words('english'))

    drug_brand_names = []
    drug = []
    group_names = []
    drug_names = []
    
    with open('/Users/bielcave/Documents/MDS/4th_Semester/MUD/lab_resources/DDI/resources/DrugBank.txt', 'r') as file:
        for line in file:
            parts = line.strip().split('|')
            if parts[1] == 'brand':
                drug_brand_names.append(parts[0])
            elif parts[1] == 'drug':
                drug.append(parts[0])
            elif parts[1] == 'group':
                group_names.append(parts[0])
    
    with open('/Users/bielcave/Documents/MDS/4th_Semester/MUD/lab_resources/DDI/resources/HSDB.txt', 'r') as file:
        for line in file:
            drug_names.append(line.strip())
    
    pos_tags = nltk.pos_tag([token[0] for token in tokens])
    lemmatizer = WordNetLemmatizer()

    for k, (token, pos_tag) in enumerate(zip(tokens, pos_tags)):
        tokenFeatures = []
        t = tokens[k][0]
        
        # Basic features
        tokenFeatures.append("pos_tag=" + pos_tag[1])
        tokenFeatures.append("form=" + t)
        tokenFeatures.append("prefix=" + t[:prefix_length])
        tokenFeatures.append("suffix=" + t[-suffix_length:])
        
        # Lemma
        lemma = lemmatizer.lemmatize(t)
        tokenFeatures.append("lemma=" + lemma)
        
        # Additional features
        tokenFeatures.append("length=" + str(len(t)))
        tokenFeatures.append("hasDash=" + str(bool(re.search('-', t))))
        tokenFeatures.append("containNumber=" + str(bool(re.search(r'\d', t))))
        tokenFeatures.append("isStopword=" + str(t.lower() in stop_words))
        tokenFeatures.append("isPunctuation=" + str(bool(re.search(r'[^\w\s]', t))))
        tokenFeatures.append("isSpecialChar=" + str(bool(re.search(r'[^a-zA-Z0-9\s]', t))))
        
        # Features for previous token
        if k > 0:
            tPrev, posPrev = tokens[k-1][0], pos_tags[k-1][1]
            tokenFeatures.append("formPrev=" + tPrev)
            tokenFeatures.append("pos_tagPrev=" + posPrev)
            tokenFeatures.append("suf3Prev=" + tPrev[-suffix_length:])
        else:
            tokenFeatures.extend(["BoS", "pos_tagPrev=BoS"])
        
        # Features for next token
        if k < len(tokens) - 1:
            tNext, posNext = tokens[k+1][0], pos_tags[k+1][1]
            tokenFeatures.append("formNext=" + tNext)
            tokenFeatures.append("pos_tagNext=" + posNext)
            tokenFeatures.append("suf3Next=" + tNext[-suffix_length:])
        else:
            tokenFeatures.extend(["EoS", "pos_tagNext=EoS"])
        
        # Capitalization
        tokenFeatures.append("capitalization=true" if t.isupper() else "capitalization=false")

        # Drug-related features
        tokenFeatures.append("brand=true" if t in drug_brand_names else "brand=false")
        tokenFeatures.append("drug=true" if t in drug_names else "drug=false")
        tokenFeatures.append("group=true" if t in group_names else "group=false")
        tokenFeatures.append("drug_n=true" if t in drug else "drug_n=false")
        
        result.append(tokenFeatures)
    
    return result

## ------------------ Extract features postags for each token in given sentence ----------------------

def extract_pos_features(tokens):
    result = []
    pos_tags = nltk.pos_tag([token[0] for token in tokens])
    # Iterate through each token and its corresponding POS tag
    for k, (token, pos_tag) in enumerate(zip(tokens, pos_tags)):
        tokenFeatures = []
        tokenFeatures.append("form=" + token[0])#add features based on form and POS tag of token
        tokenFeatures.append("pos_tag=" + pos_tag[1])
        #based on previous token
        if k > 0 :
            tPrev = tokens[k-1][0]
            tokenFeatures.append("formPrev="+tPrev)
            tokenFeatures.append("suf3Prev="+tPrev[-3:])
        else :
            tokenFeatures.append("BoS")
        #based on next token
        if k < len(tokens)-1 :
            tNext = tokens[k+1][0]
            tokenFeatures.append("formNext="+tNext)
            tokenFeatures.append("suf3Next="+tNext[-3:])
        else:
            tokenFeatures.append("EoS")
            
        result.append(tokenFeatures)
    return result
## ---------------------------------------------------------------------------- Tried extract feature functions (not used) --------------------------------------------------------------------------------- 
#TF-IDF and Synonim detection was thought of and tried but never used do to appearance of errors in the code.
## ------------------- Drugnames to use for synonym detection ---------------------
#both input files from the lab are used
def is_drug_name(token):
    with open("C:/Users/cperez/OneDrive - IREC-edu/05_MASTER/MUD/lab_resources/lab_resources/DDI/resources/HSDB.txt", 'r') as file:
        hsdb_drug_names = {line.strip().lower() for line in file}
    with open("C:/Users/cperez/OneDrive - IREC-edu/05_MASTER/MUD/lab_resources/lab_resources/DDI/resources/DrugBank.txt", 'r') as file:
        drugbank_drug_names = {line.strip().lower() for line in file}
    all_drug = hsdb_drug_names | drugbank_drug_names
    return token.lower() in all_drug

    return token.lower() in drug_names
## --------- synonyms extraction ----------- 
# get synonims
def get_synonyms(word):
    synonyms = set()
    for synset in wordnet.synsets(word):
        for lemma in synset.lemmas():
            synonyms.add(lemma.name())
    return synonyms
## ------------------- Extract synonyms -------------
def extract_synonyms_features(tokens):
    result = []
    for k in range(len(tokens)):
        tokenFeatures = []
        t = tokens[k][0]
        tokenFeatures.append("form=" + t)
        tokenFeatures.append("suf3=" + t[-3:])
        if k > 0:
            tPrev = tokens[k-1][0]
            tokenFeatures.append("formPrev=" + tPrev)
            tokenFeatures.append("suf3Prev=" + tPrev[-3:])
        else:
            tokenFeatures.append("BoS")
        if k < len(tokens) - 1:
            tNext = tokens[k+1][0]
            tokenFeatures.append("formNext=" + tNext)
            tokenFeatures.append("suf3Next=" + tNext[-3:])
        else:
            tokenFeatures.append("EoS")
        result.append(tokenFeatures)

    expanded_features = []
    #if it's drug name then synonims for tokens from get_synonyms fucntion
    for token, features in zip(tokens, result):
        if is_drug_name(token[0]):
            synonyms = get_synonyms(token[0])
            expanded_token_features = [token[0]]
            for synonym in synonyms:
                expanded_token_features.append("synonym=" + synonym)
            expanded_features.append("\t".join(features + expanded_token_features))
        else:
            expanded_features.append("\t".join(features))

    return expanded_features
## -------------------- Extract features tf-idf for each token in given sentence-------------------------------
from sklearn.feature_extraction.text import TfidfVectorizer
def extract_tfidf_features(tokens):
    result = []
    token_strings = []
    prefix_length=3
    suffix_length=3
    for k in range(len(tokens)):
        tokenFeatures = []
        t = tokens[k][0]

        tokenFeatures.append("form=" + t)
        tokenFeatures.append("prefix=" + t[:prefix_length])
        tokenFeatures.append("suffix=" + t[-suffix_length:])

        if k > 0:
            tPrev = tokens[k-1][0]
            tokenFeatures.append("formPrev=" + tPrev)
            tokenFeatures.append("suf3Prev=" + tPrev[-suffix_length:])
        else:
            tokenFeatures.append("BoS")

        if k < len(tokens) - 1:
            tNext = tokens[k+1][0]
            tokenFeatures.append("formNext=" + tNext)
            tokenFeatures.append("suf3Next=" + tNext[-suffix_length:])
        else:
            tokenFeatures.append("EoS")
        
        result.append(tokenFeatures)
        token_strings.append(t)
    tfidf_vectorizer = TfidfVectorizer()
    #Transform token strings into TF-IDF features
    tfidf_features = tfidf_vectorizer.fit_transform(token_strings)
    #Get feature names from the TF-IDF vectorizer
    feature_names = tfidf_vectorizer.get_feature_names_out()
    #Iterate over each token and its corresponding tf-idf values
    for i, tfidf_row in enumerate(tfidf_features):
        tfidf_values = tfidf_row.toarray()[0]
        tfidf_token_features = [f"tfidf_{feature}={value:.3f}" for feature, value in zip(feature_names, tfidf_values)]
        result[i] += tfidf_token_features
    
    return result

## -------------------------------------------------------------------------- MAIN PROGRAM --------------------------------------------------------------------------------
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
      features = final_extract_features(tokens) ##change this part for every testing

      # print features in format expected by crfsuite trainer
      for i in range (0,len(tokens)):
         # see if the token is part of an entity
         tag = get_tag(tokens[i], spans) 
         print (sid, tokens[i][0], tokens[i][1], tokens[i][2], tag, "\t".join(features[i]), sep='\t')

      # blank line to separate sentences
      print()
