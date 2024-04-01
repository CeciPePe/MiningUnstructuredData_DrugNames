# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 11:01:42 2024

@author: cperez
"""
import numpy as np
import argparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import SVC
from joblib import dump
import sys

def fix_format(token):
    if 'BoS' in token:
        token = token.replace('BoS','formPrev=BoS	suf3Prev=BoS')
    if 'EoS' in token:
        token = token.replace('EoS','formNext=EoS	suf3Next=EoS')
    return token

def load_data(data):
    features = []
    labels = []
    for token in data:
        token = token.strip()
        token = fix_format(token).split('\t')
        token_dict = {feat.split('=')[0]:feat.split('=')[1] for feat in token[1:]}
        features.append(token_dict)
        labels.append(token[0])
    return features, labels


if __name__ == '__main__':

	model_file = sys.argv[1]
	vectorizer_file = sys.argv[2] 	

	train_features, y_train = load_data(sys.stdin)
	y_train = np.asarray(y_train)
	classes = np.unique(y_train)

	v = DictVectorizer()
	X_train = v.fit_transform(train_features)

	clf = SVC()
	clf.partial_fit(X_train, y_train)


	dump(clf, model_file) 
	dump(v, vectorizer_file)