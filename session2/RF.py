# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 16:35:49 2024

@author: cperez
"""
from sklearn.ensemble import RandomForestClassifier
import joblib

class RandomForest:

    ## --------------------------------------------------
    ## Constructor: Load model from file
    ## --------------------------------------------------
    def __init__(self, datafile):
        # Create a CRF Tagger object, and load given model
        self.tagger =  RandomForestClassifier.Tagger()
        self.tagger.open(datafile)
        
    ## --------------------------------------------------
    ## predict best class for each element in xseq
    ## --------------------------------------------------
    def predict(self, xseq):
        return self.tagger.tag(xseq)

