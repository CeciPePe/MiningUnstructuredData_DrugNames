# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 16:35:48 2024

@author: cperez
"""
from sklearn.svm import SVC
import joblib

class SVM:

    ## --------------------------------------------------
    ## Constructor: Load model from file
    ## --------------------------------------------------
    def __init__(self, datafile):
        # Create a CRF Tagger object, and load given model
        self.tagger =  SVC.Tagger()
        self.tagger.open(datafile)
        
    ## --------------------------------------------------
    ## predict best class for each element in xseq
    ## --------------------------------------------------
    def predict(self, xseq):
        return self.tagger.tag(xseq)
