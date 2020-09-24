import numpy as np
from pyxdameraulevenshtein import damerau_levenshtein_distance

from sklearn.base import BaseEstimator

class NLP_Classifier(BaseEstimator):
    
    MAX_DIST = 1000
    dictionary = {}
    probs = []
    
    def __init__(self):
        pass
    
    def fit(self, X_train, y_train):
        
        self.classes = set(y_train)
        for c in self.classes:
            self.dictionary[c] = []
        
        X_train = X_train['metadata']
        for i, word in enumerate(X_train):
            c1 = y_train[i]
            self.dictionary[c1].append(word)
            
    def predict_proba(self, X_test):
        
        self.probs = []
        for i, word in enumerate(X_test['metadata']):
            
            edit_distances = []
            for c1 in self.classes:
                dam = [damerau_levenshtein_distance(word, field)/max([len(word), len(field)]) for field in self.dictionary[c1]]
                if len(dam) > 0:
                    # aggiungiamo 1 perchĂŠ divideremo dopo
                    edit_distances.append((min(dam)+0.000001))
                else:
                    edit_distances.append(MAX_DIST)
            
            edit_distances = [x**2 for x in edit_distances]
            edit_distances = np.true_divide(1, edit_distances)
            proba = np.true_divide(edit_distances, sum(edit_distances))
            self.probs.append(proba)
            
        return self.probs   
    
    def predict(self, X_test):
        
        self.probs = self.predict_proba(X_test)
        y_pred = []
        for series in self.probs:
            y_pred.append(np.argmax(series))
        
        return y_pred




