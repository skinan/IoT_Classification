import numpy as np
from sklearn.metrics import accuracy_score
from scipy.spatial import distance
from BOS_Wrapper import BOS_Classifier
from NLP_Classifier import NLP_Classifier
from TSC_1NN import TSC_1NN, DTWDistance
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from heapq import nlargest

class EnsembleFramework:
    
    def __init__(self, criterion = 'topk', tuning = False, 
                 layers = [{'type' : 'NLP'}, {'type' : 'BOS', 'name' : 'DecisionTreeClassifier()'}], 
                 params = {'k' : [4, 1]}):
        self.criterion = criterion
        self.layers = layers
        self.params = params
        self.tuning = tuning
        self.selected = []
    
    # filtraggio sof - la funzione riceve in input la lista di probabilità per tutte le classi e la lista delle classi rimaste
    def survival(self, probs, classes):
        survived = []
        survived_probs = []
        
        # selezioniamo solo le probabilità delle classi rimaste
        for c in classes:
            survived_probs.append(probs[c])
        
        # calcoliamo media e std
        media = np.mean(survived_probs)
        std_dev = np.std(survived_probs)
        
        # filtriamo in base al valore di sigma
        for c in classes:
            if (probs[c] >= (media + self.params['sigma']*std_dev)):
                survived.append(c)
        
        # se per qualche motivo non rimangono classi, selezioniamo solo quella più probabile
        if (len(survived) == 0):
            survived = nlargest(1, classes, key = lambda x : probs[x])
        
        # teniamo traccia di quante classi rimangono dopo ciascuno step
        self.selected.append(len(survived))
            
        return survived
    
    def run(self, X_train, y_train, X_test):
        # inizializziamo la lista di classi per ogni istanza del test set
        classes = [list(set(y_train))] * len(X_test)
        
        # lista di classificatori
        self.classifiers = []
        
        # riempiamo la lista
        for classifier in self.layers:
            if (classifier['type'] == 'BOS'):
                clf = BOS_Classifier(eval(classifier['name']))
            elif (classifier['type'] == 'NLP'):
                clf = NLP_Classifier()
            elif (classifier['type'] == 'TSC'):
                clf = eval(classifier['name'])
            clf.fit(X_train, y_train)
            self.classifiers.append(clf)
        
        # topk filtering 
        if (self.criterion == 'topk'):
            # per ogni classificatore e relativo valore k
            for clf, k in zip(self.classifiers, self.params['k']):
                # calcoliamo le probabilità di appartenenza alle classi per il test set
                probs = clf.predict_proba(X_test)
                if (self.tuning):
                    X_test = self.ftuning(X_test, probs)
                # per ogni istanza, selezioniamo le k classi più probabili tra quelle rimaste
                for i, series_probs in enumerate(probs):
                    classes[i] = nlargest(k, classes[i], key = lambda x : series_probs[x])
        
        # sof filtering
        elif (self.criterion == 'sof'):
            for j, clf in enumerate(self.classifiers):
                probs = clf.predict_proba(X_test)
                if (self.tuning):
                    X_test = self.ftuning(X_test, probs)
                # per ogni istanza del test set, filtriamo le classi tramite sof
                for i, series_probs in enumerate(probs):
                    classes[i] = self.survival(series_probs, classes[i])
                    # se il classificatore è l'ultimo nella lista, selezioniamo in automatico la classe più probabile
                    if (j == (len(self.classifiers)-1)):
                        classes[i] = nlargest(1, classes[i], key = lambda x : series_probs[x])
                        
        elif (self.criterion == 'qf'):
            for j, clf in enumerate(self.classifiers):
                probs = clf.predict_proba(X_test)
                if (self.tuning):
                    X_test = self.ftuning(X_test, probs)
                for i, series_probs in enumerate(probs):
                    k = int(len(classes[i])*self.params['q'])
                    if (j == (len(self.classifiers)-1) or k == 0):
                        classes[i] = nlargest(1, classes[i], key = lambda x : series_probs[x])
                    else:
                        classes[i] = nlargest(k, classes[i], key = lambda x : series_probs[x])
                        
        elif (self.criterion == 'tuning'):
            for j, clf in enumerate(self.classifiers):
                probs = clf.predict_proba(X_test)
                X_test = self.ftuning(X_test, probs)
                if (j == (len(self.classifiers)-1)):
                    for i, series_probs in enumerate(probs):
                        classes[i] = nlargest(1, classes[i], key = lambda x : series_probs[x])
                
        return classes  
    
    def ftuning(self, X_test, probs):
        for prob_list, (_, row) in zip(probs, X_test.iterrows()):
            row['statistics'] = row['statistics'][:-len(prob_list)]
            row['statistics'] = np.append(row['statistics'], prob_list) 
            
        return X_test
    
    def accuracy(self, classes, y_test):
        y_pred = []
        for pred in classes:
            y_pred.append(*pred)
            
        return accuracy_score(y_pred, y_test)*100