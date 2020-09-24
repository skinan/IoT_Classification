
from sklearn.base import BaseEstimator


class BOS_Classifier(BaseEstimator):
    
    def __init__(self, classifier):
        self.classifier = classifier
    
    def fit(self, X_train, y_train):
        
        X_train = list(X_train['statistics'])
        self.classifier.fit(X_train, y_train)
            
    def predict_proba(self, X_test):
        
        X_test = list(X_test['statistics'])
        return self.classifier.predict_proba(X_test)
    
    def predict(self, X_test):
        
        X_test = list(X_test['statistics'])
        return self.classifier.predict(X_test)




