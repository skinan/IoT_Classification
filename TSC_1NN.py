from scipy.spatial import distance
import numpy as np

from sklearn.base import BaseEstimator

def DTWDistance(s1, s2, w = 1000):
        rows = len(s1) + 1
        cols = len(s2) + 1
        DTW = np.zeros((rows, cols))

        if w:
            w = max(w, abs(len(s1)-len(s2)))

            for i in range(0, rows):
                for j in range(0, cols):
                    DTW[i, j] = float('inf')

            DTW[0, 0] = 0

            for i in range(1, rows):
                for j in range(max(1, i-w), min(cols, i+w+1)):
                    DTW[i, j] = 0

            distance = 0

            for i in range(1, rows):
                for j in range(max(1, i-w), min(cols, i+w+1)):
                    distance = np.sqrt((s1[i-1] - s2[j-1]) ** 2)
                    DTW[i,j] = distance + min(DTW[i-1,j], DTW[i-1,j-1], DTW[i, j-1])
        return DTW[len(s1), len(s2)]


# In[74]:


class TSC_1NN(BaseEstimator):
    
    MAX_DIST = float('inf')
    dictionary = {}
    probs = []
    
    def __init__(self, metric):
        self.metric = metric
        self.probs = []
    
    def fit(self, X_train, y_train):
        
        X_train = list(X_train['timeseries'])
        self.classes = set(y_train)
        for c in self.classes:
            self.dictionary[c] = []
        
        for i, ts in enumerate(X_train):
            c1 = y_train[i]
            self.dictionary[c1].append(ts)
            
    def predict_proba(self, X_test):
        
        X_test = list(X_test['timeseries'])
        self.probs = []
        for i, ts in enumerate(X_test):
            distances = []
            for c1 in self.classes:
                dam = [self.metric(ts, field) for field in self.dictionary[c1]]
                if len(dam) > 0:
                    # aggiungiamo 1 perchÃƒÂ¨ divideremo dopo
                    distances.append((min(dam)+0.001))
                else:
                    distances.append(MAX_DIST)

            distances = np.true_divide(1, distances)
            proba = np.true_divide(distances, sum(distances))
            self.probs.append(proba)
            
        return self.probs   
    
    def predict(self, X_test):
        
        self.probs = self.predict_proba(X_test)
        y_pred = []
        for series in self.probs:
            y_pred.append(np.argmax(series))
        
        return y_pred




