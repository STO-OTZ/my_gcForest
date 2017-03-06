import numpy as np
from sklearn.ensemble import RandomForestClassifier
import random
from sklearn.datasets import fetch_mldata
from multigrainedscaner import MultiGrainedScaner
from cascadeforest import CascadeForest

mnist = fetch_mldata('MNIST original')

# Trunk the data
n_train = 60000
n_test = 10000

# Define training and testing sets
train_idx = np.arange(n_train)
test_idx = np.arange(n_test)+n_train
random.shuffle(train_idx)

X_train, y_train = mnist.data[train_idx], mnist.target[train_idx]
X_test, y_test = mnist.data[test_idx], mnist.target[test_idx]

scan_forest_params1 = RandomForestClassifier(n_estimators=30,min_samples_split=21,max_features=1,n_jobs=-1).get_params()
scan_forest_params2 = RandomForestClassifier(n_estimators=30,min_samples_split=21,max_features='sqrt',n_jobs=-1).get_params()

cascade_forest_params1 = RandomForestClassifier(n_estimators=1000,min_samples_split=11,max_features=1,n_jobs=-1).get_params()
cascade_forest_params2 = RandomForestClassifier(n_estimators=1000,min_samples_split=11,max_features='sqrt',n_jobs=-1).get_params()

scan_params_list = [scan_forest_params1,scan_forest_params2]
cascade_params_list = [cascade_forest_params1,cascade_forest_params2]*2

def calc_accuracy(pre,y):
    return float(sum(pre==y))/len(y)
class ProbRandomForestClassifier(RandomForestClassifier):
    def predict(self, X):
        return RandomForestClassifier.predict_proba(self, X)

train_size = 1000
# gcForest

# Multi-Grained Scan Step
Scaner1 = MultiGrainedScaner(ProbRandomForestClassifier(), scan_params_list, sliding_ratio = 1./4)
Scaner2 = MultiGrainedScaner(ProbRandomForestClassifier(), scan_params_list, sliding_ratio = 1./9)
Scaner3 = MultiGrainedScaner(ProbRandomForestClassifier(), scan_params_list, sliding_ratio = 1./16)

X_train_scan =np.hstack([scaner.scan_fit(X_train[:train_size].reshape((train_size,28,28)), y_train[:train_size])
                             for scaner in [Scaner1,Scaner2,Scaner3][:1]])
X_test_scan = np.hstack([scaner.scan_predict(X_test.reshape((len(X_test),28,28)))
                             for scaner in [Scaner1,Scaner2,Scaner3][:1]])

# Cascade RandomForest Step
CascadeRF = CascadeForest(ProbRandomForestClassifier(),cascade_params_list)
CascadeRF.fit(X_train_scan, y_train[:train_size])
y_pre_staged = CascadeRF.predict_staged(X_test_scan)
test_accuracy_staged = np.apply_along_axis(lambda y_pre: calc_accuracy(y_pre,y_test), 1, y_pre_staged)
print('\n'.join('level {}, test accuracy: {}'.format(i+1,test_accuracy_staged[i]) for i in range(len(test_accuracy_staged))))

# CascadeRF baseline
BaseCascadeRF = CascadeForest(ProbRandomForestClassifier(),cascade_params_list,k_fold=3)
BaseCascadeRF.fit(X_train[:train_size], y_train[:train_size])
y_pre_staged = BaseCascadeRF.predict_staged(X_test)
test_accuracy_staged = np.apply_along_axis(lambda y_pre: calc_accuracy(y_pre,y_test), 1, y_pre_staged)
print('\n'.join('level {}, test accuracy: {}'.format(i+1,test_accuracy_staged[i]) for i in range(len(test_accuracy_staged))))

# RF baseline
RF = RandomForestClassifier(n_estimators=1000)
RF.fit(X_train[:train_size], y_train[:train_size])
y_pre = RF.predict(X_test)
print(calc_accuracy(y_pre,y_test))
