import pandas as pd
import numpy as np
import csv
from sklearn import preprocessing, linear_model ,ensemble
import pickle

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
import sys

#import time
#start_time = time.time()


class data_manager():
 def read_data(self,filename,data):
    if filename=="Y_train":
        pre_data=pd.read_csv(data,header=None) #18 feature 
        data=pre_data.values
        return data
    pre_data=pd.read_csv(data) #18 feature
    data=pre_data.values
    if filename=="X_train":
        self.scaler = preprocessing.StandardScaler().fit(data)
        return self.scaler.transform(data)          
    if filename=="X_test":
        return self.scaler.transform(data)   
    

    return preprocessing.scale(data)


class DATA_process():
 def __init__(self):
    self.mean=0.0
    self.std=0.0 
 def read_data(self,data):
     pre_data=pd.read_csv(data) #18 feature
     data=pre_data.values
     print(data)
     if self.mean==0.0:
         self.normalized(data)
         return np.array(data)

 def normalized(self,data):
     index=[0,1,3,4,5]
     #data[:,index]=np.log1p(data[:,index])
     self.mean=np.zeros(data.shape[1])
     self.std=np.ones(data.shape[1])
     hold=np.mean(data,axis=0)
     self.mean[index]= hold[index]
     hold=np.std(data,axis=0)
     self.std[index] = hold[index]
     return (data-self.mean)/self.std

 def read_Y_train(self,data):
     pre_data=pd.read_csv(data,header=None) #18 feature
     data=pre_data.values
     print(data)
     print(data.shape)
     return np.array(data)




if __name__ == "__main__":
    X_train=DATA_process().read_data(sys.argv[3])#3
    Y_train=DATA_process().read_Y_train(sys.argv[4])#4
    X_test=DATA_process().read_data(sys.argv[5])#4
    
    #X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.2, random_state=0)

    #Gradient boosting

    filename = 'weight_best_GB.sav'

    classifier = pickle.load(open(filename, 'rb'))
    '''
    classifier =ensemble.GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, min_samples_split=2, min_samples_leaf=1, max_depth=3, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False)
    ''' 
    classifier.fit(X_train, Y_train.ravel())
    Y_test = classifier.predict(X_test)  
    result=np.sum((Y_train.ravel()==classifier.predict(X_train))) /Y_train.shape[0]
    
    print("Training accuraccy:", result)
    with open(sys.argv[6], 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['id','label'])
            for i,out in enumerate(Y_test) :
                writer.writerow([str(i+1),out])
    

    #joblib.dump(classifier, 'weight.sav') 
    '''
    filename = 'weight.sav'
    pickle.dump(classifier, open(filename, 'wb'))
    '''
    #print("--- %s seconds ---" % (time.time() - start_time))

