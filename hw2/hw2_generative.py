import numpy as np
import math
import pandas as pd
import sys
import csv

dim = 106

def load_data(X_train,Y_train,X_test):
    print(X_train,Y_train,X_test)
    x_train = pd.read_csv(X_train)
    x_test = pd.read_csv(X_test)

    x_train = x_train.values
    x_test = x_test.values

    y_train = pd.read_csv(Y_train, header = None)
    y_train = y_train.values
    y_train = y_train.reshape(-1)

    return x_train, y_train, x_test

def sigmoid(z):
    res = 1 / (1.0 + np.exp(-z))
    return np.clip(res, 1e-6, 1-1e-6)
def train(x_train, y_train):
    cnt1 = 0
    cnt2 = 0
    
    mu1 = np.zeros((dim,))
    mu2 = np.zeros((dim,))
    
    for i in range(x_train.shape[0]):
        if y_train[i] == 1:
            cnt1 += 1
            mu1 += x_train[i]
        else:
            cnt2 += 1
            mu2 += x_train[i]
    mu1 /= cnt1
    mu2 /= cnt2

    sigma1 = np.zeros((dim,dim))
    sigma2 = np.zeros((dim,dim))
    print("ff",x_train[0].shape)
    for i in range(x_train.shape[0]):
        if y_train[i] == 1:
            sigma1 += np.dot(np.transpose([x_train[i] - mu1]), [(x_train[i] - mu1)])
        else:
            sigma2 += np.dot(np.transpose([x_train[i] - mu2]), [(x_train[i] - mu2)])
    sigma1 /= cnt1
    sigma2 /= cnt2

    
    share_sigma = (cnt1 / x_train.shape[0]) * sigma1 + (cnt2 / x_train.shape[0]) * sigma2
    return mu1, mu2, share_sigma, cnt1, cnt2


class DATA_process():
 def __init__(self):
    self.mean=0.0
    self.std=0.0 
 def read_data(self,data):
     pre_data=pd.read_csv(data) #18 feature
     data=pre_data.values
     #print(data)
     if self.mean==0.0:
         #self.normalized(data)
         return np.array(data)

 def normalized(self,data):
     index=[0,1,3,4,5]
     #print(data.shape)
     data[:,3:4]=np.log1p(data[:,3:4])
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
     #print(data)
     #print(data.shape)
     return np.array(data)
    
def predict(x_test, mu1, mu2, share_sigma, N1, N2):
    sigma_inverse = np.linalg.inv(share_sigma)

    w = np.dot( (mu1-mu2), sigma_inverse)
    b = (-0.5) * np.dot(np.dot(mu1.T, sigma_inverse), mu1) + (0.5) * np.dot(np.dot(mu2.T, sigma_inverse), mu2) + np.log(float(N1)/N2)

    z = np.dot(w, x_test.T) + b
    pred = sigmoid(z)
    return pred
if __name__ == '__main__':
    #x_train,y_train,x_test = load_data(sys.argv[3],sys.argv[4],sys.argv[5])
    x_train=DATA_process().read_data(sys.argv[3])#3
    y_train=DATA_process().read_Y_train(sys.argv[4])#4
    x_test=DATA_process().read_data(sys.argv[5])#4

    mu1, mu2, shared_sigma, N1, N2 = train(x_train, y_train)

    
    y = predict(x_train, mu1, mu2, shared_sigma, N1, N2)
    
    y = np.around(y)
    
    result = (y_train.flatten() == y)
    print('Train acc = %f' % (int(result.sum()) / result.shape[0]))

    y_test = predict(x_test, mu1, mu2, shared_sigma, N1, N2)
    with open(sys.argv[6], 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id','label'])
        for i,out in enumerate(y_test) :
            if out>=0.5:writer.writerow([str(i+1),1])
            else: writer.writerow([str(i+1),0])

    
    #predict x_test