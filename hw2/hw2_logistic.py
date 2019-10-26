import csv
import sys
import numpy as np
import pandas as pd
import math 


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


def cross_entropy(y_pred, Y_label):
    loss= -1*np.dot(y_pred.T,np.log(Y_label))-1*np.dot( (1-y_pred).T,np.log(1-Y_label))
    return loss



def sigmoid(x):
    return np.clip(1 / (1 + np.exp(-x)),1e-6,1-1e-6)

def log_likelihood(features,target,weights,bias):
    score=np.dot(features,weights)+bias
    l1=np.sum(target*score-np.log(1+np.exp(score)))
    return l1

def accuraccy(y_pred,y):
    return np.sum( (y_pred==y) / len(y) )


def training(max_epoch,lr,X_train,Y_train,X_testing,lam,valid_r):
    #==============validation===================
    hold=np.concatenate((X_train,Y_train),axis=1)
    
    np.random.shuffle(hold)
    _val_length=int((1-valid_r)*X_train.shape[0])

    X_train=hold[0:_val_length,0:X_train.shape[1]]
    Y_train=hold[0:_val_length,X_train.shape[1]:]

    val_X_train=hold[_val_length:,0:X_train.shape[1]]
    val_Y_train=hold[_val_length:,X_train.shape[1]:]

    #============================================
    batch_size = 64
    B=0.1
    W=np.full((X_train.shape[1],1),0.1)
    print("W matrix shape:",W.shape)
    print("B matrix value:",B)
    Loss=Y_train-sigmoid(np.dot(X_train,W)+B)
    # 打亂data順序
    index = np.arange(X_train.shape[0])
    #np.random.seed(0)
    np.random.shuffle(index)
    #x = X_train[index]
    #y = Y_train[index]
    #############Adam parameter#############
    beta_1 = np.full((X_train.shape[1],1), 0.9).reshape(-1, 1)#shape(162*1)
    beta_2 = np.full((X_train.shape[1],1), 0.99).reshape(-1, 1)#shape(162*1)
    m_t = np.full((X_train.shape[1],1), 0).reshape(-1, 1) #shape(162*1) 
    v_t = np.full((X_train.shape[1],1), 0).reshape(-1, 1)#shape(162*1) 
    m_t_b = 0.0
    v_t_b = 0.0
    t = 0
    epsilon=1e-8


    for epoch in range(0,max_epoch):
        for b in range(int(_val_length/batch_size)):
            t+=1
            x_batch = X_train[b*batch_size:(b+1)*batch_size]
            y_batch = Y_train[b*batch_size:(b+1)*batch_size]
        
            Loss=y_batch-sigmoid(np.dot(x_batch,W)+B)

            b_grad=-sum(Loss)
            w_grad=-np.transpose(x_batch).dot(Loss) +  2 * lam * np.sum(W)

            m_t = beta_1*m_t + (1-beta_1)*w_grad 
            v_t = beta_2*v_t + (1-beta_2)*np.multiply(w_grad, w_grad)
            m_cap = m_t/(1-(beta_1**t))
            v_cap = v_t/(1-(beta_2**t))

            m_t_b = 0.9*m_t_b + (1-0.9)*b_grad
            v_t_b = 0.99*v_t_b + (1-0.99)*(b_grad*b_grad) 
            m_cap_b = m_t_b/(1-(0.9**t))
            v_cap_b = v_t_b/(1-(0.99**t))
            W -= ((lr*m_cap)/(np.sqrt(v_cap)+epsilon)).reshape(-1, 1)
            B -= (lr*m_cap_b)/(math.sqrt(v_cap_b)+epsilon)

            '''
            Adagrad:
            b_adagrad=0.0
            w_adagrad=0.0
            b_adagrad+=np.dot(np.transpose(b_grad),b_grad ) 
            w_adagrad+=np.dot(np.transpose(w_grad),w_grad ) 
            B=B-lr*b_grad/np.sqrt(b_adagrad+1e-8)
            W=W-lr*w_grad/np.sqrt(w_adagrad+1e-8)
            '''
     


        if epoch %20==0: 
                #validation_test
                y_pred_val=sigmoid(np.dot(val_X_train,W)+B)
                y_pred_val[np.where( y_pred_val >= 0.5 ) ]=1
                y_pred_val[np.where( y_pred_val < 0.5 ) ]=0
                #training_test
                y_pred=sigmoid(np.dot(X_train,W)+B)
                y_pred[np.where( y_pred >= 0.5 ) ]=1
                y_pred[np.where( y_pred < 0.5 ) ]=0

                #print("="*40)
                print("Epoch",epoch)
                print("Training accuraccy:",accuraccy(y_pred,Y_train) )#log likelihood
                if valid_r>0: 
                    print("Testing accuraccy:",accuraccy(y_pred_val,val_Y_train))#log likelihood
                #print("="*40)

    return W,B


if __name__ == "__main__":
    X_train=DATA_process().read_data(sys.argv[3])#3
    Y_train=DATA_process().read_Y_train(sys.argv[4])#4
    X_test=DATA_process().read_data(sys.argv[5])#4
    #W,B=training(1000,1e-5,X_train,Y_train,X_test,0.0001,0.0)
    W=np.array(pd.read_csv("weight_log.csv",header=None)) #18 feature
    B=np.array(pd.read_csv("bias_log.csv",header=None) )#18 feature 
    #np.savetxt("weight.csv",W , delimiter=",")
    #np.savetxt("bias.csv", B, delimiter=",")
    predict=sigmoid(np.dot(X_test,W)+B)

    with open(sys.argv[6], 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['id','label'])
            for i,out in enumerate(predict) :
                if out[0]>=0.5:writer.writerow([str(i+1),1])
                else: writer.writerow([str(i+1),0])