import csv
import sys
import numpy as np
import pandas as pd
import re
import math 

from scipy.stats.stats import pearsonr 
# Import tensor dataset & data loader

def data_pre_processing(file):
    #pre_processing/filter specific char
    pre_data=pd.read_csv(file) #18 feature
    pre_data = pre_data.fillna(0) # fill 0
    pre_data = pre_data.replace(['\A','\#','\*','x','NR'], ['','','','','0'], regex=True)
    #print(pre_data.info())
    #pre_processing/dimensional change
    data=pre_data.values
    data=data[:,2:].astype("float")
    print("Before data_pre_processing:",data.shape[0],data.shape[1])
    day=np.vsplit(data,data.shape[0]/18) # 366||365* shape(18,24)
    year=np.concatenate(np.array(day),axis=1)
    print("After data_pre_processing:",year.shape)
    return year

def valid(x, y):
    if y<=0 or y>73:
        #print("Data remove",y) 
        return False
    for i in x.flatten():
        if i<= 0 or i >73 :
            #print("Data remove",x)
            return False
    
    return True

def read_Training_data(file1,file2):
    day1=data_pre_processing(file1)  
    day2=data_pre_processing(file2)  
    #================evaluate statics================
    day_total=np.concatenate((day1,day2),axis=1)

    X,Y=[],[]
    #ranges = [(2, 3), (8, 10)]


    for i in range(0,18):
        print(i,np.corrcoef(day_total[9:10,:],day_total[i:i+1,:]))
    Quartile=np.percentile(day_total, [25, 50, 75])
    _range=Quartile[2]-Quartile[0]
    Extreme_upper_bound=Quartile[2]+1.5*_range
    Extreme_lower_bound=Quartile[0]-1.5*_range
    print("Extreme value:",Extreme_upper_bound,Extreme_lower_bound)
    print("Median value:",Quartile[1])
    print("Mean value:",day_total.mean())


    for i in range(0,day1.shape[1]-day1.shape[1]%10,10):
        if(valid(day1[9:10,i:i+9],day1[9:10,i+9:i+10])):
            Y.append(day1[9:10,i+9:i+10].flatten()) 
            X.append(day1[9:10,i:i+9].flatten()) #miss 4 data
            #X.append(np.concatenate((day1[6:7,i:i+9],day1[12:13,i:i+9],day1[9:10,i:i+9])).flatten()) #miss 4 data

    for i in range(0,day2.shape[1]-day2.shape[1]%10,10):
        if(valid(day2[9:10,i:i+9],day2[9:10,i+9:i+10])):
            Y.append(day2[9:10,i+9:i+10].flatten()) 
            X.append(day2[9:10,i:i+9].flatten()) #miss 4 data
            #X.append(np.concatenate((day2[6:7,i:i+9],day2[12:13,i:i+9],day2[9:10,i:i+9])).flatten()) 
    day_total=np.array(X)


    
    print("Training data X shape:",np.array(X).shape)
    print("Training data Y shape:",np.array(Y).shape)
    #normalization
    #X=(X-np.mean(X, axis=0))/(np.std(X, axis=0) + 1e-20)
    #Y=(Y-np.mean(Y, axis=0))/(np.std(Y, axis=0) + 1e-20)
    return np.array(X),np.array(Y)

def read_Testing_data(file):
    day=data_pre_processing(file) # 366* shape(18,9)
    X=[]
    for i in range(0,day.shape[1],9):
        #X.append( np.concatenate((day[6:7,i:i+9],day[12:13,i:i+9],day[9:10,i:i+9]),axis=0).flatten() ) #miss 4 data
        X.append(day[9:10,i:i+9].flatten()) #miss 4 data
    print("Testing data shape:",np.array(X).shape)
    
    return np.array(X)

def training(max_epoch,lr,X_train,Y_train,X_testing,lam,valid_r):
    #==============validation===================
    hold=np.concatenate((X_train,Y_train),axis=1)
    the_seed=np.random.randint(1,50)
    print("The seed:",the_seed)
    np.random.seed(the_seed)
    np.random.shuffle(hold)
    _val_length=int((1-valid_r)*X_train.shape[0])

    X_train=hold[0:_val_length,0:X_train.shape[1]]
    Y_train=hold[0:_val_length,X_train.shape[1]:]

    val_X_train=hold[_val_length:,0:X_train.shape[1]]
    val_Y_train=hold[_val_length:,X_train.shape[1]:]

    #============================================
        #W=np.zeros(shape=(X_train.shape[1],1)) #shape(162*1)  y=X*W+B
    batch_size = 64
    
    #B=np.ones(shape=(_val_length,1)) #shape(1762*1) #X(1762*162)
    '''
    Closed Form Solution
    '''
    B_aug=np.full((_val_length,1),1)
    W_aug=np.concatenate((B_aug,X_train),axis=1)
    W= np.linalg.inv(np.dot(np.transpose(W_aug),W_aug)).dot(np.transpose(W_aug).dot(Y_train))
    print("W matrix shape:",W.shape)
    B=W[0:1,:]
    W=W[1:,:]
    '''
    B=0.1
    W=np.full((X_train.shape[1],1),0.1)
    print("W matrix shape:",W.shape)
    print("B matrix value:",B)
    '''
    Loss=Y_train-(np.dot(X_train,W)+B)
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
        
            Loss=y_batch-(np.dot(x_batch,W)+B)
            b_grad=-2*sum(Loss) #shape(1762*1)
            w_grad=-2*np.transpose(x_batch).dot(Loss) +  2 * lam * np.sum(W) #shape(162*64) *  shape(64*1)

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


        if epoch %1000==0: 
                print("Epoch",epoch)
                loss= Y_train - np.dot(X_train,W) - B
                print("Training Loss:",np.power(np.sum(np.power(loss,2)/_val_length),0.5))
                if valid_r>0: 
                    val_X_train_length=val_X_train.shape[0]

                    #val_predict=np.dot(val_X_train,W[0:val_X_train_length,:])+B[0:val_X_train_length,:]
                    val_predict=np.dot(val_X_train,W[0:val_X_train_length,:])+B
                    val_loss=val_Y_train-val_predict
                    print("Testing Loss:",np.power(np.sum(np.power(val_loss,2))/val_X_train_length,0.5))
            
    print("="*30)
    #print("parameter W:",W)
    #print("parameter B:",B)
    #predict
    #_shape=np.zeros((Y_train.shape[0]-X_testing.shape[0],X_train.shape[1]))
    _shape=np.zeros((_val_length-X_testing.shape[0],X_train.shape[1]))
    if valid_r>0: 
        X_testing=np.concatenate((X_testing,_shape),axis=0)
    predict=np.dot(X_testing,W)+B
    with open('output_year2.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id','value'])
        for i,out in enumerate(predict[0:500]) :
            #writer.writerow(['id_'+str(i),round(out[0], 1)])
            writer.writerow(['id_'+str(i),out[0]])
    
    np.savetxt("weight_9.csv",W , delimiter=",")
    np.savetxt("bias_9.csv", B, delimiter=",")

    

if __name__ == "__main__":
    if len(sys.argv) ==4:  #training mode
        X_train,Y_train= read_Training_data(sys.argv[1],sys.argv[2])
        X_testing=read_Testing_data(sys.argv[3])
        training(10000,1e-3,X_train,Y_train,X_testing,0.01,0) 
    else:  #testing mode 
        W=np.array(pd.read_csv("weight_9.csv",header=None)) #18 feature
        B=np.array(pd.read_csv("bias_9.csv",header=None) )#18 feature
        X_testing=np.array(read_Testing_data(sys.argv[1]))
        #print(X_testing.shape,W.shape,B.shape)
        predict=np.dot(X_testing,W)+B

        with open(sys.argv[2], 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['id','value'])
            for i,out in enumerate(predict[0:500]) :
                writer.writerow(['id_'+str(i),out[0]])



