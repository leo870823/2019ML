import math
import numpy as np
import pandas as pd
import sys
# Only Needed on Google Colab
import csv

def readdata(data):
    
	for col in list(data.columns[2:]):
		data[col] = data[col].astype(str).map(lambda x: x.rstrip('x*#A'))
	data = data.values
	
	
	data = np.delete(data, [0,1], 1)
	

	data[ data == 'NR'] = 0
	data[ data == ''] = 0
	data[ data == 'nan'] = 0
	data = data.astype(np.float)

	return data

def extract(data):
	N = data.shape[0] // 18

	temp = data[:18, :]
    
    
	for i in range(1, N):
		temp = np.hstack((temp, data[i*18: i*18+18, :]))
	return temp

def valid(x, y):
	if y <= 2 or y > 100:
		return False
	for i in range(9):
		if x[9,i] <= 2 or x[9,i] > 100:
			return False
	return True

def parse2train(data):
	x = []
	y = []
	
	
	total_length = data.shape[1] - 9
	for i in range(total_length):
		x_tmp = data[:,i:i+9]
		y_tmp = data[9,i+9]
		if valid(x_tmp, y_tmp):
			x.append(x_tmp.reshape(-1,))
			y.append(y_tmp.reshape(-1,))
	
	x = np.array(x)
	y = np.array(y)
	return x,y

def parse2test(data):
    x=[]
    total_length = data.shape[1]//9
    for i in range(total_length):
        x_tmp=data[:,i*9:(i+1)*9]
        x.append(x_tmp.reshape(-1,))
    x=np.array(x)
    return x

def data_pre_processing(file):

    pre_data=pd.read_csv(file) #18 feature
    pre_data = pre_data.fillna(0) # fill 0
    pre_data = pre_data.replace(['\A','\#','\*','x','NR'], ['','','','','0'], regex=True)
    data=pre_data.values
    data=data[:,2:].astype("float")
    print("Before data_pre_processing:",data.shape[0],data.shape[1])
    day=np.vsplit(data,data.shape[0]/18) # x * shape(18,9) 
    year=np.concatenate(np.array(day),axis=1)
    print("After data_pre_processing:",year.shape)
    return year


def read_Testing_data(file):
    day=data_pre_processing(file) 
    X=[]
    for i in range(0,day.shape[1],9):
        X.append(day[:,i:i+9].flatten()) 
    print("Testing data shape:",np.array(X).shape)
    return np.array(X)





def minibatch(x, y):
    
    index = np.arange(x.shape[0])
    np.random.shuffle(index)
    x = x[index]
    y = y[index]
    
    
    batch_size = 64
    lr = 1e-3
    lam = 0.001
    beta_1 = np.full(x[0].shape, 0.9).reshape(-1, 1)
    beta_2 = np.full(x[0].shape, 0.99).reshape(-1, 1)
    w = np.full(x[0].shape, 0.1).reshape(-1, 1)
    bias = 0.1
    m_t = np.full(x[0].shape, 0).reshape(-1, 1)
    v_t = np.full(x[0].shape, 0).reshape(-1, 1)
    m_t_b = 0.0
    v_t_b = 0.0
    t = 0
    epsilon = 1e-8
    
    for num in range(1000):
        for b in range(int(x.shape[0]/batch_size)):
            t+=1
            x_batch = x[b*batch_size:(b+1)*batch_size]
            y_batch = y[b*batch_size:(b+1)*batch_size].reshape(-1,1)
            loss = y_batch - np.dot(x_batch,w) - bias
            
            
            g_t = np.dot(x_batch.transpose(),loss) * (-2) +  2 * lam * np.sum(w)
            g_t_b = loss.sum(axis=0) * (2)
            m_t = beta_1*m_t + (1-beta_1)*g_t 
            v_t = beta_2*v_t + (1-beta_2)*np.multiply(g_t, g_t)
            m_cap = m_t/(1-(beta_1**t))
            v_cap = v_t/(1-(beta_2**t))
            m_t_b = 0.9*m_t_b + (1-0.9)*g_t_b
            v_t_b = 0.99*v_t_b + (1-0.99)*(g_t_b*g_t_b) 
            m_cap_b = m_t_b/(1-(0.9**t))
            v_cap_b = v_t_b/(1-(0.99**t))
            w_0 = np.copy(w)
            
            
            w -= ((lr*m_cap)/(np.sqrt(v_cap)+epsilon)).reshape(-1, 1)
            bias -= (lr*m_cap_b)/(math.sqrt(v_cap_b)+epsilon)
            

    return w, bias


def minibatch_my(x, y):


    print(x.shape,y.shape)
    



    valid_r=0
    hold=np.concatenate((x,y),axis=1)

    

    _val_length=int((1-valid_r)*x.shape[0])

    X_train=hold[0:_val_length,0:x.shape[1]]
    Y_train=hold[0:_val_length,x.shape[1]:]

    val_X_train=hold[_val_length:,0:x.shape[1]]
    val_Y_train=hold[_val_length:,x.shape[1]:]





    index = np.arange(X_train.shape[0])
    np.random.shuffle(index)
    X_train = X_train[index]
    Y_train = Y_train[index]


    batch_size = 64
    lr = 1e-3
    lam = 0.001
    beta_1 = np.full(x[0].shape, 0.9).reshape(-1, 1)
    beta_2 = np.full(x[0].shape, 0.99).reshape(-1, 1)
    w = np.full(X_train.shape[1], 0.1).reshape(-1, 1)
    bias = 0.1
    print("W matrix shape:",w.shape)
    print("B matrix value:",bias)

    m_t = np.full(x[0].shape, 0).reshape(-1, 1)
    v_t = np.full(x[0].shape, 0).reshape(-1, 1)
    m_t_b = 0.0
    v_t_b = 0.0
    t = 0
    epsilon = 1e-8

    for epoch in range(1000):
        for b in range(int(X_train.shape[0]/batch_size)):
            t+=1
            x_batch = X_train[b*batch_size:(b+1)*batch_size]
            y_batch = Y_train[b*batch_size:(b+1)*batch_size].reshape(-1,1)
            loss = y_batch - np.dot(x_batch,w) - bias
            
            g_t = np.dot(x_batch.transpose(),loss) * (-2) +  2 * lam * np.sum(w)
            g_t_b = loss.sum(axis=0) * (2)
            m_t = beta_1*m_t + (1-beta_1)*g_t 
            v_t = beta_2*v_t + (1-beta_2)*np.multiply(g_t, g_t)
            m_cap = m_t/(1-(beta_1**t))
            v_cap = v_t/(1-(beta_2**t))
            m_t_b = 0.9*m_t_b + (1-0.9)*g_t_b
            v_t_b = 0.99*v_t_b + (1-0.99)*(g_t_b*g_t_b) 
            m_cap_b = m_t_b/(1-(0.9**t))
            v_cap_b = v_t_b/(1-(0.99**t))
            w_0 = np.copy(w)
            
           
            w -= ((lr*m_cap)/(np.sqrt(v_cap)+epsilon)).reshape(-1, 1)
            bias -= (lr*m_cap_b)/(math.sqrt(v_cap_b)+epsilon)
        if epoch %20==0: 
            
                print("Epoch",epoch)
                loss= Y_train - np.dot(X_train,w) - bias
                print("Training Loss:",np.power(np.sum(np.power(loss,2)/_val_length),0.5))
                if valid_r>0:
                    val_X_train_length=val_X_train.shape[0]
                    val_predict=np.dot(val_X_train,w)+bias
                    val_loss=val_Y_train-val_predict
                    val_mse=np.power(np.sum(np.power(val_loss,2))/val_X_train_length,0.5)
                    print("Testing Loss:",val_mse)


    return w, bias











if __name__ == "__main__":
    if len(sys.argv)==3:
        W=np.array(pd.read_csv("weight.csv",header=None)) #18 feature
        B=np.array(pd.read_csv("bias.csv",header=None) )#18 feature
        X_testing=np.array(read_Testing_data(sys.argv[1]))
        print(X_testing.shape,W.shape,B.shape)
        predict=np.dot(X_testing,W)+B

        with open(sys.argv[2], 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['id','value'])
            for i,out in enumerate(predict[0:500]) :
                writer.writerow(['id_'+str(i),out[0]])
        
    else:
        year1_pd = pd.read_csv(sys.argv[1])
        year1 = readdata(year1_pd)

    
        year2_pd = pd.read_csv(sys.argv[2])
        year2 = readdata(year2_pd)    

    
        year=np.concatenate((year1,year2))

        train_data = extract(year)
        train_x, train_y = parse2train(train_data)

        #TA's way process training data 
        test_pd = pd.read_csv(sys.argv[3])
        test_pd = readdata(test_pd)
        test_x=extract(test_pd)
        test_x=parse2test(test_x) 

        #my way process training data 
        test_x_my=  read_Testing_data(sys.argv[3])

        #np.savetxt("test_data_my.csv", test_x_my.reshape(-1,), delimiter=",")
        #np.savetxt("test_data_TA.csv", test_x.reshape(-1,), delimiter=",")

        w, bias = minibatch(train_x, train_y)
        loss= train_y - np.dot(train_x,w) - bias
        print("Training Loss:",np.power( np.sum(np.power(loss,2) )/train_x.shape[0],0.5 ))
        predict=np.dot(test_x,w)+bias

        np.savetxt("weight.csv",w , delimiter=",")
        np.savetxt("bias.csv", bias, delimiter=",")



        with open('output_my_test.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['id','value'])
            for i,out in enumerate(predict) :
                writer.writerow(['id_'+str(i),out[0]])

