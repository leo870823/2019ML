import cv2
import sys
import csv
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import os 
from torch.utils.data import TensorDataset, DataLoader ,sampler
import cv2
import matplotlib.pyplot as plt
from PIL import Image

def draw_result(lst_iter, lst_loss, lst_acc, title):
    plt.plot(lst_iter, lst_loss, '-b', label='Training')
    plt.plot(lst_iter, lst_acc, '-r', label='Validation')

    plt.xlabel("Epoch")
    plt.legend(loc='upper left')
    plt.title(title)

    # save image
    plt.savefig(title+".png")  # should before show method

    # show
    plt.close()#show()




class train_hw3(TensorDataset):
 def __init__(self,data_dir,label):
     self.data_dir=data_dir
     self.label=label
     #print(label.shape)

 def __getitem__(self,index):
     pic_file='{:05d}.jpg'.format(self.label[index,0])
     img =cv2.imread(os.path.join(self.data_dir,pic_file),cv2.IMREAD_GRAYSCALE) #/ 255.0
     img =np.expand_dims(img,axis=0)
     #img=torch.FloatTensor(img)
     #img =torch.stack([img,img,img],dim=0)
     return torch.FloatTensor(img), self.label[index,1]
 def __len__(self):
     return len(self.label)

class test_hw3(TensorDataset):
 def __init__(self,data_dir,label):
     self.data_dir=data_dir
     self.label=label
 def __getitem__(self,index):
     pic_file='{:04d}.jpg'.format(self.label[index])
     img =cv2.imread(os.path.join(self.data_dir,pic_file),cv2.IMREAD_GRAYSCALE) #/ 255.0
     img =np.expand_dims(img,axis=0)
     #img = torch.as_tensor(img)
     #img = Image.open(os.path.join(self.data_dir,pic_file))
     return torch.FloatTensor(img), self.label[index]
 def __len__(self):
     return len(self.label)

class ChunkSampler(sampler.Sampler):
    """Samples elements sequentially from some offset. 
    Arguments:
        num_samples: # of desired datapoints
        start: offset where we should start selecting from
    """
    def __init__(self, num_samples, start = 0):
        self.num_samples = num_samples
        self.start = start
        

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples


def read_Train(file):
    data=pd.read_csv(file) #18 feature
    return np.array(data)
def train(device,model,optimizer,MAX_EPOCH,train_loader,val_loader):
    train_draw,val_draw=[],[]
    train_draw_loss,val_draw_loss=[],[]
    best_validation_accuracy=0.0
    model=model.to(device)
    for epoch in range(MAX_EPOCH):
        model.train()
        train_nums,val_nums=0,0
        train_loss,val_loss=0,0
        train_acc,val_acc=0.0,0.0
        
        for batch_idx,(data,label) in enumerate(train_loader):
            
            data,label =data.to(device),label.to(device)
            optimizer.zero_grad()
            output =model(data)
            #print(output)
            loss =F.cross_entropy(output,label)
            train_loss +=F.cross_entropy(output,label).item()
        
            #print('Epoch: ', epoch, '| batch: ', batch_idx,'| loss: ',train_loss)
            loss.backward()
            optimizer.step()
            _,pred_label=torch.max(output,1)
            train_acc+=sum(label==pred_label).item()
            train_nums+=pred_label.size(0)
        #validation part
    
        with torch.no_grad():
                for batch_idx,(data,label) in enumerate(val_loader):
                    data,label =data.to(device),label.to(device)
                    output =model(data)
                    val_loss +=F.cross_entropy(output,label).item()
                    _,pred_label=torch.max(output,1)
                    val_acc+=sum(pred_label==label).item()
                    #print(pred_label.size(0))
                    val_nums+=pred_label.size(0)
    
                print("Epoch:{}".format(epoch))
                print("Training loss:{}".format(train_loss),"Training Accuracy:{}".format(train_acc/train_nums ) )
                print("Testing  loss:{}".format(val_loss),  "Testing  Accuracy:{}".format(val_acc/val_nums ) )
                print("val_acc{},val_nums{}".format(val_acc,val_nums) )
                train_draw.append(train_acc/train_nums )
                val_draw.append(val_acc/val_nums)

                train_draw_loss.append( train_loss)
                val_draw_loss.append(val_loss)
    #PATH = './model/model_train:{}_val:{}.pth'.format(train_acc/train_nums,val_acc/val_nums)
        
        if  val_acc/val_nums> best_validation_accuracy:
            PATH = './model/model_draw.pkl'
            best_validation_accuracy = val_acc/val_nums
            torch.save(model.state_dict(), PATH)
            print("New best found and saved.")
        
    #index=np.array(range(0,MAX_EPOCH)).flatten()
    draw_result(range(0,MAX_EPOCH),train_draw_loss ,val_draw_loss , "Model Loss")
    draw_result(range(0,MAX_EPOCH),train_draw ,val_draw , "Model accuracy")
    '''
    PATH = './model/model_draw.pkl'
    torch.save(model.state_dict(), PATH)
    '''

def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 : #and classname.find('Conv') == 0
        m.weight.data.normal_(0.0, 0.02)
    
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)



class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.cnn = nn.Sequential(
            #input size (48 **2)
            nn.Conv2d(in_channels=1,out_channels= 64, kernel_size=4, stride=2, padding=1),  # [64, 24, 24]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            #input size ( 23=(48+2-4+1)/2 **2)
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2, 0),      # [64, 12, 12]
            #input size(12)
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2, 0),      # [128, 6, 6]
            #input size(6)
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            #input size(3)
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2, 0)       # [256, 3, 3]
        )

        self.fc = nn.Sequential(
            nn.Linear(256*3*3, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.5),

            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.5),
            nn.Linear(512, 7)
            
            #nn.Linear(512,256),
            #nn.LeakyReLU(0.2),
            #nn.Dropout(p=0.5),
        

        )

        self.cnn.apply(gaussian_weights_init)
        self.fc.apply(gaussian_weights_init)

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(-1,256*3*3)
        return self.fc(out)







if __name__ == "__main__":
    '''''
    Data
    '''''
    '''
    _val_rate=0.1
    train_label=read_Train(sys.argv[2]) 
    TOTAL_LABEL=len(train_label)
    NUM_TRAIN  = int(TOTAL_LABEL-_val_rate*TOTAL_LABEL)
    NUM_VAL    = int(_val_rate*TOTAL_LABEL)

    train_dataset=train_hw3(sys.argv[1],train_label)
    train_loader=torch.utils.data.DataLoader(train_dataset,
                                            batch_size=256,
                                            #shuffle=True,
                                            sampler=ChunkSampler(NUM_TRAIN, 0)
                                             )
    val_loader=torch.utils.data.DataLoader(train_dataset,
                                            batch_size=NUM_VAL,
                                            #shuffle=True,
                                            sampler=ChunkSampler(NUM_VAL, NUM_TRAIN) )
    '''    
    test_dataset=test_hw3 ( sys.argv[1], range(0,7000)) 
    test_loader=torch.utils.data.DataLoader(test_dataset)
    '''''
    Model
    '''''
    use_cuda=torch.cuda.is_available()
    if use_cuda:
        device=torch.device('cuda')# 
    else :
        device=torch.device('cpu' )# 
   
    '''
    coding=UTF-8
    import torchvision.models as models  
    #呼叫模型  

    model = models.resnet50(pretrained=True)  
    model=model.to(device)
    #提取fc層中固定的引數  
    fc_features = model.fc.in_features  
    #修改類別為9  
    model.fc = nn.Linear(fc_features, 7)  
    '''

    '''
    model=Classifier()#ConvNet()
    model=model.to(device)
    optimizer =torch.optim.Adam(model.parameters(),lr=1e-4)
    train(device,model,optimizer,100,train_loader,val_loader)
    '''
    '''''
    Evaluation
    '''''


    model=Classifier()
    model=model.to(device)
    model.load_state_dict(torch.load('my_best.pkl'))
    model.eval()

    prediction=[]
    with torch.no_grad():
        for batch_idx,(img,index) in enumerate(test_loader):
            img=img.to(device)
            out=model(img)
            index,pred_label =torch.max(out,1)
            prediction.append(pred_label.item())
    with open(sys.argv[2], 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['id','label'])
            for i,out in enumerate(prediction) :
                writer.writerow([str(i),out])
    

    
    


'''
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6,kernel_size=(5,5),stride=(1,1))
        self.conv2 = nn.Conv2d(6,16,kernel_size=(5,5),stride=(1,1))
        self.fc1 = nn.Linear(in_features=1296,out_features=120,bias=True)
        self.fc2 = nn.Linear(in_features=120,out_features=84,bias=True)
        self.fc3 = nn.Linear(in_features=84,out_features=7,bias=True)

    def forward(self, x):
        x =F.relu(F.max_pool2d(self.conv1(x),2))
        x =F.relu(F.max_pool2d(self.conv2(x),2))
        #print(x.shape)
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        return   out
'''