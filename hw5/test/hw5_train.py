#standard
#script dowland en.model!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
import os
import csv
import sys
from multiprocessing import Pool
#optional
import pandas as pd
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from gensim.models import Word2Vec
import spacy
import numpy as np
nlp= spacy.load('en_core_web_sm')
#nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader ,sampler
'''
#plot
import matplotlib.pyplot as plt
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
'''
def eva(model,filename,test_loader,best=False):
    if best:
        model.load_state_dict(torch.load('./model/model_W2V.pkl'))
    model.eval()
    prediction=[]
    with torch.no_grad():
        for batch_idx,img in enumerate(test_loader):
            img=img.to(device)
            out=model(img)
            index,pred_label =torch.max(out,1)
            for x in pred_label.data:
                prediction.append(x.item())
    with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['id','label'])
            for i,out in enumerate(prediction) :
                writer.writerow([str(i),out])








class Preprocess():
    def __init__(self, data_dir, label_dir):
        self.tokenizer = nlp.Defaults.create_tokenizer(nlp)
        self.index2word = []
        self.word2index = {}
        self.vectors = []
        self.embed_dim=100
        self.seq_max=30
        if data_dir!=None:
            # Read data
            print("Read data!")
            dm = pd.read_csv(data_dir)#[0:100]
            data = dm['comment']
            # Tokenize with multiprocessing
            # List in list out with same order
            # Multiple workers
            print("Multiple Process!")
            P = Pool(processes=4) 
            data = P.map(self.tokenize, data)#tokennize
            #print(data)
            P.close()
            P.join()
            self.data = data
            #print(type(data))
        if label_dir!=None:
            # Read Label
            dm = pd.read_csv(label_dir)#[0:100]
            self.label = [int(i) for i in dm['label']]

    def tokenize(self, sentence):
        """ 
        Args:
            sentence (str): One string.
        Return:
            tokens (list of str): List of tokens in a sentence.
        """
        tokens=[ ]
        doc =  nlp(sentence)
        for t in doc :
            if t.text!="@user" :tokens.append(t.text) #or t.text!="URL"
        return tokens
    def Embedding_Word2Vector(self,load=False):
        print("===Embedding_Word2Vector_with_gensim===")
        
        if load:
            embed = Word2Vec.load("W2V")
        else:
            embed = Word2Vec(self.data, size=self.embed_dim, window=5, min_count=5, iter=1000, workers=8)#cython
            embed.save("W2V")
        # Create word2index dictinonary
        # Create index2word list
        # Create word vector list
        for i, word in enumerate(embed.wv.vocab):
            print('=== get words #{}'.format(i+1), end='\r')
            #e.g. self.word2index['魯'] = 1 
            #e.g. self.index2word[1] = '魯'
            #e.g. self.vectors[1] = '魯' vector
            self.word2index[word] = len(self.word2index)
            self.index2word.append(word)
            self.vectors.append(embed[word])
        self.vectors = torch.tensor(self.vectors)
        # Add special tokens
        self.add_embedding("PAD")
        self.add_embedding("UNKNOWN")
        print("=== total words: {}".format(len(self.vectors)))
        return self.vectors
    def add_embedding(self, word):
        # Add random uniform vector
        vector = torch.empty(1, self.embed_dim)
        torch.nn.init.uniform_(vector)
        self.word2index[word] = len(self.word2index)
        self.index2word.append(word)
        self.vectors = torch.cat([self.vectors, vector], 0)
    def get_indices(self,test=False):
        # Transform each words to indices
        # e.g. if 機器=0,學習=1,好=2,玩=3 
        # [機器,學習,好,好,玩] => [0, 1, 2, 2,3]
        all_indices = []
        # Use tokenized data
        for i, sentence in enumerate(self.data):
            print('=== sentence count #{}'.format(i+1), end='\r')
            sentence_indices = []
            for word in sentence:
                # if word in word2index append word index into sentence_indices
                if word in self.word2index:
                    sentence_indices.append(self.word2index[word])
                else:
                # if word not in word2index append unk index into sentence_indices
                    sentence_indices.append(self.word2index["UNKNOWN"])
            # pad all sentence to fixed length
            sentence_indices = self.pad_to_len(sentence_indices, self.seq_max, self.word2index["PAD"])
            all_indices.append(sentence_indices)
        if test:
            return torch.LongTensor(np.array(all_indices))         
        else:
            return torch.LongTensor(np.array(all_indices)), self.label #torch.LongTensor(
    def pad_to_len(self, arr, padded_len, padding=0):
        """ 
        if len(arr) < padded_len, pad arr to padded_len with padding.
        If len(arr) > padded_len, truncate arr to padded_len.
        Example:
            pad_to_len([1, 2, 3], 5, 0) == [1, 2, 3, 0, 0]
            pad_to_len([1, 2, 3, 4, 5, 6], 5, 0) == [1, 2, 3, 4, 5]
        Args:
            arr (list): List of int.
            padded_len (int)
            padding (int): Integer used to pad.
        Return:
            arr (list): List of int with size padded_len.
        """
        step=padded_len-len(arr)
        if step>=0:
            for i in range(abs(step)):
                arr.append(padding)
        else:
            for i in range(abs(step)):
                arr.pop(-1)
        return arr


def train(model,lr,MAX_EPOCH,train_loader,val_loader):
    if not os.path.exists("./model"):
        os.mkdir("./model")
    model.train()
    optimizer =torch.optim.Adam(model.parameters(),lr)
    train_draw,val_draw=[],[]
    train_draw_loss,val_draw_loss=[],[]
    best_validation_accuracy=0.0
    #model=model.to(device)
    for epoch in range(MAX_EPOCH):
        train_nums,val_nums=0,0
        train_loss,val_loss=0,0
        train_acc,val_acc=0.0,0.0
        
        for batch_idx,(data,label) in enumerate(train_loader):
            data,label =data.to(device),label.to(device)
            optimizer.zero_grad()
            output =model(data)
            #print(output.shape,data.shape)
            loss =F.cross_entropy(output,label)
            train_loss +=F.cross_entropy(input=output,target=label,size_average=False).item()
        
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
                    val_loss +=F.cross_entropy(input=output,target=label,size_average=False).item()
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

                train_draw_loss.append( train_loss/train_nums)
                val_draw_loss.append(val_loss/val_nums)
    #PATH = './model/model_train:{}_val:{}.pth'.format(train_acc/train_nums,val_acc/val_nums)
        
        if  val_acc/val_nums> best_validation_accuracy:
            PATH = './model/model_W2V.pkl'
            best_validation_accuracy = val_acc/val_nums
            torch.save(model.state_dict(), PATH)
            print("New best found and saved.")
        
    #index=np.array(range(0,MAX_EPOCH)).flatten()
    #draw_result(range(0,MAX_EPOCH),train_draw_loss ,val_draw_loss , "Model Loss")
    #draw_result(range(0,MAX_EPOCH),train_draw ,val_draw , "Model accuracy")




def orthogonal_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('GRU') != -1 : #and classname.find('Conv') == 0
        #nn.init.orthogonal(m.weight)
        print("Initialization GRU!!")
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)
    

class BiRNN(nn.Module):
    def __init__(self, embedding, embedding_dim=100, hidden_dim=100, num_layers=2, dropout=0.5, fix_emb=True):
        super(BiRNN, self).__init__()
        # Create embedding layer
        self.embedding = torch.nn.Embedding(embedding.size(0),embedding.size(1))
        self.embedding.weight = torch.nn.Parameter(embedding)
        # Fix/Train embedding 
        self.embedding.weight.requires_grad = False if fix_emb else True
        self.embedding_dim = embedding.size(1)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.GRU(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        #self.lstm.apply(orthogonal_weights_init)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim*2, 128),
            nn.BatchNorm1d(128),
            nn.SELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(128,16),
            nn.BatchNorm1d(16),
            nn.SELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(16,2),
            
            nn.Sigmoid())
    def forward(self, inputs):
        inputs = self.embedding(inputs)
        x, _ = self.lstm(inputs, None) 
        #x, _ = self.lstm(x, None)
        # x dimension(batch, seq_len, hidden_size)
        # Use LSTM last hidden state (maybe we can use more states)
        x = x[:, -1, :] # (batch_size, seq_length, hidden_size*2)
        x = self.classifier(x)
        return x

class hw5_loader(TensorDataset):
 def __init__(self,data,label):
     self.data=data
     self.label=label
     print("Number of data:",len(self.data))
 def __getitem__(self,index):
     #print(index)
     #print(self.data[index],self.label[index])
     if self.label:
        return  self.data[index], self.label[index]
     else:
        return self.data[index]
         
 def __len__(self):
     return len(self.data)

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
    
if __name__ == "__main__":
    '''
    Data load
    '''
    Data_keeper=Preprocess(sys.argv[1],sys.argv[2])
    embedding=Data_keeper.Embedding_Word2Vector(False)
    data, label = Data_keeper.get_indices(False)
    train_dataset=hw5_loader(data, label)

    #=====parameter=====
    _val_rate=0.2
    TOTAL_LABEL=train_dataset.__len__()
    NUM_TRAIN  = int(TOTAL_LABEL-_val_rate*TOTAL_LABEL)
    NUM_VAL    = int(_val_rate*TOTAL_LABEL)
    #===================
    print("=======Data load=======")
    #train
    train_loader=torch.utils.data.DataLoader(train_dataset,
                                            batch_size=256,
                                            sampler=ChunkSampler(NUM_TRAIN, 0)
                                             )
    val_loader=torch.utils.data.DataLoader(train_dataset,
                                            batch_size=NUM_VAL,
                                            sampler=ChunkSampler(NUM_VAL, NUM_TRAIN) )

    '''''
    Model
    '''''
    use_cuda=torch.cuda.is_available()
    if use_cuda:
        device=torch.device('cuda')# 
    else :
        device=torch.device('cpu' )# 
    
    #model=LSTM_Net(embedding)
    model=BiRNN(embedding)
    model=model.to(device)
    train(model,1e-3,100,train_loader,val_loader)
    '''''
    Evaluate
    '''''
    #test
    Data_keeper_test=Preprocess( sys.argv[3], None) 
    Data_keeper_test.Embedding_Word2Vector(True)
    data= Data_keeper_test.get_indices(True)
    test_dataset= hw5_loader(data, None)
    test_loader=torch.utils.data.DataLoader(test_dataset,
                                            batch_size=test_dataset.__len__()
                                            )
    eva(model,"ans.csv",test_loader,best=True) #sys.argv[4]


