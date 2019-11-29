import sys

import numpy as np 
import torch
import torch.nn as nn
import pandas as pd
from torch import optim
from torch.utils.data import DataLoader, Dataset
from sklearn import manifold, datasets
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import SpectralClustering
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.cluster import FeatureAgglomeration,MeanShift

C=1
z_input=2048
hidden=12
before="before_t_0_5"
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(#32*32
          nn.Conv2d(3, 32*C, 3, 2, 1), #16*16
          nn.SELU(), 
          nn.Conv2d(32*C,64*C, 3, 2, 1),#8*8
          nn.SELU(),

          nn.Conv2d(64*C,128*C, 3, 2, 1),  
          nn.SELU()

        )
        # define: decoder
        self.decoder = nn.Sequential(
          nn.ConvTranspose2d( 128*C,64*C, 2, 2),
          nn.ConvTranspose2d(64*C,32*C, 2, 2),
          nn.ConvTranspose2d(32*C, 3, 2, 2),
          nn.Tanh()
        )
        self.linear_i=nn.Linear(z_input,hidden)
        self.linear_o=nn.Linear(hidden,z_input)
 
    def forward(self, x):
 
        encoded = self.encoder(x)
        encoded=self.linear_i(encoded.view(encoded.size(0), -1))
        #print(encoded.shape)
        hold=self.linear_o(encoded)
        decoded = self.decoder(hold.reshape(-1,128,4,4))
        return hold, decoded

    

    

def train_kmeans_tsne(train_dataloader,test_dataloader,autoencoder,Maxepoch):
    # We set criterion : L1 loss (or Mean Absolute Error, MAE)
    criterion = nn.MSELoss()#nn.L1Loss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
    
    # Now, we train 20 epochs.
    '''
    for epoch in range(Maxepoch):
 
        cumulate_loss = 0
        for x in train_dataloader:
            latent, reconstruct = autoencoder(x)
            loss = criterion(reconstruct, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            cumulate_loss = loss.item() * x.shape[0]
 
        print(f'Epoch { "%03d" % epoch }: Loss : { "%.5f" % (cumulate_loss / trainX.shape[0])}')
    '''
    autoencoder1= torch.load('model_best.pkl')
    #autoencoder2= torch.load('model_best2.pkl')
    
    # Collect the latents and stdardize it.
    latents = []
    reconstructs = []
    for x in test_dataloader:
 
        latent1, reconstruct1 = autoencoder1(x)
        #latent2, reconstruct2 = autoencoder2(x)
        latent=latent1#(latent1+latent2)/2
        reconstruct=reconstruct1#(reconstruct1+reconstruct2)/2
        latents.append(latent.cpu().detach().numpy())
        reconstructs.append(reconstruct.cpu().detach().numpy())
        
    reconstructs=np.concatenate(reconstructs, axis=0)
    reconstructs=np.transpose(reconstructs,(0,2,3,1))
    
    latents = np.concatenate(latents, axis=0).reshape([9000, -1])
    latents = (latents - np.mean(latents, axis=0)) / np.std(latents, axis=0)
    
 
    # Use PCA to lower dim of latents and use K-means to clustering.
    #print(latents.shape)
    #latents = PCA(n_components=32).fit_transform(latents)
    #latents =RandomTreesEmbedding(n_jobs=-1).fit(latents).labels_

    print("TSNE")
    
    # what the hell is tsne
    latents = PCA(n_components=32).fit_transform(latents)
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=8700)
    latents = tsne.fit_transform(latents)
    torch.save(autoencoder, 'model.pkl')  # 保存整个网络
    '''
    latents = TSNE(n_components =2).fit_transform(latents) #random_state=8700
    print(latents.shape)
    '''
    result = KMeans(n_clusters = 2).fit(latents).labels_
    print("KMeans")
    print(latents.shape)
    #result = MeanShift(bandwidth=2).fit(latents).labels_
    # We know first 5 labels are zeros, it's a mechanism to check are your answers
    # need to be flipped or not.
    if np.sum(result[:5]) >= 3:
        result = 1 - result
    
    return latents,result

if __name__ == '__main__':
 
    #==========global constants==========
    BATCH_SIZE = 32
    BETA = 1e-5
    RHO = 1e-3
    N_HIDDEN = 64*4*4
    use_sparse = False
    Maxepoch=40
    #print(rho)
    #####################################
    use_gpu = torch.cuda.is_available()
 
    autoencoder = Autoencoder()
    
    # load data and normalize to [-1, 1]
    trainX = np.load(sys.argv[1])#'trainX.npy'

    ar=np.array(trainX[5:10,:,:,:])
    
    trainX = np.transpose(trainX, (0, 3, 1, 2)) / 255. * 2 - 1
    print(trainX.shape)
    trainX = torch.Tensor(trainX)
    

    if use_gpu:
        autoencoder.cuda()
        trainX = trainX.cuda()
        device=torch.device('cuda')# 
    else:
        device=torch.device('cpu' )# 

        
 
    # Dataloader: train shuffle = True
    train_dataloader = DataLoader(trainX, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(trainX, batch_size=32, shuffle=False)
    RESULT=0
    for i in range(0,1):
        print("====Round_"+str(i)+"====")
        latents,result=train_kmeans_tsne(train_dataloader,test_dataloader,autoencoder,Maxepoch)
        df = pd.DataFrame({'id': np.arange(0, len(result)), 'label': result})
        RESULT+=result
        
    for element in RESULT:
        if element>=6: element=1
        else: element=0

    

    # Generate your submission
    df = pd.DataFrame({'id': np.arange(0, len(result)), 'label': result})
    df.to_csv(sys.argv[2],index=False)
   