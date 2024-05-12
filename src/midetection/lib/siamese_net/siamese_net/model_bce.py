import torch.nn as nn
import torch
from midetection.lib.siamese_net.siamese_net.config import batch_size
import tensorflow as tf
import numpy as np

train_on_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if train_on_gpu else "cpu")

# create a siamese network
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        self.cnn2d = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=1),                 # 95
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5, alpha=0.001, beta=0.75, k=2),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout(p=0.3),                                          # 47    
            nn.Conv2d(96, 128, kernel_size=5, stride=2, padding=1),     # 23
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5, alpha=0.001, beta=0.75, k=2),
            nn.MaxPool2d(3, stride=2),                                  # 11
            nn.Dropout(p=0.3),
        )

        self.cnn_2d = nn.Sequential(
            nn.Conv2d(batch_size, 64, kernel_size=3, stride=1),                 
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5, alpha=0.001, beta=0.75, k=2),
            nn.MaxPool2d(3, stride=2),    
            nn.Dropout(p=0.3),                          
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),    
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5, alpha=0.001, beta=0.75, k=2),
            nn.MaxPool2d(3, stride=2),                                  
            nn.Dropout(p=0.3),
            nn.Conv2d(32, batch_size, kernel_size=3, stride=1),    
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5, alpha=0.001, beta=0.75, k=2),
            nn.MaxPool2d(3, stride=2),                                  
            nn.Dropout(p=0.3),
        )
    
        self.final_fc = nn.Sequential(
            nn.Linear(389, 300),    
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(300, 150),    
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(150, 1)
        )
 
    def concat_moreFeatures(self, input, wM, mT, batchSize):
        # print("shape : " + str(input.shape))
        input = input.view(batch_size, -1)
        # print("shape after view: " + str(input.shape))
        output = torch.narrow(input, 0, 0, batchSize)
        output = torch.cat((output, wM, mT), dim=1)
        # print("shape after cat: " + str(output.shape))
 
        return output

    def concat_features(self, input1, input2):
        input1 = input1.cpu().detach().numpy()
        input2 = input2.cpu().detach().numpy()
        input_shape = input1.shape
        output = np.hstack((input1, input2))    # for each batch, stack the inputs side-by-side
        output_shape = output.shape
        
        output = torch.from_numpy(np.array(output, dtype=np.float32))
        return output.to(device)
   
    def forward_2d(self, x):
        output = self.cnn2d(x)
        output = output.view(output.size()[0], output.size()[1], -1)
        return output

    def forward_2dagain(self, x):
        output = self.cnn_2d(x)
        return output
      
    def forward(self, input1, input2, wallMotion, myocardialThickening):
        net1 = self.forward_2d(input1)
        net2 = self.forward_2d(input2)

        netComb = self.concat_features(net1, net2)
        current_batchsize = netComb.size()[0]
        # print("here " + str(netComb.shape))

        if (current_batchsize < batch_size) :
            # pad zeros
            pad = torch.zeros(batch_size-current_batchsize, netComb.size()[1], netComb.size()[2]).to(device)
            netComb = torch.cat((netComb, pad), dim=0)
        
        netComb = self.forward_2dagain(netComb)
        netComb = self.concat_moreFeatures(netComb, wallMotion, myocardialThickening, current_batchsize)
        netComb = self.final_fc(netComb)
            
        return netComb 