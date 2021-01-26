from random import shuffle
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

class Phi(nn.Module):
    def __init__(self):
        super(Phi, self).__init__()                                               #Phi is the encoder network.
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3,3), stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=2, padding=1)
    def forward(self,x):
        x = F.normalize(x)
        y = F.elu(self.conv1(x))
        y = F.elu(self.conv2(y))
        y = F.elu(self.conv3(y))
        y = F.elu(self.conv4(y)) #size [1, 32, 3, 3] batch, channels, 3 x 3
        y = y.flatten(start_dim=1) #size N, 512
        return y
class Gnet(nn.Module):
    def __init__(self):                                                            #Gnet is the inverse model.
        super(Gnet, self).__init__()
        self.linear1 = nn.Linear(1024,256)
        self.linear2 = nn.Linear(256,15)
    def forward(self, state1,state2):
        x = torch.cat( (state1, state2) ,dim=1)
        y = F.relu(self.linear1(x))
        y = self.linear2(y)
        y = F.softmax(y,dim=1)
        return y
class Fnet(nn.Module):
    def __init__(self):                                                            #Fnet is the forward model.
        super(Fnet, self).__init__()
        self.linear1 = nn.Linear(527,256)
        self.linear2 = nn.Linear(256,512)
    def forward(self,state,action):
        action_ = torch.zeros(action.shape[0],15).cuda()   #The actions are stored as integers in the replay memory, so we convert to a one-hot encoded vector.
        indices = torch.stack( (torch.arange(action.shape[0]).cuda(),
                                action.squeeze().cuda()), dim=0)
        indices = indices.tolist()
        action_[indices] = 1.
        x = torch.cat( (state,action_) ,dim=1)
        y = F.relu(self.linear1(x))
        y = self.linear2(y)
        return y
