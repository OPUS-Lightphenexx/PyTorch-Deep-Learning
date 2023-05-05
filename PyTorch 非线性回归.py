import torch
import torch.nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

#给定原始数据
x_data = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
noise = torch.randn(x_data.size())

y_sample = x_data.pow(3) + 1 + 0.1 * noise

#搭建神经网络
class Neural_net(torch.nn.Module):
    def __init__(self,n_features,hidden,output):
        super(Neural_net,self).__init__()
        self.hidden = torch.nn.Linear(n_features,hidden)
        self.predict = torch.nn.Linear(hidden,output)


    def forward(self, x):
        h1 = self.hidden(x)
        s1 = F.relu(h1)
        out = self.predict(s1)
        return out


model = Neural_net(1,10,1)
y_pred = model.forward(x_data)


