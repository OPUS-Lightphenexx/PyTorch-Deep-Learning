import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

Test_GPU = torch.cuda.is_available()
print(Test_GPU)
x = torch.randn(3,4)
print(x)
x_data = torch.randn(1,10)
print(x_data)
noise = torch.randn(x_data.size())
y_data = 2*x_data+1+0.5*noise
#定义一个线性回归神经网络
class LinearRegression(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(LinearRegression,self).__init__()
        self.predict = nn.Linear(input_dim,output_dim)

    def forward(self,x):
        y_predict = self.predict(x)
        return y_predict

Model = LinearRegression(10,10)
print(Model)
#优化损失函数
iteration = 100
LR = 0.08
opt = torch.optim.SGD(Model.parameters(),lr=LR)
criterion = nn.MSELoss()
#优化开始
iteration_list = []
loss_list = []
for epoch in range(iteration):
    y_pred = Model(x_data)
    loss = criterion(y_pred,y_data)
    print(epoch,loss.item())
    iteration_list.append(epoch)
    loss_list.append(loss.item())

    opt.zero_grad()
    loss.backward()
    opt.step()

plt.plot(iteration_list,loss_list)
plt.title('Optimizing The Loss Function(Linear) Using SGD')
plt.xlabel('Iteration')
plt.grid()
plt.show()

#
class Sigmoid(nn.Module):
    def __init__(self):
        super(Sigmoid,self).__init__()
        self.predict = nn.Sigmoid()

    def forward(self,x):
        y_predict = self.predict(x)
        return y_predict

Model1 = Sigmoid()
print(Model1)
#优化损失函数
iteration = 500
LR = 0.01
opt = torch.optim.SGD(Model.parameters(),lr=LR)
criterion = nn.MSELoss()
#优化开始
iteration_list = []
loss_list = []
for epoch in range(iteration):
    y_pred = Model(x_data)
    loss = criterion(y_pred,y_data)
    print(epoch,loss.item())
    iteration_list.append(epoch)
    loss_list.append(loss.item())

    opt.zero_grad()
    loss.backward()
    opt.step()

plt.plot(iteration_list,loss_list)
plt.title('Optimizing The Loss Function(Sigmoid) Using SGD')
plt.xlabel('Iteration')
plt.grid()
plt.show()


#运行模型

