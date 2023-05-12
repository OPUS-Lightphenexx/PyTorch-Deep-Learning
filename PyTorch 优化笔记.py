import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

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
        self.hidden1 = nn.Linear(input_dim,output_dim)

    def forward(self,x):
        y_predict = self.hidden1(x)
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


x_test = torch.linspace(0,2,10)
#-----------------------------------------------------------
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


#笔记
import numpy as np
import matplotlib.pyplot as plt
import torch as t
def get_data(x, w, b, d):
    c, r = x.shape
    y = (w*x + b * x**2 + d) + (1.3 * (2 * np.random.rand(c, r) - 1))
    return (y)


xs = np.arange(0, 3, 0.01).reshape(-1, 1)
ys = get_data(xs, 1, -2, 3)

xs = t.Tensor(xs)
ys = t.Tensor(ys)


class Fit_model(t.nn.Module):
    def __init__(self):
        super(Fit_model, self).__init__()
        self.linear1 = t.nn.Linear(1, 16)
        self.relu = t.nn.ReLU()
        self.linear2 = t.nn.Linear(16, 1)

        self.criterion = t.nn.MSELoss()
        self.opt = t.optim.SGD(self.parameters(), lr=0.05)

    def forward(self, input):
        y = self.linear1(input)
        y = self.relu(y)
        y = self.linear2(y)
        return y


model = Fit_model()
iteration_list1 = []
loss_list1 = []
for e in range(2000):
    y_pre = model(xs)

    loss1 = model.criterion(y_pre, ys)

    # Zero gradients
    model.opt.zero_grad()
    # perform backward pass
    loss1.backward()
    model.opt.step()
    iteration_list1.append(e)
    loss_list1.append(loss1.item())

ys_pre = model(xs)

plt.title("Fit Curve Using Neural Network")
plt.scatter(xs.data.numpy(), ys.data.numpy())
plt.plot(xs.data.numpy(), ys_pre.data.numpy(),color='red')
#--------------------------------------------------------------------------------------------------
plt.grid()
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
plt.plot(iteration_list1,loss_list1)
plt.title('Optimizing the Loss Function')
plt.grid()
plt.show()


print(ys_pre)

