import torch
import torch.nn as nn

Test_GPU = torch.cuda.is_available()
print(Test_GPU)
x = torch.randn(3,4)
print(x)

#定义一个线性回归神经网络
class LinearRegression(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(LinearRegression,self).__init__()
        self.predict = nn.Linear(input_dim,output_dim)

    def forward(self,x):
        result = self.predict(x)
        return result

Model = LinearRegression(1,1)
print(Model)
#优化损失函数
iteration = 1000
LR = 0.01
opt = torch.optim.SGD(Model.parameters(),lr=LR)
criterion = nn.MSELoss()

#开始优化
#for i in range(iteration):



#运行模型
x_test = torch.randn(1,10)
noise = torch.randn(x_test.size())
y_test = 2*x_test+1+0.1*noise
print(x_test)
