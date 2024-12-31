import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.utils.data as Data
from torchvision import transforms, datasets
import time
from pyswarms.single.global_best import GlobalBestPSO
# from turtle import forward
import torch.nn as nn
import torch.nn.functional as F


def wash_data(df: pd.DataFrame, flag):
    """箱型图法"""
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    iqr = q3 - q1
    mi = q1 - 1.5 * iqr
    ma = q3 + 1.5 * iqr
    # error = df[(df < mi) | (df > ma)]
    data_c = df[(df >= mi) & (df <= ma)]
    if (flag == 0):
        pd = data_c.bfill().fillna(method='ffill')
    if (flag == 1):
        pd = data_c.interpolate().bfill()
    if (flag == 3):
        pd = data_c.interpolate(method='spline', order=3).bfill()  # (method='spline', order=3)#.bfill()  # sanci插值
    return pd


def split_wash_data(df, n, flag):
    cleaned_splits = []
    cleaned_df = []
    for i in range(0, len(df), n):
        split = df[i:i + n]
        cleaned_split = wash_data(split, flag)
        cleaned_splits.append(cleaned_split)
        cleaned_df = pd.concat(cleaned_splits, ignore_index=True)
    return pd.DataFrame(cleaned_df)


# 文件读取
def get_Data(data_path):
    data = pd.read_excel(data_path)
    # data = wash_data(data)
    #data = data.iloc[:int(len(data_path)*5/6), :3]  # 以三个特征作为数据
    data = split_wash_data(data, 100, 0)
    data = split_wash_data(data, 20, 1)
    data = data.iloc[:, :3]
    label = data.iloc[:, 2:]  # 取最后一个特征作为标签
    return data, label


# 数据预处理
def normalization(data, label):
    mm_x = MinMaxScaler()  # 导入sklearn的预处理容器
    mm_y = MinMaxScaler()
    data = data.values  # 将pd的系列格式转换为np的数组格式
    label = label.values
    data = mm_x.fit_transform(data)  # 对数据和标签进行归一化等处理
    label = mm_y.fit_transform(label)
    return data, label, mm_y


# 时间向量转换
def split_windows(data, seq_length):
    x = []
    y = []
    for i in range(len(data) - seq_length - 1):  # range的范围需要减去时间步长和1
        _x = data[i:(i + seq_length), :]
        _y = data[i + seq_length, -1]
        x.append(_x)
        y.append(_y)
    x, y = np.array(x), np.array(y)
    print('x.shape,y.shape=\n', x.shape, y.shape)
    return x, y


# 数据分离
def split_data(x, y, split_ratio):
    train_size = int(len(y) * split_ratio)
    test_size = len(y) - train_size

    x_data = Variable(torch.Tensor(np.array(x)))
    y_data = Variable(torch.Tensor(np.array(y)))

    x_train = Variable(torch.Tensor(np.array(x[0:train_size])))
    y_train = Variable(torch.Tensor(np.array(y[0:train_size])))
    y_test = Variable(torch.Tensor(np.array(y[train_size:len(y)])))
    x_test = Variable(torch.Tensor(np.array(x[train_size:len(x)])))

    print('x_data.shape,y_data.shape,x_train.shape,y_train.shape,x_test.shape,y_test.shape:\n{}{}{}{}{}{}'
          .format(x_data.shape, y_data.shape, x_train.shape, y_train.shape, x_test.shape, y_test.shape))

    return x_data, y_data, x_train, y_train, x_test, y_test


# 数据装入
def data_generator(x_train, y_train, x_test, y_test, n_iters, batch_size):
    num_epochs = n_iters / (len(x_train) / batch_size)  # n_iters代表一次迭代
    num_epochs = int(num_epochs)
    train_dataset = Data.TensorDataset(x_train, y_train)
    test_dataset = Data.TensorDataset(x_test, y_test)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False,
                                               drop_last=True)  # 加载数据集,使数据集可迭代
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False,
                                              drop_last=True)

    return train_loader, test_loader, num_epochs


# 定义模型
# 定义一个类
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size, seq_length) -> None:
        super(Net, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.num_directions = 1  # 单向LSTM

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True)  # LSTM层
        self.lstm2 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers,
                             batch_first=True)  # LSTM层
        self.fc1 = nn.Linear(hidden_size, hidden_size)  # 全连接层
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size, output_size)  # 全连接层

    def forward(self, x):
        h_0 = torch.randn(self.num_directions * self.num_layers, x.size(0), self.hidden_size)
        c_0 = torch.randn(self.num_directions * self.num_layers, x.size(0), self.hidden_size)
        output, _ = self.lstm(x, (h_0, c_0))
        output2, _ = self.lstm2(output, (h_0, c_0))
        pred = self.fc1(output2)
        pred = self.relu(pred)
        ##pred = self.fc(output)
        pred = self.fc(pred)
        pred = pred[:, -1, :]
        return pred


# 参数设置
def fitness_function(position, n):
    pso_val_loss = []
    val_losses = []
    for i in range(n):
        seq_length = int(position[i, 0])
        num_layers = int(position[i, 1])
        hidden_size = int(position[i, 2])
        n_iters = int(position[i, 3]) * 100
        lr = position[i, 4]
        # seq_length = 5  # 时间步长
        input_size = 3
        # num_layers = 1
        # hidden_size = 12
        batch_size = int(position[i, 5])
        # n_iters = 5000
        # lr = 0.001
        output_size = 1
        split_ratio = 0.8

        moudle = Net(input_size, hidden_size, num_layers, output_size, batch_size, seq_length)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(moudle.parameters(), lr=lr)
        print(moudle)

        # 数据导入
        data, label = get_Data(path)
        data, label, mm_y = normalization(data, label)
        x, y = split_windows(data, seq_length)
        x_data, y_data, x_train, y_train, x_test, y_test = split_data(x, y, split_ratio)
        train_loader, test_loader, num_epochs = data_generator(x_train, y_train, x_test, y_test, n_iters, batch_size)

        # train
        for epochs in range(num_epochs):
            for i, (batch_x, batch_y) in enumerate(train_loader):
                outputs = moudle(batch_x)
                optimizer.zero_grad()  # 将每次传播时的梯度累积清除
                loss = criterion(outputs.reshape(-1), batch_y)  # 计算损失
                loss.backward()  # 反向传播
                optimizer.step()
            print(epochs)
        moudle.eval()
        with torch.no_grad():
            losses = []
            for batch_x, batch_y in test_loader:
                outputs = moudle(batch_x)
                loss = criterion(outputs.reshape(-1), batch_y)
                losses.append(loss)
        val_losses.append(np.mean(losses))
        print("loss", val_losses)
    return val_losses  # 返回适应度值，这里使用验证集上的损失作为示例


def result(x_data, y_data):
    moudle.eval()
    train_predict = moudle(x_data)

    data_predict = train_predict.data.numpy()
    y_data_plot = y_data.data.numpy()
    y_data_plot = np.reshape(y_data_plot, (-1, 1))
    data_predict = mm_y.inverse_transform(data_predict)
    y_data_plot = mm_y.inverse_transform(y_data_plot)

    print('MAPE/MAE/RMSE')
    print(mean_absolute_percentage_error(y_data_plot, data_predict))
    print(mean_absolute_error(y_data_plot, data_predict))
    print(np.sqrt(mean_squared_error(y_data_plot, data_predict)))
    end_time = time.time()
    print("代码运行时间: ", end_time - start_time, "秒")
    plt.plot(y_data_plot)
    plt.plot(data_predict)
    plt.legend(('real', 'predict'), fontsize='15')
    plt.show()


start_time = time.time()
path = "D:\lstmss\myb.xlsx"

options = {'c1': 2, 'c2': 2, 'w': 1.5}
bounds = (np.array([3, 1, 8, 20, 0.0001, 30]),
          np.array([7, 5, 15, 70, 0.01, 70]))
n_particles = 5
init_pos = np.random.uniform(low=np.array(bounds)[0, :], high=np.array(bounds)[1, :], size=(n_particles, 6))
init_pos[1, :] = np.array([5, 1, 12, 50, 0.001, 50])
init_pos[2, :] = np.array([7, 1, 15, 70, 0.001, 32])
init_pos[3, :] = np.array([5, 1, 12, 40, 0.001, 50])
optimizer = GlobalBestPSO(n_particles=n_particles, dimensions=6, options=options, bounds=bounds,
                          init_pos=init_pos)
cost, pos = optimizer.optimize(fitness_function, iters=2, n=n_particles)
print(pos)

seq_length = int(pos[0])  # 时间步长
input_size = 3
num_layers = int(pos[1])
hidden_size = int(pos[2])
batch_size = int(pos[5])
n_iters = int(pos[3]) * 100
lr = pos[4]
output_size = 1
split_ratio = 0.8

moudle = Net(input_size, hidden_size, num_layers, output_size, batch_size, seq_length)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(moudle.parameters(), lr=lr)
print(moudle)

# 数据导入
data, label = get_Data(path)
data, label, mm_y = normalization(data, label)
x, y = split_windows(data, seq_length)
x_data, y_data, x_train, y_train, x_test, y_test = split_data(x, y, split_ratio)
train_loader, test_loader, num_epochs = data_generator(x_train, y_train, x_test, y_test, n_iters, batch_size)

# train
iter = 0
for epochs in range(num_epochs):
    for i, (batch_x, batch_y) in enumerate(train_loader):
        outputs = moudle(batch_x)
        optimizer.zero_grad()  # 将每次传播时的梯度累积清除
        # print(outputs.shape, batch_y.shape)
        loss = criterion(outputs.reshape(-1), batch_y)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()
        iter += 1
        if iter % 100 == 0:
            print("iter: %d, loss: %1.5f" % (iter, loss.item()))
            print(x_data.shape)
moudle.eval()
train_predict = moudle(x_data)

result(x_data, y_data)
start_time = time.time()
result(x_test, y_test)

torch.save(moudle.state_dict(), 'flstm_model.pth')

parameters = {
    'param1': seq_length,
    'param2': input_size,
    'param3': num_layers,
    'param4': hidden_size,
    'param5': batch_size,
    'param6': n_iters,
    'param7': lr,
    'param8': output_size,
    'param9': split_ratio
}

# 将参数保存到JSON文件中
with open('fparameters.json', 'w') as file:
    json.dump(parameters, file)
