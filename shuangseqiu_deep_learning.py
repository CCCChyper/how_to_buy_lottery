
import pprint

import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from torch.nn import Sequential, LSTM

from shuangseqiu_extract import extractor
from torch.utils.data import Dataset, DataLoader

from shuangseqiu_machine_learning import trainer, data_transformer, transformdata_and_write


class lottery_dataset(Dataset):
    def __init__(self,rd):
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit([range(1, 34)])
        self.rawdata=rd

    def __getitem__(self, index):
        return


import torch
import torch.nn as nn
import numpy as np




# 定义多标签分类的神经网络模型
class MultiLabelClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_labels):
        super(MultiLabelClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_labels)
        self.sigmoid = nn.Sigmoid()  # 使用 Sigmoid 激活函数进行多标签分类

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)  # 使用 Sigmoid 将输出转换为概率
        return out



def create_sequence(data,seq_length):
    X,y=[],[]
    for i in range(1,1000):
        tmp=data[i]
        for j in range(1,seq_length):
            tmp=tmp+data[i+j]
        X.append(tmp)
        y.append(data[i-1])
    return np.array(X),np.array(y)

if __name__=='__main__':
    ex=extractor()
    rawdata=[sorted([int(j) for j in i[1:]]) for i in ex.historical_data[::-1]]
    #pprint.pprint(rawdata)

    mlb = MultiLabelBinarizer()
    mlb.fit([range(1, 34)])
    one_hot_rawdata = mlb.transform(rawdata)

    seq_length=10
    data = np.array(one_hot_rawdata, dtype=np.float32)

    X,y=create_sequence(data,seq_length)

    train_x_path = './train_x-new.csv'
    train_y_path = './train_y-new.csv'

    tf = data_transformer()
    transformdata_and_write(tf, tuple((7, 1007)), train_x_path, train_y_path)
    tr = trainer()
    tr.load_x_y_data(train_x_path, train_y_path)
    #X=np.array(tr.x,dtype=np.float32)
    #y=np.array(tr.y,dtype=np.float32)


    X=torch.from_numpy(X)
    y=torch.from_numpy(y)
    # 初始化模型和定义参数
    input_size = 33  # 特征数
    hidden_size = 64  # 隐藏层大小
    num_labels = 33  # 类别数量

    min=4.00


    for i in range(10000):
        model = MultiLabelClassifier(input_size, hidden_size, num_labels)

        # 定义损失函数和优化器
        criterion = nn.BCELoss()  # 使用二元交叉熵损失函数适用于多标签分类任务
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # 训练模型
        num_epochs = 120
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            #if (epoch + 1) % 10 == 0:
                #print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

        # 在训练集上进行预测
        n = 1111
        target = data[n]
        tmp = data[n]
        for j in range(1, seq_length):
            tmp = tmp + data[n + j]

        input = torch.from_numpy(np.array(tmp))

        threshhold = 0.25
        model.eval()
        with torch.no_grad():
            predicted_labels = (model(input) > threshhold).float()

        result = []
        check = []
        c = 1
        for i in range(33):
            if predicted_labels[i] == 1:
                result.append(c)
            if target[i] == 1:
                check.append(c)
            c += 1
        # print(predicted_labels)
        #print(result)
        #print(check)

        totalcount = 0
        c1 = 0
        countdict = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0}
        for mi in range(1100, 1400):

            n = mi
            target = data[n]
            tmp = data[n+1]
            for j in range(1, seq_length):
                tmp = tmp + data[n +1+ j]
            input = torch.from_numpy(np.array(tmp))

            with torch.no_grad():
                predicted_labels = (model(input) > threshhold).float()

            result = predicted_labels
            check = target
            #print(result)
            #print(check)

            count = 0
            for i, j in zip(check, result):
                if i == 1 and j == 1:
                    count += 1
                if j == 1:
                    c1 += 1
            totalcount += count
            #print(f'----------------------答对{count}个')
            countdict[str(count)] += 1
        print(f'共计答对{totalcount}个')
        print(f'一共有{c1}个1')
        if c1 / totalcount <min:
            torch.save(model,'./dl_model.pth')
            min=c1 / totalcount
        print(c1 / totalcount)
        print(countdict)