import pandas as pd
import numpy as np
import torch
import torch.optim as optimizer
import csv

def DataLoad_Process():
    train = pd.read_csv('./datasets/train.csv')
    test = pd.read_csv('./datasets/test.csv')
    d_all = pd.concat([train,test],ignore_index=True)
    # 缺失值：1.数值填入mean 2.分类填入众数 3.模型预测缺失值 TMD!!!fillna后float变obj！！
    d_all['Age'] = d_all['Age'].fillna(22)
    d_all['Fare'] = d_all['Fare'].fillna(7.25)
    # str转int
    d_all.loc[d_all['Sex']=='male','Sex'] = 0
    d_all.loc[d_all['Sex']=='female','Sex'] = 1
    d_all['Embarked'] = d_all['Embarked'].fillna('S')
    d_all.loc[d_all['Embarked']=='S','Embarked'] = 0
    d_all.loc[d_all['Embarked']=='C','Embarked'] = 1
    d_all.loc[d_all['Embarked']=='Q','Embarked'] = 2
    # 类型转换
    d_all['Age'] = d_all['Age'].astype("float64")
    d_all['Fare'] = d_all['Fare'].astype("float64")
    d_all['Embarked'] = d_all['Embarked'].astype("int64")
    d_all['Sex'] = d_all['Sex'].astype("int64")
    # 划分数据集
    d_train = d_all.loc[0:890,['Pclass','Sex','Age','Parch','Fare','Embarked']]
    d_test = d_all.loc[891:,['Pclass','Sex','Age','Parch','Fare','Embarked']]
    label = pd.DataFrame(train['Survived'])
    print(d_test.info())
    tr_input = torch.utils.data.TensorDataset(torch.Tensor(np.array(d_train)), torch.Tensor(np.array(label)))
    te_input = torch.utils.data.TensorDataset(torch.Tensor(np.array(d_test)))
    return tr_input,te_input

def net():
    input_data = 6
    hidden_layer1 = 50
    hidden_layer2 = 100
    hidden_layer3 = 50
    output_data = 1
    model = torch.nn.Sequential(
        torch.nn.Linear(input_data,hidden_layer1),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden_layer1,hidden_layer2),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_layer2,hidden_layer3),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_layer3,output_data),
        torch.nn.Sigmoid()
    )
    return model

if __name__ == "__main__":
    data,test = DataLoad_Process()
    model = net()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    optimizer = optimizer.SGD(model.parameters(), lr=0.01, weight_decay=1e-6, momentum=0.7, nesterov=True)
    loss_func = torch.nn.MSELoss()
    for epoch in range(500):
        tr_loss = []
        for tr_d,tar in data:
            tr_d = tr_d.to(device)
            tar = tar.to(device)
            model = model.to(device)
            out = model(tr_d)
            loss = loss_func(out,tar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_loss.append(loss.item())
        torch.save(model,'./model/model'+str(epoch)+'.pkl')
        print('epoch:',epoch,'loss:',np.mean(tr_loss))

    path_model = './model/model'+str(225)+'.pkl'
    net_load = torch.load(path_model)
    print("Testing Begining ... ")
    csvFile = open('./result/result.csv','w',newline='')
    writer = csv.writer(csvFile)
    writer.writerow(["PassengerId","Survived"])
    for i, data_tuple in enumerate(test, 0):
        test_d = data_tuple[0]
        test_d = test_d.to(device)
        net_load = net_load.to(device)
        out = net_load(test_d)
        out = 1 if out>0.5 else 0
        writer.writerow([892+i,out])
    

