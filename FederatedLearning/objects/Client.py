import torch
from torch import optim, nn


class Client:
    def __init__(self, id, train_data, train_label, device, local_net, hyperparameter):
        self.id = id
        # 训练集数据
        self.train_data = train_data
        # 训练集标签
        self.train_label = train_label
        # 客户端使用设备
        self.device = device
        # 客户端网络模型
        self.local_net = local_net.to(self.device)
        # 客户端相关超参数
        self.learning_rate = hyperparameter["learning_rate"]
        self.momentum = hyperparameter["momentum"]
        self.batch_size = hyperparameter["batch_size"]
        self.train_epochs = int(len(self.train_data)/self.batch_size)
        # 客户端使用交叉熵损失函数
        self.loss_fn = nn.CrossEntropyLoss().to(self.device)
        # 客户端使用随机梯度下降优化器
        self.optimizer = optim.SGD(self.local_net.parameters(), lr=self.learning_rate, momentum=self.momentum)

    def train(self):
        self.local_net.train()
        loss = 0
        for epoch in range(self.train_epochs):
            train_data = self.train_data[epoch*self.batch_size:(epoch+1)*self.batch_size]
            train_label = self.train_label[epoch*self.batch_size:(epoch+1)*self.batch_size]
            train_data, train_label = train_data.to(self.device), train_label.to(self.device)
            outputs = self.local_net(train_data)
            loss = self.loss_fn(outputs, train_label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return float(loss)

    def evaluate(self):
        self.local_net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            test_data, test_label = self.train_data.to(self.device), self.train_label.to(self.device)
            outputs = self.local_net(test_data)
            _, predicted = torch.max(outputs.data, 1)
            total += test_label.size(0)
            correct += (predicted == test_label).sum().item()

        accuracy = correct / total
        return accuracy
