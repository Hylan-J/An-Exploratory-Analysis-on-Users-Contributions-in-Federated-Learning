import torch
from torch import nn, optim


class Server:
    def __init__(self, test_data, test_label, device, global_net, hyperparameter):
        """
        服务器端测试相关参数
        """
        # 测试集数据
        self.test_data = test_data.clone().detach().to(torch.float)
        # 测试集标签
        self.test_label = test_label.clone().detach().to(torch.float)
        # 设备
        self.device = device
        # 网络模型
        self.global_net = global_net.to(self.device)
        # 客户端相关超参数
        self.learning_rate = hyperparameter["learning_rate"]
        self.momentum = hyperparameter["momentum"]
        self.batch_size = hyperparameter["batch_size"]
        self.train_epochs = int(len(self.test_data)/self.batch_size)
        # 客户端使用交叉熵损失函数
        self.loss_fn = nn.CrossEntropyLoss().to(self.device)
        # 客户端使用随机梯度下降优化器
        self.optimizer = optim.SGD(self.global_net.parameters(), lr=self.learning_rate, momentum=self.momentum)

    def train(self):
        self.global_net.train()
        loss = 0
        for epoch in range(self.train_epochs):
            train_data = self.test_data[epoch*self.batch_size:(epoch+1)*self.batch_size].to(self.device)
            train_label = self.test_label[epoch*self.batch_size:(epoch+1)*self.batch_size].to(self.device)
            # 前向传播,即网络如何根据输入得到输出的
            outputs = self.global_net(train_data)
            # loss计算
            loss = self.loss_fn(outputs, train_label)
            # 反向传播与优化,反向传播算法的核心是代价函数对网络中参数（各层的权重和偏置）的偏导表达式和。
            self.optimizer.zero_grad()  # 梯度清零：重置模型参数的梯度。默认是累加，为了防止重复计数，在每次迭代时显式地将它们归零。
            loss.backward()  # 反向传播计算梯度：计算当前张量w.r.t图叶的梯度。
            self.optimizer.step()  # 参数更新：根据上面计算的梯度，调整参数

        return float(loss)

    def evaluate(self, net):
        net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            test_data, test_label = self.test_data.to(self.device), self.test_label.to(self.device)
            outputs = net(test_data)
            _, predicted = torch.max(outputs.data, 1)
            total += test_label.size(0)
            correct += (predicted == test_label).sum().item()

        accuracy = correct / total
        return accuracy
