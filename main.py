import math
import time
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torchvision.transforms import RandomHorizontalFlip, RandomAffine, ToTensor

from DeepLearning.models import *
from FederatedLearning.aggregation_algorithm import FedAvg
from FederatedLearning.objects.Client import Client
from FederatedLearning.objects.SQLiteHelper import SQLiteHelper
from FederatedLearning.objects.Server import Server
from utils import get_clients_subsets


class Experiment:
    def __init__(self, device):
        # 实验设备
        self.device = device
        self.NUM_CLASSES = {"MNIST": 10, "FashionMNIST": 10, "CIFAR10": 10, "CIFAR100": 100, "SVHN": 10, "CelebA": 20}

    # 获取数据集
    def get_data_and_labels(self, dataset):
        # 数据集特性转化器
        dataset_transforms = torchvision.transforms.Compose([RandomHorizontalFlip(),
                                                             RandomAffine(degrees=0, translate=(0.1, 0.1))])
        trainset, testset = None, None

        ################################################################################################################
        # 使用 MNIST 数据集
        # ------------------------------------------------------------------------------------------------------------ #
        if dataset == "MNIST":
            trainset = torchvision.datasets.MNIST('./DeepLearning/datasets',
                                                  train=True,
                                                  download=True,
                                                  transform=dataset_transforms)
            testset = torchvision.datasets.MNIST('./DeepLearning/datasets',
                                                 train=False,
                                                 download=True,
                                                 transform=dataset_transforms)
        # ------------------------------------------------------------------------------------------------------------ #

        ################################################################################################################
        # 使用 FashionMNIST 数据集
        # ------------------------------------------------------------------------------------------------------------ #
        elif dataset == "FashionMNIST":
            trainset = torchvision.datasets.FashionMNIST('./DeepLearning/datasets',
                                                         train=True,
                                                         download=True,
                                                         transform=dataset_transforms)
            testset = torchvision.datasets.FashionMNIST('./DeepLearning/datasets',
                                                        train=False,
                                                        download=True,
                                                        transform=dataset_transforms)
        # ------------------------------------------------------------------------------------------------------------ #

        ################################################################################################################
        # 使用 CIFAR10 数据集
        # ------------------------------------------------------------------------------------------------------------ #
        elif dataset == "CIFAR10":
            trainset = torchvision.datasets.CIFAR10('./DeepLearning/datasets',
                                                    train=True,
                                                    download=True,
                                                    transform=dataset_transforms)
            testset = torchvision.datasets.CIFAR10('./DeepLearning/datasets',
                                                   train=False,
                                                   download=True,
                                                   transform=dataset_transforms)
        # ------------------------------------------------------------------------------------------------------------ #

        ################################################################################################################
        # 使用 CIFAR100 数据集
        # ------------------------------------------------------------------------------------------------------------ #
        elif dataset == "CIFAR100":
            trainset = torchvision.datasets.CIFAR100('./DeepLearning/datasets',
                                                     train=True,
                                                     download=True,
                                                     transform=dataset_transforms)
            testset = torchvision.datasets.CIFAR100('./DeepLearning/datasets',
                                                    train=False,
                                                    download=True,
                                                    transform=dataset_transforms)
        # ------------------------------------------------------------------------------------------------------------ #

        ################################################################################################################
        # 使用 SVHN 数据集
        # ------------------------------------------------------------------------------------------------------------ #
        elif dataset == "SVHN":
            trainset = torchvision.datasets.SVHN('./DeepLearning/datasets',
                                                 split="train",
                                                 download=True,
                                                 transform=dataset_transforms)
            testset = torchvision.datasets.SVHN('./DeepLearning/datasets',
                                                split="test",
                                                download=True,
                                                transform=dataset_transforms)
        # ------------------------------------------------------------------------------------------------------------ #

        ################################################################################################################
        # 使用 CelebA 数据集
        # ------------------------------------------------------------------------------------------------------------ #
        elif dataset == "CelebA":
            trainset = torchvision.datasets.CelebA('./DeepLearning/datasets',
                                                   split="train",
                                                   download=True,
                                                   transform=dataset_transforms)
            testset = torchvision.datasets.CelebA('./DeepLearning/datasets',
                                                  split="test",
                                                  download=True,
                                                  transform=dataset_transforms)
        # ------------------------------------------------------------------------------------------------------------ #

        data_train = trainset.data.unsqueeze(1).float() / 255.0  # 数据归一化
        labels_train = trainset.targets
        data_test = testset.data.unsqueeze(1).float() / 255.0  # 数据归一化
        labels_test = testset.targets

        return data_train, labels_train, data_test, labels_test

    def get_model(self, dataset):
        net = None
        if dataset == "MNIST" or dataset == "FashionMNIST":
            net = LeNet5_MNIST()

        elif dataset == "CIFAR10" or dataset == "CelebA":
            net = CustomModel()

        elif dataset == "SVHN":
            net = LeNet5_SVHN()

        elif dataset == "CIFAR100":
            net = Cifar100ResNet(depth=7, num_classes=self.NUM_CLASSES[dataset])

        return net

    def flip_label(self, dataset, labels_train, noise_ratio):
        # 获取训练标签的数量
        n_train_labels = len(labels_train)
        # 获取噪声标签的数量
        n_noisy_labels = int(n_train_labels * noise_ratio / 100)
        # 获取噪声标签的索引
        noisy_labels_index = torch.randperm(n_train_labels)[:n_noisy_labels]
        # 替换标签的类别
        for i in noisy_labels_index:
            other_class_list = list(range(self.NUM_CLASSES[dataset]))
            other_class_list.remove(labels_train[i])
            labels_train[i] = np.random.choice(other_class_list)
        return labels_train

    def influence(self, server, clients, SQLite_helper, malicious_clients):
        n_client = len(clients)

        print("\n\033[93m influence指标评估: Start \033[0m")
        # 获得基础
        global_parameter = FedAvg(server.global_net, [client.local_net for client in clients])
        server.global_net.load_state_dict(global_parameter)
        # 对服务器网络进行评估
        accuracy_baseline = server.evaluate(server.global_net)

        influences = []
        for i in range(n_client):
            chosen_client_ids = list(np.arange(n_client))
            chosen_client_ids.remove(i)
            chosen_client_nets = [clients[id].local_net for id in chosen_client_ids]
            global_parameter = FedAvg(server.global_net, chosen_client_nets)
            server.global_net.load_state_dict(global_parameter)
            accuracy = server.evaluate(server.global_net)
            influence = accuracy_baseline - accuracy
            influences.append(influence)
            # 数据插入数据库
            SQLite_helper.insert_server_table(chosen=str(chosen_client_ids), indicator_value=str(influence))
            print(f"remove client: {i}, server accuracy: {accuracy}, influence: {influence}")
        print("\033[93m influence指标评估: End \033[0m\n")

        Honest_influence = []
        honest_clients = list(set(list(np.arange(n_client))) - set(malicious_clients))
        for honest_client in honest_clients:
            Honest_influence.append(influences[honest_client])
        Honest_influence = sum(Honest_influence) / len(Honest_influence)

        Malicious_influence = []
        for malicious_client in malicious_clients:
            Malicious_influence.append(influences[malicious_client])
        Malicious_influence = sum(Malicious_influence) / len(Malicious_influence)

        return Honest_influence, Malicious_influence

    def reputation(self, server, clients, SQLite_helper, malicious_clients):
        n_client = len(clients)

        print("\n\033[93m reputation指标评估: Start \033[0m")
        # 获得基础
        global_parameter = FedAvg(server.global_net, [client.local_net for client in clients])
        server.global_net.load_state_dict(global_parameter)
        # 对服务器网络进行评估
        accuracy_baseline = server.evaluate(server.global_net)

        reputations = []
        for i in range(n_client):
            chosen_client_ids = list(np.arange(n_client))
            chosen_client_ids.remove(i)
            chosen_client_nets = [clients[id].local_net for id in chosen_client_ids]
            global_parameter = FedAvg(server.global_net, chosen_client_nets)
            server.global_net.load_state_dict(global_parameter)
            accuracy = server.evaluate(server.global_net)
            reputation = 1 if (accuracy_baseline - accuracy) >= 0 else 0
            reputations.append(reputation)
            # 数据插入数据库
            SQLite_helper.insert_server_table(chosen=str(chosen_client_ids), indicator_value=str(reputation))
            print(f"remove client: {i}, server accuracy: {accuracy}, reputation: {reputation}")
        print("\033[93m reputation指标评估: End \033[0m\n")

        Honest_reputation = []
        honest_clients = list(set(list(np.arange(n_client))) - set(malicious_clients))
        for honest_client in honest_clients:
            Honest_reputation.append(reputations[honest_client])
        Honest_reputation = sum(Honest_reputation) / len(Honest_reputation)

        Malicious_reputation = []
        for malicious_client in malicious_clients:
            Malicious_reputation.append(reputations[malicious_client])
        Malicious_reputation = sum(Malicious_reputation) / len(Malicious_reputation)

        return Honest_reputation, Malicious_reputation

    def shapley(self, server, clients, SQLite_helper, malicious_clients):
        n_client = len(clients)
        shapleys = []
        client_list = list(np.arange(n_client))
        total = 0
        factorialTotal = math.factorial(n_client)
        clients_subsets = get_clients_subsets(n_client)

        print("\n\033[93m shapley指标评估: Start \033[0m")
        for client in client_list:
            clientShapley = 0
            chosen_clients = []
            for client_subset in clients_subsets:
                if client in client_subset:
                    chosen_clients.append(client)

                    subset_client_nets = [clients[id].local_net for id in list(client_subset)]
                    global_parameter = FedAvg(server.global_net, subset_client_nets)
                    server.global_net.load_state_dict(global_parameter)
                    subset_accuracy = server.evaluate(server.global_net)

                    remainderSet = client_subset.difference({client})
                    remainder_accuracy = 0
                    b = len(remainderSet)
                    factValue = (len(client_list) - b - 1)
                    if remainderSet != frozenset():
                        remainder_client_nets = [clients[id].local_net for id in list(remainderSet)]
                        global_parameter = FedAvg(server.global_net, remainder_client_nets)
                        server.global_net.load_state_dict(global_parameter)
                        remainder_accuracy = server.evaluate(server.global_net)

                    difference = subset_accuracy - remainder_accuracy
                    divisor = (math.factorial(factValue) * math.factorial(b) * 1.0) / (factorialTotal * 1.0)
                    weightValue = divisor * difference
                    clientShapley += weightValue

            shapleys.append(clientShapley)
            print(f"client{client}'s Shapley: {clientShapley}")
            # 数据插入数据库
            SQLite_helper.insert_server_table(chosen=str(chosen_clients), indicator_value=str(clientShapley))
            total = total + clientShapley
        print("\033[93m shapley指标评估: End \033[0m\n")

        Honest_shapley = []
        honest_clients = list(set(list(np.arange(n_client))) - set(malicious_clients))
        for honest_client in honest_clients:
            Honest_shapley.append(shapleys[honest_client])
        Honest_shapley = sum(Honest_shapley) / len(Honest_shapley)

        Malicious_shapley = []
        for malicious_client in malicious_clients:
            Malicious_shapley.append(shapleys[malicious_client])
        Malicious_shapley = sum(Malicious_shapley) / len(Malicious_shapley)

        return Honest_shapley, Malicious_shapley

    def main(self,
             indicator_type,
             dataset,
             client_data_number,
             noise_ratio,
             malicious_client,  # 恶意客户端
             hyperparameter,  # 超参数
             epochs):

        SQLite_helper = SQLiteHelper()
        SQLite_helper.create_database(f"results/{time.strftime('%Y-%m-%d %H-%M')} {len(malicious_client)} attacker.db")
        SQLite_helper.create_tables()

        SQLite_helper.insert_experiment_table(indicator_type=indicator_type,
                                              dataset=dataset,
                                              client_data_number=str(client_data_number),
                                              noise_ratio=str(noise_ratio) + "%",
                                              malicious_client=str(malicious_client),
                                              learning_rate=str(hyperparameter['learning_rate']),
                                              batch_size=hyperparameter['batch_size'],
                                              momentum=str(hyperparameter['momentum']))
        Honest = 0
        Malicious = 0
        slots = 5 if indicator_type == "reputation" else 1
        for slot in range(slots):
            data_train, labels_train, data_test, labels_test = self.get_data_and_labels(dataset)
            model = self.get_model(dataset)
            server = Server(data_test, labels_test, self.device, deepcopy(model), hyperparameter)
            clients = []
            data_used_index = 0
            for i in range(len(client_data_number)):
                if i not in malicious_client:
                    train_data = data_train[data_used_index: data_used_index + client_data_number[i]]
                    train_label = labels_train[data_used_index: data_used_index + client_data_number[i]]
                else:
                    train_data = data_train[data_used_index: data_used_index + client_data_number[i]]
                    train_label = self.flip_label(dataset,
                                                  labels_train[
                                                  data_used_index: data_used_index + client_data_number[i]],
                                                  float(noise_ratio))
                client = Client(i, train_data, train_label, self.device, deepcopy(model), hyperparameter)
                clients.append(client)
                data_used_index = data_used_index + client_data_number[i]

            print("\n\033[93m客户端训练: Start\033[0m")
            for epoch in range(epochs):
                print("=======================> Epoch {}/{}  <=======================".format(epoch + 1, epochs))
                for client in clients:
                    # 客户端训练
                    loss = client.train()
                    # 客户端测试
                    # accuracy = client.evaluate()  # 用客户端自己数据进行测试
                    accuracy = server.evaluate(client.local_net)  # 用服务器数据进行测试
                    SQLite_helper.insert_clients_table(epoch=epoch + 1,
                                                       client_id=client.id,
                                                       loss=str(loss),
                                                       accuracy=str(accuracy))
                    # 打印信息
                    print("\tclient{:2d} ====> loss: {:6f}, accuracy: {:6f}".format(client.id, loss, accuracy))
            print("\033[93m客户端训练: End\033[0m\n")

            if indicator_type == "influence":
                Honest_indicator_value, Malicious_indicator_value = self.influence(server=server,
                                                                                   clients=clients,
                                                                                   SQLite_helper=SQLite_helper,
                                                                                   malicious_clients=malicious_client)
                Honest = Honest_indicator_value
                Malicious = Malicious_indicator_value
            elif indicator_type == "reputation":
                Honest_indicator_value, Malicious_indicator_value = self.reputation(server=server,
                                                                                    clients=clients,
                                                                                    SQLite_helper=SQLite_helper,
                                                                                    malicious_clients=malicious_client)
                Honest += Honest_indicator_value / slots
                Malicious += Malicious_indicator_value / slots
            elif indicator_type == "shapley":
                Honest_indicator_value, Malicious_indicator_value = self.reputation(server=server,
                                                                                    clients=clients,
                                                                                    SQLite_helper=SQLite_helper,
                                                                                    malicious_clients=malicious_client)
                Honest = Honest_indicator_value
                Malicious = Malicious_indicator_value
        SQLite_helper.close()

        return Honest, Malicious


if __name__ == '__main__':
    experiment = Experiment(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    indicator_type = "influence"
    dataset = "MNIST"
    noise_ratio = list(range(0, 110, 10))
    client_data_number = [6000, 6000, 6000, 6000]

    malicious_client = [[0], [0, 1], [0, 1, 2]]
    batch_size = 10  # 默认64
    learning_rate = 0.0001
    momentum = 0.9  # 动量，在卷积神经网络（CNN）中，Momentum（动量）的值通常设置为0.9左右，这也是在许多深度学习框架中默认的设置值。这个值是通过大量实践验证得出的，可以较好地平衡模型训练的速度和稳定性。
    epochs = 100  # 默认500

    hyperparameter = {
        "learning_rate": 0.001,
        "batch_size": 64,
        "momentum": 0.9
    }

    Honest_indicator_values = []
    Malicious_indicator_values = []
    for i in malicious_client:
        Honest_indicator_values_ = []
        Malicious_indicator_values_ = []
        for j in noise_ratio:
            Honest_indicator_value, Malicious_indicator_value = experiment.main(indicator_type,
                                                                                dataset,
                                                                                client_data_number,
                                                                                j,
                                                                                i,
                                                                                hyperparameter,
                                                                                epochs)
            Honest_indicator_values_.append(Honest_indicator_value)
            Malicious_indicator_values_.append(Malicious_indicator_value)

        Honest_indicator_values.append(Honest_indicator_values_)
        Malicious_indicator_values.append(Malicious_indicator_values_)

    plt.plot(Honest_indicator_values[0], linestyle="-.", color="orange", label="Honest users - 1 attacker")
    plt.plot(Honest_indicator_values[1], linestyle="--.", color="orange", label="Honest users - 2 attackers")
    plt.plot(Honest_indicator_values[2], linestyle="-..", color="orange", label="Honest users - 3 attackers")
    plt.plot(Malicious_indicator_values[0], linestyle="-", color="blue", label="Malicious users - 1 attacker")
    plt.plot(Malicious_indicator_values[1], linestyle="--", color="blue", label="Malicious users - 2 attackers")
    plt.plot(Malicious_indicator_values[2], linestyle=":", color="blue", label="Malicious users - 3 attackers")
    plt.show()
    # print(Honest_indicator_values)
    # print(Malicious_indicator_values)
