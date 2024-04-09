import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, decay, more_filters=False, first=False):
        super(ResidualBlock, self).__init__()

        self.more_filters = more_filters

        if more_filters:
            # 添加更多的配置，例如更多的卷积层、BatchNorm 层等
            # 你可能需要根据实际情况修改这里的代码
            # 示意性地展示如何添加一个额外的卷积层
            self.extra_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.extra_bn = nn.BatchNorm2d(out_channels)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None

        if in_channels != out_channels or more_filters:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        if self.more_filters:
            # 如果有更多的配置，添加相应的处理
            extra_out = self.extra_conv(x)
            extra_out = self.extra_bn(extra_out)
            out += extra_out

        return out

    # 在你的 ResNet 模型中，确保 BasicBlock 的输入通道数与前一层的输出通道数匹配
    # 例如，如果你的第一层卷积层输出通道数是16，那么 BasicBlock 的输入通道数应该也是16


class Cifar100ResNet(nn.Module):
    def __init__(self, depth, num_classes):
        super(Cifar100ResNet, self).__init__()

        self.num_conv = 3
        self.decay = 2e-3
        self.filters = 16

        self.conv1 = nn.Conv2d(3, self.filters, kernel_size=self.num_conv, stride=1, padding=self.num_conv // 2)
        self.batch_norm = nn.BatchNorm2d(self.filters)
        self.relu = nn.ReLU()

        self.layer1 = self.make_layer(ResidualBlock, self.filters, depth)
        self.layer2 = self.make_layer(ResidualBlock, self.filters * 2, depth, more_filters=True)
        self.layer3 = self.make_layer(ResidualBlock, self.filters * 4, depth, more_filters=True)

        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.filters * 4, num_classes)

        # Initialize convolutional layers
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc.weight, mode='fan_out', nonlinearity='relu')

    def make_layer(self, block, filters, blocks, more_filters=False):
        layers = []
        layers.append(block(self.num_conv, filters, self.decay, more_filters=more_filters, first=True))
        for _ in range(1, blocks):
            layers.append(block(self.num_conv, filters, self.decay))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.global_avg_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return F.softmax(x, dim=1)

"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# 设置随机种子以保证结果的可重复性
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# 定义数据转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 下载并加载CIFAR-100数据集
train_dataset = torchvision.datasets.CIFAR100(root='../datasets', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR100(root='../datasets', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 实例化模型并将其移到GPU
depth = 6  # You can adjust this value based on your requirements
num_classes = 100  # Assuming 100 classes for CIFAR-100
model = Cifar100ResNet(depth, num_classes).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 训练模型
for epoch in range(5):  # 仅进行5个epoch以加快演示速度
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

print('Finished Training')

# 在测试集上评估模型
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy on the test set: %d %%' % (100 * correct / total))
"""