import torch.nn as nn
import torch.nn.functional as F


########################################################################################################################
# 残差块
# -------------------------------------------------------------------------------------------------------------------- #
class ResidualBlock(nn.Module):
    def __init__(self,
                 in_channels=16,
                 out_channels=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
        super(ResidualBlock, self).__init__()

        layers = []
        if conv_first:
            layers.append(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=kernel_size,
                          stride=strides,
                          padding=1,
                          bias=False))
            if batch_normalization:
                layers.append(nn.BatchNorm2d(out_channels))

            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())

            self.model = nn.Sequential(*layers)
        else:
            if batch_normalization:
                layers.append(nn.BatchNorm2d(in_channels))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            layers.append(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=kernel_size,
                          stride=strides,
                          padding=1,
                          bias=False))
            self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)
        return out


class ResNet_V1(nn.Module):
    def __init__(self, in_channels, depth, num_classes=10):
        super(ResNet_V1, self).__init__()
        if (depth - 2) % 6 != 0:
            raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
        num_filters = 16
        num_res_blocks = int((depth - 2) / 6)

        self.conv1 = nn.Conv2d(in_channels, num_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)

        self.layer1 = self.make_layer(num_filters, num_filters, num_res_blocks, stride=1)
        self.layer2 = self.make_layer(num_filters, num_filters * 2, num_res_blocks, stride=2)
        self.layer3 = self.make_layer(num_filters * 2, num_filters * 4, num_res_blocks, stride=2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(num_filters * 4, num_classes)

    def make_layer(self, in_channels, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(in_channels, out_channels, strides=stride))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


class ResNet_V2(nn.Module):
    def __init__(self, in_channels, depth, num_classes=10):
        super(ResNet_V2, self).__init__()
        num_res_blocks = int((depth - 2) / 9)
        self.num_filters = 16
        self.conv1 = nn.Conv2d(in_channels, self.num_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.num_filters)
        self.relu = nn.ReLU()

        self.layer1 = self.make_layer(ResidualBlock, self.num_filters, num_res_blocks)
        self.layer2 = self.make_layer(ResidualBlock, self.num_filters * 2, num_res_blocks, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, self.num_filters * 4, num_res_blocks, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, self.num_filters * 8, num_res_blocks, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.num_filters * 8, num_classes)

    def make_layer(self, block, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(block(self.num_filters, out_channels, strides=stride))
        self.num_filters = out_channels
        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # Store for the skip connection
        residual = out
        out = self.layer1(out)
        out += residual
        out = self.relu(out)
        residual = out
        out = self.layer2(out)
        out += residual
        out = self.relu(out)
        residual = out
        out = self.layer3(out)
        out += residual
        out = self.relu(out)
        residual = out
        out = self.layer4(out)
        out += residual
        out = self.relu(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return F.log_softmax(out, dim=1)


"""model = ResNet_V2(in_channels=3, depth=11, num_classes=100)

# Print the models architecture
print(model)
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations for the CIFAR-100 datasets
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-100 datasets
train_dataset = torchvision.datasets.CIFAR100(root='.././datasets', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR100(root='.././datasets', train=False, download=True, transform=transform)

# Create datasets loaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the models, optimizer, and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
num_epochs = 10
model.to(device)
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Print training loss after each epoch
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

# Test the models
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f"Test Accuracy: {100 * accuracy:.2f}%")
"""