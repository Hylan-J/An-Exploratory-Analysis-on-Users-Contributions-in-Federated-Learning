import torch.nn as nn


class CustomModel(nn.Module):
    def __init__(self, num_classes=10):
        super(CustomModel, self).__init__()

        # Block 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.activation1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.activation2 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(128)
        self.activation3 = nn.ReLU()
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.batchnorm4 = nn.BatchNorm2d(128)
        self.activation4 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 3
        self.conv5 = nn.Conv2d(128, 196, kernel_size=3, padding=1)
        self.batchnorm5 = nn.BatchNorm2d(196)
        self.activation5 = nn.ReLU()
        self.conv6 = nn.Conv2d(196, 196, kernel_size=3, padding=1)
        self.batchnorm6 = nn.BatchNorm2d(196)
        self.activation6 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(196 * 4 * 4, 256)
        self.batchnorm_fc = nn.BatchNorm1d(256)
        self.activation_fc = nn.ReLU()

        self.fc2 = nn.Linear(256, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.activation1(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.activation2(x)
        x = self.maxpool1(x)

        # Block 2
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.activation3(x)
        x = self.conv4(x)
        x = self.batchnorm4(x)
        x = self.activation4(x)
        x = self.maxpool2(x)

        # Block 3
        x = self.conv5(x)
        x = self.batchnorm5(x)
        x = self.activation5(x)
        x = self.conv6(x)
        x = self.batchnorm6(x)
        x = self.activation6(x)
        x = self.maxpool3(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.batchnorm_fc(x)
        x = self.activation_fc(x)

        x = self.fc2(x)
        x = self.softmax(x)

        return x

"""
torch.manual_seed(42)

# Define transformations for the datasets
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 datasets
trainset = torchvision.datasets.CIFAR10(root='datasets', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)

# Instantiate the model
num_classes = 10
model = CustomModel(num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Train the model
for epoch in range(2):  # loop over the datasets multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
"""