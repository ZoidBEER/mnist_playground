import torch


class LeNet(torch.nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2)
        self.conv1 = torch.nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.classifier = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    model = LeNet()
    x = torch.randn(1, 1, 28, 28)
    y = model(x)
    print(y.shape)
