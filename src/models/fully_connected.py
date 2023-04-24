import torch


class FullyConnected(torch.nn.Module):
    def __init__(self):
        super(FullyConnected, self).__init__()
        self.relu = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(28 * 28, 512)
        self.fc2 = torch.nn.Linear(512, 384)
        self.fc3 = torch.nn.Linear(384, 10)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    model = FullyConnected()
    x = torch.randn(1, 28 * 28)
    y = model(x)
    print(y.shape)
