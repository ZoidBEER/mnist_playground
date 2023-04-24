import torch


class BasicBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(BasicBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        if in_channels != out_channels:
            self.skip = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, 1),
                torch.nn.BatchNorm2d(out_channels))
        else:
            self.skip = None

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)

        y = self.conv2(y)
        y = self.bn2(y)

        if self.skip is not None:
            y += self.skip(x)

        y = self.relu(y)
        return y


class ResNet18(torch.nn.Module):
    @staticmethod
    def _make_block(in_channels, out_channels, downscale=False):
        layers = [BasicBlock(in_channels, out_channels, kernel_size=3, padding=1),
                  BasicBlock(out_channels, out_channels, kernel_size=3, padding=1)]
        if downscale:
            layers.append(torch.nn.MaxPool2d(kernel_size=2, stride=2))
        return torch.nn.Sequential(*layers)

    def __init__(self):
        super(ResNet18, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.relu = torch.nn.ReLU(inplace=True)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.basic1 = self._make_block(64, 64, downscale=False)
        self.basic2 = self._make_block(64, 128, downscale=True)
        self.basic3 = self._make_block(128, 256, downscale=True)
        self.basic4 = self._make_block(256, 512, downscale=True)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = torch.nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.basic1(x)
        x = self.basic2(x)
        x = self.basic3(x)
        x = self.basic4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ResNet18MNIST(torch.nn.Module):
    @staticmethod
    def _make_block(in_channels, out_channels, downscale=False):
        layers = [BasicBlock(in_channels, out_channels, kernel_size=3, padding=1),
                  BasicBlock(out_channels, out_channels, kernel_size=3, padding=1)]
        if downscale:
            layers.append(torch.nn.MaxPool2d(kernel_size=2, stride=2))
        return torch.nn.Sequential(*layers)

    def __init__(self):
        super(ResNet18MNIST, self).__init__()
        self.basic1 = self._make_block(1, 64, downscale=False)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.basic2 = self._make_block(64, 128, downscale=True)
        self.basic3 = self._make_block(128, 256, downscale=True)
        self.basic4 = self._make_block(256, 512, downscale=True)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = torch.nn.Linear(512, 10)

    def forward(self, x):
        x = self.basic1(x)
        x = self.basic2(x)
        x = self.basic3(x)
        x = self.basic4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    model = ResNet18MNIST()
    x = torch.randn(1, 1, 28, 28)
    y = model(x)
    print(y.shape)
