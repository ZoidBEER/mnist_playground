import torchvision.datasets


if __name__ == "__main__":
    mnist_trainset = torchvision.datasets.MNIST(root='../../data', train=True, download=True, transform=None)
    mnist_testset = torchvision.datasets.MNIST(root='../../data', train=False, download=True, transform=None)
    print(len(mnist_trainset))
    print(len(mnist_testset))
    print(mnist_trainset[0])
