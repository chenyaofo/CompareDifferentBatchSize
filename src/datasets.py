from torchvision import datasets, transforms


def mnist_dataset(train):
    return datasets.MNIST('../data/MNIST', train=train, download=True,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,))
                          ]))


def cifar10_dataset(train):
    return datasets.CIFAR10(
        root='../data/CIFAR-10', train=train, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]))


def cifar100_dataset(train):
    return datasets.CIFAR10(
        root='../data/CIFAR-100', train=train, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]))
