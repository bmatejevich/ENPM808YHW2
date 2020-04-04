from __main__ import *
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
from CNNBuild import *
from trainNN import *


def Problem2Main():

    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = torchvision.datasets.CIFAR10(root='./cifardata', train=True, download=True, transform=transform)

    test_set = torchvision.datasets.CIFAR10(root='./cifardata', train=False, download=True, transform=transform)

    classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Training
    train_sampler = SubsetRandomSampler(np.arange(50000, dtype=np.int64))

    # Validation
    val_sampler = SubsetRandomSampler(np.arange(10000, 25000, dtype=np.int64))

    # Test
    test_sampler = SubsetRandomSampler(np.arange(10000, dtype=np.int64))


    CNN = CNNBuild()

    trainNet(CNN, batch_size=500, epochs=50, lr=1.5, train_set=train_set, train_sampler=train_sampler,
             val_sampler=val_sampler, test_set=test_set, test_sampler=test_sampler, classes = classes)

