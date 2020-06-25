from __main__ import *
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
from MNistCNN import *
from MNisttrainNN import *

def unpickle(file):
    """ loads pickle file"""
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def Problem1Main():
    """ trains and tests CNN for MNIST data"""
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    train_set = torchvision.datasets.MNIST('../data', train=True, download=True,transform=transform)
    test_set = torchvision.datasets.MNIST('../data', train=False,  transform=transform)

    classes = ('1', '2', '3', '4','5', '5', '7', '8', '9', '0')

    # Training
    train_sampler = SubsetRandomSampler(np.arange(20000, dtype=np.int64))
    # Validation
    val_sampler = SubsetRandomSampler(np.arange(20000, 25000, dtype=np.int64))
    # Test
    test_sampler = SubsetRandomSampler(np.arange(5000, dtype=np.int64))

    CNN = MNistCNN()

    MNisttrainNN(CNN, batch_size=200, epochs=3, lr=0.01, train_set=train_set, train_sampler=train_sampler,
             val_sampler=val_sampler, test_set=test_set, test_sampler=test_sampler, classes=classes)
