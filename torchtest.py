import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

train = datasets.MNIST("", train=True, download = True,
                       transform = transforms.Compose([transforms.ToTensor()]))

test = datasets.MNIST("", train=False, download = True,
                       transform = transforms.Compose([transforms.ToTensor()]))
trainset =  torch.utils.data.DataLoader(train, batch_size = 10 , shuffle = True)
testset =  torch.utils.data.DataLoader(test, batch_size = 10 , shuffle = True)

total = 0
counter_dict = {0:0, 1:0, 2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}
for data in trainset:
    xs, ys = data
    for y in ys :
        counter_dict[int(y)] += 1
    for i in counter_dict:
        print(f"{i}: {counter_dict[i] / total * 100}")
    plt.imshow(data[0][0].view(28,28))
    plt.show()
    break
print(counter_dict)
