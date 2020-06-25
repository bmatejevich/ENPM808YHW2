import numpy as np
import torch
import torch.nn as NN
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib as mpl
from Problem1 import *
from Problem2 import *
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'


flag = False
runboolean = True

def mainprog(runboolean):
    if __name__ == '__main__':  # if this is the main file in which all files are imported to
        problem = 1 # make sure to define the problem you want 1 or 2

        if problem==1:
            Problem1Main()
        elif problem==2:
            Problem2Main()
        else:
            runboolean = False
            return runboolean

        runboolean = False
        return runboolean


if __name__ == '__main__':
    runboolean = True
    while runboolean == True:
        runboolean = mainprog(runboolean)
