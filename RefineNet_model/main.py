import numpy as np
import torch

def test():
    x = torch.tensor(0.0)
    print(x)
    x += torch.tensor(5.0)
    print(x)

if __name__=='__main__':
    test()