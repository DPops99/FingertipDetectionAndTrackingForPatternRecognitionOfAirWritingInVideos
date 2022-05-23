from train import *

if __name__=='__main__':
    train(batch_size=8, num_classes=1, epochs=100, loss_type='bce')