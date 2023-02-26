import matplotlib.pyplot as plt
import json
import os
import torch
from RefineNet_model.model import rf101

def plot_loss(loss, type, save_path):
    x = [x for x in range(len(loss))]
    plt.plot(x[1:], loss[1:])
    plt.xlabel('Epochs')
    plt.ylabel('{}'.format(type))
    plt.savefig('{}'.format(save_path))
    plt.clf()

def load_losses(losses_path):
    with open(losses_path, 'r') as f:
        losses = json.load(f)
    return losses

def create_loss_report():
    losses_path = '/home/popa/Documents/diplomski_rad/FingertipDetectionAndTrackingForPatternRecognitionOfAirWritingInVideos/segmentation_model_results/RefineNet/BCELoss_100_epochs_hgr1_only/final_model_losses.json'
    save_root = '/home/popa/Documents/diplomski_rad/FingertipDetectionAndTrackingForPatternRecognitionOfAirWritingInVideos/segmentation_model_results/RefineNet/BCELoss_100_epochs_hgr1_only'
    save_paths = [os.path.join(save_root,'train_losses.jpg'), os.path.join(save_root,'valid_losses.jpg')]
    losses = load_losses(losses_path=losses_path)
    for loss_item, save_path in zip(losses.items(), save_paths):
        key, value = loss_item
        print(type(value))
        plot_loss(loss=value, type=key, save_path=save_path)

def get_refinenet_model(model_path, device):
    model = rf101(num_classes=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model

if __name__=='__main__':
    create_loss_report()
