import matplotlib.pyplot as plt
import json
import os
import torch
from datetime import datetime 
import yaml
import argparse
from model import rf101

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
    model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
    return model

def generate_save_root(config):
    fromat = '%Y-%m-%d %H:%M:%S'
    current_time = datetime.now().strftime(fromat)
    return os.path.join(config['data']['save_root'],current_time, 'checkpoints')

def load_config(file_path):
    with open(file_path, 'r') as file:
        value = yaml.safe_load(file)
    return value

def get_args():
    parser = argparse.ArgumentParser(description='Train RefineNet model')
    parser.add_argument('--config', type=str)
    return parser.parse_args()

if __name__=='__main__':
    create_loss_report()
