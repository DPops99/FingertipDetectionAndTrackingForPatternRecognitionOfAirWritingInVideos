import matplotlib.pyplot as plt
import json

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
    losses_path = '/segmentation_model_results/RefineNet/MSELoss_15_epochs_first_try/refinenet_losses_mse_hgr_only_lr_1e-4.json'
    save_paths = ['refinenet_train_losses_mse_hgr_only.jpg', 'refinenet_val_losses_mse_hgr_only.jpg']
    losses = load_losses(losses_path=losses_path)
    for loss_item, save_path in zip(losses.items(), save_paths):
        key, value = loss_item
        plot_loss(loss=value, type=key, save_path=save_path)

if __name__=='__main__':
    create_loss_report()
