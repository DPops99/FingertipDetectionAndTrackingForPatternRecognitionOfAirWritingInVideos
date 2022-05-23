import sys
import time
import torch
import json
from RefineNet_model.model import *
from datasets.data import *
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
from losses import dice_loss


def train(batch_size=2, num_classes=1, epochs=2, save_path='/content/drive/MyDrive/refinenet', loss_type='bse', checkpoint_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = get_dataset(type='train', hgr1_only=True)
    valid_dataset = get_dataset(type='val', hgr1_only=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)
    if loss_type == 'mse':
        criterion = torch.nn.MSELoss(reduction='mean')
    elif loss_type == 'bce':
        criterion = torch.nn.BCEWithLogitsLoss()
    elif loss_type == 'dice':
        criterion = dice_loss
    else:
        criterion = torch.nn.BCEWithLogitsLoss()
    model = rf101(num_classes=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    train_losses = []
    val_losses = []
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
    else:
        start_epoch = 0
    model.to(device)
    for epoch in range(start_epoch, epochs):
        try:
            start_time = time.time()
            print('---------------------TRANING---------------------')
            train_loss = train_one_epoch(model=model, optimizer=optimizer, criterion=criterion,
                                         dataloader=train_dataloader, epoch=epoch, device=device)
            print('---------------------EVALUATION---------------------')
            val_loss = eval_one_epoch(model=model, criterion=criterion, dataloader=valid_dataloader, epoch=epoch,
                                      device=device)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_losses': train_losses,
                    'val_losses': val_losses
                }, os.path.join(save_path, "final_model_{}.pt".format(epoch + 1)))
                torch.save(model.state_dict(), os.path.join(save_path, "final_model_{}.pt".format(epoch + 1)))
            scheduler.step()
            print('Epoch {} finished in {} seconds'.format(epoch, time.time() - start_time))
        except KeyboardInterrupt:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict':scheduler.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses
            }, os.path.join(save_path, "current_model.pt"))
            with open(os.path.join(save_path, 'current_model_losses.json'), 'w') as f:
                json.dump({
                    'train_losses': train_losses,
                    'val_losses': val_losses
                }, f)
            sys.exit(0)
    with open(os.path.join(save_path, 'final_model_losses.json'), 'w') as f:
        json.dump({
            'train_losses': train_losses,
            'val_losses': val_losses
        }, f)


def train_one_epoch(model, optimizer, criterion, dataloader, epoch, device):
    model.train()
    total_loss = 0.0
    # img_size = dataloader.dataset.img_size
    img_size = dataloader.dataset.datasets[0].img_size
    for index, batch in enumerate(dataloader):
        loss = 0.0
        inputs = batch['img'].to(device)
        masks = batch['mask'].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        for output, mask in zip(outputs, masks):
            output = transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BILINEAR)(output)
            loss += criterion(
                output,
                mask
            )
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        if (index + 1) % 10 == 0:
            print(f"Epoch {epoch + 1} {index}/{len(dataloader)} current loss: {total_loss / (index + 1)}")
        index += 1

    print(f"Epoch {epoch + 1} done, total loss: {total_loss / len(dataloader)}")
    return total_loss / len(dataloader)


def eval_one_epoch(model, criterion, dataloader, epoch, device):
    model.eval()
    total_loss = 0.0
    # img_size = dataloader.dataset.img_size
    img_size = dataloader.dataset.datasets[0].img_size
    for index, batch in enumerate(dataloader):
        loss = 0.0
        inputs = batch['img'].to(device)
        masks = batch['mask'].to(device)
        with torch.no_grad():
            outputs = model(inputs)
        for output, mask in zip(outputs, masks):
            output = transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BILINEAR)(output)
            loss += criterion(
                output,
                mask
            )
        total_loss += loss.item()
        if (index + 1) % 10 == 0:
            print(f"Epoch {epoch + 1} {index}/{len(dataloader)} current loss: {total_loss / (index + 1)}")
        index += 1

    print(f"Epoch {epoch + 1} done, total loss: {total_loss / len(dataloader)}")
    return total_loss / len(dataloader)


if __name__ == '__main__':
    datapath = '/home/popa/Documents/diplomski_rad/FingertipDetectionAndTrackingForPatternRecognitionOfAirWritingInVideos/segmentation_dataset/hgr1'
    train(batch_size=8, num_classes=1, epochs=15, loss_type='bce')
