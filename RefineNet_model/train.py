import torch

from RefineNet_model.model import *
from datasets.data import *
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
from losses import dice_loss

def train(datapath, batch_size=2, num_classes=1, epochs=2, save_path='/content/drive/MyDrive/refinenet'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = HGR1Dataset(root=datapath, type='train', transform=None)
    valid_dataset = HGR1Dataset(root=datapath, type='val', transform=None)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)
    # criterion = torch.nn.MSELoss(reduction='mean')
    criterion = dice_loss
    model = rf101(num_classes=num_classes)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    for epoch in range(epochs):
        print('---------------------TRANING---------------------')
        train_one_epoch(model=model, optimizer=optimizer, criterion=criterion, dataloader=train_dataloader, epoch=epoch, device=device)
        print('---------------------EVALUATION---------------------')
        eval_one_epoch(model=model, criterion=criterion, dataloader=valid_dataloader, epoch=epoch, device=device)

        torch.save(model.state_dict(), save_path + "_{}.pt".format(epoch + 1))

def train_one_epoch(model, optimizer, criterion, dataloader, epoch, device):
    model.train()
    total_loss = 0.0
    img_size = dataloader.dataset.img_size
    for index, batch in enumerate(dataloader):
        loss = 0.0
        inputs = batch['img'].to(device)
        masks = batch['mask'].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        # outputs = transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BILINEAR)(outputs)
        for output,mask in zip(outputs,masks):
            output = transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BILINEAR)(output)
            # print('output size: {}'.format(output.size()))
            # print('mask size: {}'.format(masks[0].size()))
            loss += criterion(
                # F.interpolate(output, img_size, mode='bilinear', align_corners=False).squeeze(dim=0),
                output,
                mask
            )
        # print('outputs:\n{}'.format(outputs))
        # for output in outputs:
        #     print('outputs size:\n{}'.format(output.size()))
        #     # print('labels size:\n{}'.format(labels.size()))
        #     transforms.ToPILImage()(output).show()
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        if (index+1)%10 == 0:
            print(f"Epoch {epoch+1} {index}/{len(dataloader)} current loss: {total_loss/(index+1)}")
        index += 1

    print(f"Epoch {epoch+1} done, total loss: {total_loss/len(dataloader)}")

def eval_one_epoch(model, criterion, dataloader, epoch, device):
    model.eval()
    total_loss = 0.0
    img_size = dataloader.dataset.img_size
    for index, batch in enumerate(dataloader):
        loss = 0.0
        inputs = batch['img'].to(device)
        masks = batch['mask'].to(device)
        with torch.no_grad():
            outputs = model(inputs)
        for output,mask in zip(outputs,masks):
            output = transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BILINEAR)(output)
            # print('output size: {}'.format(output.size()))
            # print('mask size: {}'.format(masks[0].size()))
            loss += criterion(
                # F.interpolate(output, img_size, mode='bilinear', align_corners=False).squeeze(dim=0),
                output,
                mask
            )
        total_loss += loss.item()
        if (index+1)%10 == 0:
            print(f"Epoch {epoch+1} {index}/{len(dataloader)} current loss: {total_loss/(index+1)}")
        index += 1

    print(f"Epoch {epoch+1} done, total loss: {total_loss/len(dataloader)}")

if __name__=='__main__':
    datapath = '/home/popa/Documents/diplomski_rad/FingertipDetectionAndTrackingForPatternRecognitionOfAirWritingInVideos/segmentation_dataset/hgr1'
    train(datapath=datapath, batch_size=8, num_classes=1, epochs=5)