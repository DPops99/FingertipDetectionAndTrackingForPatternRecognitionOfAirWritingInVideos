from RefineNet_model.model import *
from datasets.data import *
from torch.utils.data.dataloader import DataLoader

def train(datapath, batch_size=2, num_classes=1, epochs=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = HGR1Dataset(root=datapath, type='train', transform=None)
    valid_dataset = HGR1Dataset(root=datapath, type='train', transform=None)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)
    criterion = torch.nn.MSELoss(reduction='mean')
    model = rf101(num_classes=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    for epoch in range(epochs):
        print('---------------------TRANING---------------------')
        train_one_epoch(model=model, optimizer=optimizer, criterion=criterion, dataloader=train_dataloader, epoch=epoch, device=device)
        print('---------------------EVALUATION---------------------')
        eval_one_epoch(model=model, criterion=criterion, dataloader=valid_dataloader, epoch=epoch, device=device)


def train_one_epoch(model, optimizer, criterion, dataloader, epoch, device):
    model.train()
    total_loss = 0.0
    img_size = dataloader.dataset.img_size
    for index, batch in enumerate(dataloader):
        inputs = batch['img'].to(device)
        labels = batch['mask'].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        outputs = transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BILINEAR)(outputs)
        # print('outputs:\n{}'.format(outputs))
        # print('outputs size:\n{}'.format(outputs.size()))
        # print('labels size:\n{}'.format(labels.size()))
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        if (index+1)%1 == 0:
            print(f"Epoch {epoch+1} current loss: {total_loss/(index+1)}")
        index += 1

    print(f"Epoch {epoch+1} done, total loss: {total_loss/len(dataloader)}")

def eval_one_epoch(model, criterion, dataloader, epoch, device):
    model.eval()
    total_loss = 0.0
    for index, batch in enumerate(dataloader):
        inputs = batch['img'].to(device)
        labels = batch['mask'].to(device)
        with torch.no_grad():
            outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        if (index+1)%1 == 0:
            print(f"Epoch {epoch+1} current loss: {total_loss/(index+1)}")
        index += 1

    print(f"Epoch {epoch+1} done, total loss: {total_loss/len(dataloader)}")

if __name__=='__main__':
    datapath = '/home/popa/Documents/diplomski_rad/FingertipDetectionAndTrackingForPatternRecognitionOfAirWritingInVideos/segmentation_dataset/hgr1'
    train(datapath=datapath, batch_size=1, num_classes=1, epochs=2)