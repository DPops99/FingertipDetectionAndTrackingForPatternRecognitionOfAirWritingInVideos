import cv2
import torch
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation import deeplabv3_resnet101
from data import *
import matplotlib.pyplot as plt
import numpy as np
import time

def get_model():
    num_classes = 1
    model = deeplabv3_resnet101(pretrained=True, progress=True)
    model.classifier = DeepLabHead(2048, num_classes)
    return model

def train_one_epoch(model, optimizer, criterion, dataloader, epoch, device):
    model.train()
    total_loss = 0.0
    for index, batch in enumerate(dataloader):
        inputs = batch['img'].to(device)
        labels = batch['mask'].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs['out'], labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        if (index+1)%10 == 0:
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
        loss = criterion(outputs['out'], labels)
        total_loss += loss.item()
        if (index+1)%10 == 0:
            print(f"Epoch {epoch+1} current loss: {total_loss/(index+1)}")
        index += 1

    print(f"Epoch {epoch+1} done, total loss: {total_loss/len(dataloader)}")


def predict(model, criterion, dataloader, device):
    model.eval()

    batch_dict = {}
    start_time = time.time()
    print('start prediction')
    for index, batch in enumerate(dataloader):
        if index == 0:
            inputs = batch['img'].to(device)
            labels = batch['mask'].to(device)

            with torch.no_grad():
                outputs = model(inputs)['out']

            # loss = criterion(outputs, labels)
            # batch_dict[index] = loss.item()
            # for label, out in zip(labels, outputs):
            #     output_predictions = out[0]
            #     print(output_predictions.shape)
            #     print(label[0].shape)
            #     palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
            #     colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
            #     colors = (colors % 255).numpy().astype("uint8")
            #
            #     # plot the semantic segmentation predictions of 21 classes in each color
            #     r = Image.fromarray(output_predictions.byte().cpu().numpy())
            #     r.putpalette(colors)
            #     plt.imshow(r)
            #     plt.show()
            #
            #     plt.imshow(label[0])
            #     plt.show()
            break

    print('end time: {}'.format(time.time()-start_time))
    batch_dict = {k: v for k, v in sorted(batch_dict.items(), key=lambda item: item[1])}
    [print(v) for k, v in batch_dict.items()]


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = 'model_second_acc_20.pt'
    checkpoint = torch.load(path, map_location='cpu')
    model = get_model()
    model.load_state_dict(checkpoint['model_state_dict'])
    filepath = "/home/popa/Documents/diplomski_rad/FingertipDetectionAndTrackingForPatternRecognitionOfAirWritingInVideos/segmentation_dataset"
    test_dataset = SegmentationDataset(filepath, type='test')
    test_dataloader = DataLoader(test_dataset, batch_size=4)
    criterion = torch.nn.MSELoss(reduction='mean')
    predict(model=model,criterion=criterion, dataloader=test_dataloader, device=device)


