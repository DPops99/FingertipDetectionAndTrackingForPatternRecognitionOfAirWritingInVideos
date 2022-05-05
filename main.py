from train import *
import torch

def train(load_pretrained=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model()
    model.to(device)
    filepath = "/home/popa/Documents/diplomski_rad/FingertipDetectionAndTrackingForPatternRecognitionOfAirWritingInVideos/segmentation_dataset"
    save_path = "/home/popa/Documents/diplomski_rad/FingertipDetectionAndTrackingForPatternRecognitionOfAirWritingInVideos/model"
    train_dataset = SegmentationDataset(filepath, type='train')
    valid_dataset = SegmentationDataset(filepath, type='valid')
    train_dataloader = DataLoader(train_dataset, batch_size=2)
    valid_dataloader = DataLoader(valid_dataset, batch_size=2)

    epochs = 20
    start_epoch = 0
    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    if load_pretrained:
        path = 'seg_model_9.pt'
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']+1


    for epoch in range(epochs):
        print("----------------------BEGIN TRANING EPOCH {}----------------------".format(epoch+1))
        train_one_epoch(model=model, optimizer=optimizer, criterion=criterion, dataloader=train_dataloader, epoch=epoch, device=device)
        print("----------------------END TRANING EPOCH {}----------------------".format(epoch+1))
        print("----------------------BEGIN VALIDATION EPOCH {}----------------------".format(epoch+1))
        eval_one_epoch(model=model, criterion=criterion, dataloader=valid_dataloader, epoch=epoch, device=device)
        print("----------------------END VALIDATION EPOCH {}----------------------".format(epoch+1))

        if (epoch+1)%5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, save_path+"_{}.pt".format(epoch+start_epoch+1))


if __name__=="__main__":
    train(load_pretrained=True)