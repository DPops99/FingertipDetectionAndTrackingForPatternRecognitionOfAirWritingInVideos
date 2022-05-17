from model import rf101
import torch
from datasets.data import *
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import torch.nn.functional as F

def eval_model():
    datapath = '/content/hgr1'
    batch_size =1
    num_classes = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = HGR1Dataset(root=datapath, type='test', transform=None)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    model = rf101(num_classes=num_classes)
    model.load_state_dict(torch.load('/content/drive/MyDrive/refinenet_15.pt'))
    model.to(device)

    predict(model, test_dataloader, device)

def predict(model, dataloader,  device):
    model.eval()
    img_size = dataloader.dataset.img_size
    for index, batch in enumerate(dataloader):
        if index < 3:
            inputs = batch['img'].to(device)
            masks = batch['mask'].to(device)
            with torch.no_grad():
                outputs = model(inputs)
            for out ,mask in zip(outputs ,masks):
                out = transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BILINEAR)(out)
                out = F.sigmoid(out)
                out[out>=0.65] =1.0
                out[out<0.65] =0.0
                print(out)
                out_mask = transforms.ToPILImage()(out)
                out_mask.save('/content/img_{}.jpg'.format(index))
        else:
            break

def inference(model_path = '/content/drive/MyDrive/refinenet_15.pt', img_path='/content/hgr1/hgr1_images/original_images/O_P_hgr1_id11_2.jpg', save_path='/content/O_P_hgr1_id11_2.jpg'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = rf101(num_classes=1)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    image = Image.open(img_path)
    image = transforms.ToTensor()(image).to(device)
    image = torch.unsqueeze(image,dim=0)
    image.to(device)
    pred_mask = model(image)
    pred_mask = transforms.Resize((371, 462), interpolation=transforms.InterpolationMode.BILINEAR)(pred_mask)
    pred_mask = torch.squeeze(pred_mask,dim=0)
    pred_mask = F.sigmoid(pred_mask)
    pred_mask[pred_mask>=0.65]=1.0
    pred_mask[pred_mask<0.65]=0.0
    print(pred_mask)
    out_mask = transforms.ToPILImage()(pred_mask)
    out_mask.save(save_path)