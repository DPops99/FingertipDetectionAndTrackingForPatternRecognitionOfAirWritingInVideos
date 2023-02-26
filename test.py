from fingertip_detection.detection import detect_fingers
from RefineNet_model.utils import get_refinenet_model
from fingertip_detection.detection import get_yolo_model
from os.path import join
import torch
import cv2

def test_detection(img_path, yolo_model_path, seg_model_path, device):
    yolo_model = get_yolo_model(yolo_model_path)
    seg_model = get_refinenet_model(model_path=seg_model_path, device=device)
    yolo_model.to(device)
    seg_model.to(device)
    seg_values, finger_values = detect_fingers(img_path=img_path,
                    yolo_model=yolo_model,
                    seg_model=seg_model)
    for mask, fingers in zip(seg_values, finger_values):
        cv2.imshow('Mask', mask)
        cv2.waitKey()
        


if __name__=='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_path = './data/test_data/test_2.jpg'
    yolo_model_path = join('trained_models', 'yolov5', 'best.pt')
    seg_model_path = join('trained_models', 'refine_net', 'final_model_100.pt')
    test_detection(
        img_path=img_path,
        yolo_model_path=yolo_model_path,
        seg_model_path=seg_model_path,
        device=device
    )
