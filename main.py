import cv2
from os.path import join, split
import numpy as np
import torch
from PIL import Image
from fingertip_detection.detection import get_segmented_hand, signature
from RefineNet_model.utils import get_refinenet_model
from fingertip_detection.detection import get_yolo_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run(video_path, yolo_path, refine_path):
    print('-----------------SETTING UP YOLOv5 AND REFINENET MODEL-----------------')
    yolo_model = get_yolo_model(model_path=yolo_path)
    yolo_model.eval()
    yolo_model.to(device)
    refinenet_model = get_refinenet_model(model_path=refine_path, device=device)
    refinenet_model.eval()
    refinenet_model.to(device)
    print('-----------------STARTING VIDEO-----------------')
    video = cv2.VideoCapture(video_path)
    video_frame_rate = int(video.get(cv2.CAP_PROP_FPS))
    trackers = cv2.legacy.MultiTracker_create()
    current_frame = 0
    trackers_set = False
    frames = []

    while True:
        ret, frame = video.read()
        if ret:
            if not trackers_set:
                if current_frame%1 == 0:
                    name = './data/yolo_frames/frame{}.jpg'.format(current_frame)
                    print('Creating...' + name)
                    cv2.imwrite(name, frame)
                    output = yolo_model(name)
                    selected = output.crop(save=False)

                    for hand in output.crop(save=False):
                        # if len(hand['im']) == 0:
                        #     continue
                        prev_shape = hand['im'].shape
                        print('hand: {}'.format(hand))
                        image = cv2.resize(hand['im'], (224, 224))
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        mask = get_segmented_hand(image=image, 
                                                    model=refinenet_model)
                        print('prev shape: {}'.format(prev_shape))
                        mask = cv2.resize(mask, prev_shape[:2])
                        # cv2.imshow('Mask', mask)
                        fingertips = signature(mask=mask, 
                                                image_real=image)
                        if len(fingertips) == 0:
                            continue
                        print('fingertips: {}'.format(fingertips))
                        fingertips = fix_coordinates(croped_im_coordinates=hand['reshaped_xyxy'], 
                                                    fingertips_coordinates=fingertips)
                        trackers = set_trackers(trackers, fingertips, frame)
                        trackers_set = True

            if not trackers:
                continue
            (success, boxes) = trackers.update(frame)
            for box in boxes:
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            current_frame += 1
            # cv2.imshow("Frame", frame)
            # k = cv2.waitKey(1) & 0xff
            # if k == 27: break
            frames.append(frame.copy())
        else:
            break

    # Release all space and windows once done
    video.release()
    cv2.destroyAllWindows()
    print('len(frames) : {}'.format(len(frames)))
    
    save_video(frames=frames, 
                input_video_path=video_path,
                video_frame_rate=video_frame_rate)

def draw_bounding_box(prediction, image_path, reshaped_xyxy):
    print(prediction.xyxy[0])
    xmin = int(prediction.xyxy[0][0][0].item())
    ymin = int(prediction.xyxy[0][0][1].item())
    xmax = int(prediction.xyxy[0][0][2].item())
    ymax = int(prediction.xyxy[0][0][3].item())

    img = cv2.imread(image_path)
    cv2.circle(img, (xmin,ymin),3, (0,255,0),5)
    cv2.rectangle(img, (xmin,ymin), (xmax,ymax), (255,0,0),5)
    print(prediction.imgs[0])
    original_img = cv2.cvtColor(np.array(prediction.imgs[0]), cv2.COLOR_BGR2RGB)
    cv2.rectangle(original_img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 1)
    cv2.rectangle(original_img, (reshaped_xyxy[0][0].item(), reshaped_xyxy[0][1].item()), 
                (reshaped_xyxy[0][2].item(), reshaped_xyxy[0][3].item()), (0, 255, 0), 1)
    cv2.imshow('img', original_img)
    cv2.waitKey()


def fix_coordinates(croped_im_coordinates, fingertips_coordinates):
    xmin = int(croped_im_coordinates[0][0].item())
    ymin = int(croped_im_coordinates[0][1].item())
    print('xmin:{} ymin:{}'.format(xmin,ymin))
    print('before')
    for point in fingertips_coordinates:
        if len(point) == 0:
            continue 
        print(point[0])
        point[0][0] += xmin
        point[0][1] += ymin

    return fingertips_coordinates

def show_img(image_path, fingertips):
    img = cv2.imread(image_path)
    print('after')
    for point in fingertips:
        print(point[0])
        cv2.circle(img, point[0], 3, (0, 0, 255), 5)
    cv2.imshow('img',img)
    cv2.waitKey(0)

def set_trackers(trackers, fingertips, frame):
    bounding_boxes = get_bounding_boxes(fingertips)
    for bbox in bounding_boxes:
        tracker = cv2.legacy.TrackerCSRT_create()
        trackers.add(tracker, frame, bbox)
    return trackers

def get_bounding_boxes(fingertips):
    bounding_boxes = []
    for point in fingertips:
        if len(point) == 0:
            continue
        xmin = point[0][0]-5
        ymin = point[0][1]-5
        xmax = point[0][0]+5
        ymax = point[0][1]+5
        box_height = ymax-ymin
        box_width = xmax - xmin
        bbox = (xmin, ymin, box_width, box_height)
        bounding_boxes.append(bbox)
    return bounding_boxes

def save_video(frames, input_video_path, video_frame_rate):
    prefix = split(input_video_path)[1]
    height, width, layers = frames[1].shape
    print('frames: {}'.format(frames[0].shape))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    save_path = join('results', prefix.replace('.mp4','_result.mp4'))
    video = cv2.VideoWriter(save_path, 
                            fourcc, 
                            video_frame_rate, 
                            (width, height))
    print('saving video to: {}'.format(save_path))

    for j in range(len(frames)):
        video.write(frames[j])
        if cv2.waitKey(20) == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


if __name__=='__main__':
    video_path = 'data/input_video/test_1.mp4'
    yolo_path = join('trained_models', 'yolov5', 'best.pt')
    refine_path = join('trained_models', 'refine_net', 'checkpoint_10.pt')
    run(video_path=video_path, yolo_path=yolo_path, refine_path=refine_path)