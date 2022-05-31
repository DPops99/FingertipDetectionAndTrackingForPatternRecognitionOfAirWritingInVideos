import cv2
import torch
from PIL import Image
from fingertip_detection.detection import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_yolo_model():
    model_path = '/home/popa/Documents/diplomski_rad/FingertipDetectionAndTrackingForPatternRecognitionOfAirWritingInVideos/yolov5_best_model/best.pt'
    model = torch.hub.load(
        '/home/popa/Documents/diplomski_rad/FingertipDetectionAndTrackingForPatternRecognitionOfAirWritingInVideos/yolov5',
        'custom',
        path=model_path,
        source='local')
    return model

def get_refinenet_model():
    model_path = '/home/popa/Documents/diplomski_rad/FingertipDetectionAndTrackingForPatternRecognitionOfAirWritingInVideos/trained_models/final_model_100.pt'
    model = rf101(num_classes=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model

def run(model_path='', video_path=''):
    print('-----------------SETTING UP YOLOv5 AND REFINENET MODEL-----------------')
    yolo_model = get_yolo_model()
    yolo_model.eval()
    yolo_model.to(device)
    refinenet_model = get_refinenet_model()
    refinenet_model.eval()
    refinenet_model.to(device)
    print('-----------------STARTING VIDEO-----------------')
    video = cv2.VideoCapture(video_path)
    trackers = cv2.legacy.MultiTracker_create()
    current_frame = 0
    trackers_set = False

    while True:
        ret, frame = video.read()
        if ret:
            if not trackers_set:
                if current_frame%1 == 0:
                    name = './data/frame{}.jpg'.format(current_frame)
                    print('Creating...' + name)
                    cv2.imwrite(name, frame)
                    output = yolo_model(name)
                    selected = output.crop(save=False)
                    # draw_bounding_box(prediction=output, image_path=name, reshaped_xyxy= selected[0]['reshaped_xyxy'])

                    for hand in output.crop(save=False):
                        # cv2.imwrite(name.replace('.jpg','_yolo.jpg').replace('data','cropped_data'), hand['im'])
                        # cv2.imshow('real hand', cv2.cvtColor(hand['im'], cv2.COLOR_BGR2RGB))
                        # cv2.waitKey()
                        mask = get_segmented_hand(image=cv2.cvtColor(hand['im'], cv2.COLOR_BGR2RGB), model=refinenet_model)
                        fingertips = signature(mask=mask, image_real=cv2.cvtColor(hand['im'], cv2.COLOR_BGR2RGB))
                        fingertips = fix_coordinates(croped_im_coordinates=hand['reshaped_xyxy'], fingertips_coordinates=fingertips)
                        trackers = set_trackers(trackers, fingertips, frame)
                        trackers_set = True
                        # show_img(image_path=name, fingertips=fingertips)

            (success, boxes) = trackers.update(frame)
            for box in boxes:
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            current_frame += 1
            cv2.imshow("Frame", frame)
            k = cv2.waitKey(1) & 0xff
            if k == 27: break
        else:
            break

        # Release all space and windows once done
    video.release()
    cv2.destroyAllWindows()

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
    cv2.rectangle(original_img, (reshaped_xyxy[0][0].item(), reshaped_xyxy[0][1].item()), (reshaped_xyxy[0][2].item(), reshaped_xyxy[0][3].item()), (0, 255, 0), 1)
    cv2.imshow('img', original_img)
    cv2.waitKey()


def fix_coordinates(croped_im_coordinates, fingertips_coordinates):
    xmin = int(croped_im_coordinates[0][0].item())
    ymin = int(croped_im_coordinates[0][1].item())
    print('xmin:{} ymin:{}'.format(xmin,ymin))
    print('before')
    for point in fingertips_coordinates:
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
        xmin = point[0][0]-5
        ymin = point[0][1]-5
        xmax = point[0][0]+5
        ymax = point[0][1]+5
        box_height = ymax-ymin
        box_width = xmax - xmin
        bbox = (xmin, ymin, box_width, box_height)
        bounding_boxes.append(bbox)
    return bounding_boxes





if __name__=='__main__':
    run(video_path='/home/popa/Documents/diplomski_rad/FingertipDetectionAndTrackingForPatternRecognitionOfAirWritingInVideos/webcam_data/videos/hand_only_test2.mp4')