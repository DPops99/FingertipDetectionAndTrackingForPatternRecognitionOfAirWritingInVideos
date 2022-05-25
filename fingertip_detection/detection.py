import cv2
import numpy as np
from torchvision.transforms import transforms
import math
import matplotlib.pyplot as plt
import torch
from RefineNet_model.train import *
from PIL import Image
import torch.nn.functional as F


def get_image(image_path):
    return cv2.imread(image_path)

def get_hand_center(mask):
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print('len of contours: {}'.format(len(contours)))
    largest_contour = contours[0]
    for index, contour in enumerate(contours):
        if cv2.contourArea(largest_contour) < cv2.contourArea(contour):
            largest_contour = contour

    M = cv2.moments(largest_contour)
    center_X = int(M["m10"] / M["m00"])
    center_Y = int(M["m01"] / M["m00"])

    print(np.array(largest_contour).shape)
    return (center_X,center_Y)

def get_contour(mask):
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    print(gray.shape)
    print(np.array(gray).shape)
    _,thresh = cv2.threshold(gray,200,255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = contours[0]
    for index, contour in enumerate(contours):
        print('len: {}'.format(len(contour)))
        if cv2.contourArea(largest_contour) < cv2.contourArea(contour):
            largest_contour = contour
    return largest_contour, contours, hierarchy

def get_hull(contour):
    return cv2.convexHull(contour, returnPoints=False)

def get_defects(contour, hull):
    return cv2.convexityDefects(contour,hull)

def get_far_points(contour, defects):
    far_points = []
    far_indicies = []
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(contour[s][0])
        end = tuple(contour[e][0])
        far = tuple(contour[f][0])
        triangle_area = cv2.contourArea(np.array([start, end, far]))
        if triangle_area > 1000:
            print(f"triangle_area: {triangle_area}")
            far_points.append(far)
            far_indicies.append(f)
    return far_points, far_indicies

def get_hand_center(contour):
    M = cv2.moments(contour)
    center_X = int(M["m10"] / M["m00"])
    center_Y = int(M["m01"] / M["m00"])
    return (center_X, center_Y)

def signature(mask,image_real):

    contour, contours, hierarchy = get_contour(mask)
    hull = get_hull(contour)
    defects = get_defects(contour,hull)
    far_points, far_indicies = get_far_points(contour, defects)
    hand_center = get_hand_center(contour)

    real_finger_contours = get_finger_contours(contour=contour, far_indicies=far_indicies)
    fixed_finger_contours = fix_finger_contours(finger_contours=real_finger_contours)
    real_distences = get_finget_contour_dist(hand_center=hand_center, finger_contours=fixed_finger_contours)
    real_fingertips = get_fingertips(finger_contour_dist=real_distences, finger_contour=fixed_finger_contours)

    # fig, axs = plt.subplots(len(real_distences))
    # for index, dist in enumerate(real_distences):
    #     x = [i for i in range(len(dist))]
    #     axs[index].plot(x,dist)
    # plt.show()

    # image_real = get_image('/home/popa/Documents/diplomski_rad/FingertipDetectionAndTrackingForPatternRecognitionOfAirWritingInVideos/fingertip_detection/runs/detect/exp3/crops/hand/2022-01-19-215301_3.jpg')
    image_real = cv2.cvtColor(image_real,cv2.COLOR_BGR2RGB)
    cv2.drawContours(mask, [contour], -1, (0, 255, 0), 3)
    cv2.circle(mask, hand_center, 1, (0, 0, 255), -1)
    for point in far_points:
        cv2.circle(mask, point, 3, (0, 0, 255), -1)
    for point in real_fingertips:
        cv2.circle(mask, point[0], 3, (255, 0, 0), -1)
        cv2.circle(image_real, point[0], 3, (255, 0, 0), -1)
    # cv2.imshow('mask',mask)
    cv2.imshow('image_real',image_real)
    cv2.waitKey()

def get_longest_distance(distances):
    return max(distances, key= lambda x:len(x))

def get_max_value_index(distances):
    return np.argmax(distances)

def fix_finger_contours(finger_contours):
    longest_contour = get_longest_distance(finger_contours)
    finger_contours.remove(longest_contour)
    size = int(len(longest_contour)*0.33)
    finger_contours.append(longest_contour[:size])
    finger_contours.append(longest_contour[-size:])
    return finger_contours

def get_finger_contours(contour, far_indicies):
    finger_contours = []
    for index in range(len(far_indicies)-1):
        current_far_index = far_indicies[index]
        next_far_index = far_indicies[index+1]
        finger_contours.append(contour[current_far_index:next_far_index])

    last_index = far_indicies[-1]
    last_contour = np.concatenate((contour[last_index:], contour[:far_indicies[0]]))
    finger_contours.append(last_contour)

    return finger_contours

def get_finget_contour_dist(hand_center, finger_contours):
    contour_dist = []
    for finger in finger_contours:
        current_dist = []
        for point in finger:
            current_dist.append(math.dist(hand_center, point[0]))
        contour_dist.append(current_dist)
    return contour_dist

def get_fingertips(finger_contour_dist, finger_contour):
    fingertips = []
    for finger_dist, finger in zip(finger_contour_dist, finger_contour):
        contour_dist_index = np.argmax(finger_dist)
        fingertips.append(finger[contour_dist_index])
    return fingertips

def get_fingertip_indicies(fingertips, contour):
    fingertip_indicies = []
    for index, point in enumerate(contour):
        for finger_point in fingertips:
            if np.array_equal(finger_point[0], point[0]):
                fingertip_indicies.append(index)

    return fingertip_indicies


def get_yolo_pic():
    model_path = '/home/popa/Documents/diplomski_rad/FingertipDetectionAndTrackingForPatternRecognitionOfAirWritingInVideos/yolov5_best_model/best.pt'
    model = torch.hub.load('/home/popa/Documents/diplomski_rad/FingertipDetectionAndTrackingForPatternRecognitionOfAirWritingInVideos/yolov5', 'custom',
                           path=model_path,
                           source='local')
    img = '/home/popa/Pictures/Webcam/2022-01-19-215301_3.jpg'
    output = model(img)
    hands = []
    for hand in output.crop(save=False):
        hands.append(cv2.cvtColor(hand['im'], cv2.COLOR_BGR2RGB))
    # img = output.crop(save=False)[0]['im']
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # cv2.imshow('img', img)
    # cv2.waitKey()
    return hands


def get_segmented_hand(image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # SET UP MODEL
    model_path = '/home/popa/Documents/diplomski_rad/FingertipDetectionAndTrackingForPatternRecognitionOfAirWritingInVideos/trained_models/final_model_100.pt'
    model = rf101(num_classes=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # SET UP IMAGE
    # img_path = '/home/popa/Documents/diplomski_rad/FingertipDetectionAndTrackingForPatternRecognitionOfAirWritingInVideos/fingertip_detection/runs/detect/exp3/crops/hand/2022-01-19-215301_3.jpg'
    # image = Image.open(img_path).convert('RGB')
    image = Image.fromarray(image).convert('RGB')
    image = transforms.ToTensor()(image).to(device)
    image = torch.unsqueeze(image, dim=0)
    image.to(device)

    #GET PREDICTION
    pred_mask = model(image)
    print(image.shape[2:])
    pred_mask = transforms.Resize(image.shape[2:], interpolation=transforms.InterpolationMode.BICUBIC)(pred_mask)
    pred_mask = torch.squeeze(pred_mask, dim=0)
    pred_mask = F.sigmoid(pred_mask)
    pred_mask[pred_mask>=0.5]=1.0
    pred_mask[pred_mask<0.5]=0.0
    print(pred_mask)
    out_mask = transforms.ToPILImage()(pred_mask)
    # out_mask.show()
    out_mask = np.asarray(out_mask)
    out_mask = cv2.cvtColor(out_mask, cv2.COLOR_GRAY2BGR)
    # out_mask.save(img_path.replace('.jpg','_segmented.jpg'))
    return out_mask


def detect_fingers():
    hands = get_yolo_pic()
    for hand in hands:
        mask = get_segmented_hand(image=hand)
        signature(mask=mask, image_real=hand)

if __name__=='__main__':
    detect_fingers()