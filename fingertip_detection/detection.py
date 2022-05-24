import cv2
import numpy as np
from torchvision.transforms import transforms
import math
import matplotlib.pyplot as plt
import torch
from RefineNet_model.train import *
from PIL import Image
import torch.nn.functional as F

def get_image_with_contours(image):
    print(f"image: {image}")
    gray = image.copy()

    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    # print(f"gray: {gray.shape}")

    contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"len of contours: {len(contours)}")
    hull_list = []
    largest_contour = contours[0]
    for index, contour in enumerate(contours):
        hull = cv2.convexHull(contour)
        hull_list.append(hull)
        if cv2.contourArea(largest_contour) < cv2.contourArea(contour):
            largest_contour = contour

    largest_hull = cv2.convexHull(largest_contour)
    largest_hull_indicies = cv2.convexHull(largest_contour, returnPoints=False)
    defects = cv2.convexityDefects(largest_contour, largest_hull_indicies)
    print(f"len of defects: {len(defects)}")
    # all_hulls_mask = mask.copy()
    # all_contours_mask = mask.copy()
    # largest_contour_mask = mask.copy()
    # largest_hull_mask = mask.copy()

    # img_show = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)

    cv2.drawContours(image, [largest_contour], -1, (0, 255, 0), 3)
    cv2.drawContours(image, [largest_hull], -1, (0, 0, 255), 3)



    # cv2.imwrite(hull_path, img)
    for defect in defects:
        start_index, end_index, farthest_index, depth = defect[0]

        start_point = tuple(largest_contour[start_index][0])
        end_point = tuple(largest_contour[end_index][0])
        farthest_point = tuple(largest_contour[farthest_index][0])
        triangle_area = cv2.contourArea(np.array([start_point, end_point, farthest_point]))
        if triangle_area > 1000:
            print(f"triangle_area: {triangle_area}")
            cv2.circle(image, farthest_point, 5, (255, 0, 0), -1)

    # cv2.imwrite(hull_points_path, img)
    # for hull in largest_hull:
    #     print(f"hull: {hull[0]}")
    #     cv2.circle(img, hull[0], 5, (255, 0, 0), -1)

    print(f"defects shape: {defects.shape}")
    print(f"defects shape[0]: {defects.shape[0]}")

    cv2.imshow('image_final', image)
    cv2.waitKey(0)
    return

    # cv2.drawContours(largest_hull_mask, [largest_hull], -1, (0, 0, 255), 3)
    # cv2.drawContours(all_contours_mask, contours, -1, (0, 0, 255), 3)
    # cv2.drawContours(all_hulls_mask, hull_list, -1, (0, 0, 255), 3)

    # cv2.imshow("largest_contour_mask",largest_contour_mask)
    # cv2.imshow("largest_hull_maskk", largest_hull_mask)
    # cv2.imshow("all_contours_mask", all_contours_mask)
    # cv2.imshow("all_hulls_mask", all_hulls_mask)

    # cv2.imshow("img_show", img)

    # cv2.waitKey(0)

    # cv2.drawContours(mask, [largest_hull], -1, (0, 0, 255), 3)
    # cv2.imshow("hull",mask)
    # cv2.waitKey(0)

def get_image(image_path):
    return cv2.imread(image_path)

def get_hand_center(mask):
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
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
    # largest_hull = cv2.convexHull(largest_contour)
    # print(mask.shape)
    # print(type(mask))
    # print(mask.shape)
    # blank_array = np.zeros(mask.shape)
    # cv2.drawContours(blank_array, [largest_contour], -1, (0, 255, 0), 3)
    # cv2.drawContours(mask_rgb, [largest_contour], -1, (0, 255, 0), 3)
    # cv2.drawContours(blank_array, [largest_hull], -1, (0, 0, 255), 3)
    # cv2.drawContours(mask_rgb, [largest_hull], -1, (0, 0, 255), 3)
    # cv2.circle(blank_array, (center_X, center_Y), 1, (255, 255, 255), -1)
    # cv2.circle(mask_rgb, (center_X, center_Y), 7, (255, 255, 255), -1)
    #
    #
    # print(blank_array.shape)
    # cv2.imshow('blank',blank_array)
    # cv2.imshow('mask', mask_rgb)
    # cv2.waitKey(0)

    return (center_X,center_Y)

def get_contour(mask):
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    print(gray.shape)
    print(np.array(gray).shape)
    _,thresh = cv2.threshold(gray,200,255, cv2.THRESH_BINARY)
    # for x in gray:
    #     for y in x:
    #         if y != 0:
    #             y = 255
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print('len of contours: {}'.format(len(contours)))
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
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(contour[s][0])
        end = tuple(contour[e][0])
        far = tuple(contour[f][0])
        triangle_area = cv2.contourArea(np.array([start, end, far]))
        if triangle_area > 1000:
            print(f"triangle_area: {triangle_area}")
            far_points.append(far)
    return far_points

def get_hand_center(contour):
    M = cv2.moments(contour)
    center_X = int(M["m10"] / M["m00"])
    center_Y = int(M["m01"] / M["m00"])
    return (center_X, center_Y)

def signature(mask):

    # mask = 255*(np.logical_xor(mask, np.ones(mask.shape)).astype('uint8'))
    contour, contours, hierarchy = get_contour(mask)
    hull = get_hull(contour)
    defects = get_defects(contour,hull)
    far_points = get_far_points(contour, defects)
    hand_center = get_hand_center(contour)
    distances = []
    distance = []
    all_dist = []
    hull_distances = []
    fingertip_points = []
    #SEPERATE CONTOURS BY DEFECT POINTS
    for point in contour:
        distance.append(math.dist(hand_center, point[0]))
        all_dist.append(math.dist(hand_center, point[0]))
        for far_point in far_points:
            if np.array_equal(far_point, point[0]):
                distances.append(distance)
                # max_index = get_max_value_index(distance)
                # for dist in distances[:-1]:
                #     max_index += len(dist)
                # fingertip = contour[max_index]
                # fingertip_points.append(fingertip)
                distance = []
    print(fingertip_points)

    # #CUT THE LARGEST DISTANCE CONTOUR INTO 3 PARTS
    longest_dist = get_longest_distance(distances)
    size = int(len(longest_dist)*0.33)
    distances.remove(longest_dist)
    distances.append(longest_dist[:size])
    distances.append(longest_dist[-size:])

    for index, dist in enumerate(distances):
        max_index = get_max_value_index(dist)
        for prev_dist in distances[:index]:
            max_index += len(prev_dist)
        if index > 2:
            max_index += size
        fingertip = contour[max_index]
        fingertip_points.append(fingertip)



    print(distances)
    max_point = contour[np.argmax(distances)]
    print(max_point)
    fig, axs = plt.subplots(len(distances))
    for index, dist in enumerate(distances):
        x = [i for i in range(len(dist))]
        axs[index].plot(x,dist)
    plt.show()


    cv2.drawContours(mask, [contour], -1, (0, 255, 0), 3)
    # cv2.drawContours(mask, [hull], -1, (0, 0, 255), 3)
    cv2.circle(mask, hand_center, 1, (0, 0, 255), -1)
    for point in far_points:
        cv2.circle(mask, point, 3, (0, 0, 255), -1)
    for point in fingertip_points:
        cv2.circle(mask, point[0], 3, (255, 0, 0), -1)
    # cv2.circle(mask, max_point[0], 5, (255, 0, 0), -1)
    # for i in range(len(contours)):
    #     color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))
    #     cv2.drawContours(mask, contours, i, color, 2, cv2.LINE_8, hierarchy, 0)
    cv2.imshow('mask',mask)
    image_real = get_image('/home/popa/Documents/diplomski_rad/FingertipDetectionAndTrackingForPatternRecognitionOfAirWritingInVideos/fingertip_detection/runs/detect/exp3/crops/hand/2022-01-19-215301_3.jpg')
    cv2.circle(image_real, max_point[0], 5, (255, 0, 0), -1)
    cv2.imshow('image_real',image_real)
    cv2.waitKey()

def get_longest_distance(distances):
    return max(distances, key= lambda x:len(x))

def get_max_value_index(distances):
    return np.argmax(distances)

def get_yolo_pic():
    model_path = '/home/popa/Documents/diplomski_rad/FingertipDetectionAndTrackingForPatternRecognitionOfAirWritingInVideos/yolov5_best_model/best.pt'
    model = torch.hub.load('/home/popa/Documents/diplomski_rad/FingertipDetectionAndTrackingForPatternRecognitionOfAirWritingInVideos/yolov5', 'custom',
                           path='/home/popa/Documents/diplomski_rad/FingertipDetectionAndTrackingForPatternRecognitionOfAirWritingInVideos/yolov5_best_model/best.pt',
                           source='local')
    # model.load_state_dict(torch.load(model_path, map_location='cpu')['model'].state_dict())
    # model = model.fuse().autoshape()
    img = '/home/popa/Pictures/Webcam/2022-01-19-215301_3.jpg'
    output = model(img)
    print(output)
    output.crop()

def get_segmented_hand():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # SET UP MODEL
    model_path = '/home/popa/Documents/diplomski_rad/FingertipDetectionAndTrackingForPatternRecognitionOfAirWritingInVideos/trained_models/final_model_100.pt'
    model = rf101(num_classes=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # SET UP IMAGE
    img_path = '/home/popa/Documents/diplomski_rad/FingertipDetectionAndTrackingForPatternRecognitionOfAirWritingInVideos/fingertip_detection/runs/detect/exp3/crops/hand/2022-01-19-215301_3.jpg'
    image = Image.open(img_path).convert('RGB')
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
    out_mask.show()
    out_mask.save(img_path.replace('.jpg','_segmented.jpg'))

def test_cv2_thresh():
    img_path = '/home/popa/Documents/diplomski_rad/FingertipDetectionAndTrackingForPatternRecognitionOfAirWritingInVideos/fingertip_detection/runs/detect/exp3/crops/hand/2022-01-19-215301_3.jpg'
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img, 190, 255, cv2.THRESH_BINARY)
    print(thresh)
    cv2.imshow('thresh',thresh)
    cv2.imshow('img',img)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = max(contours, key=lambda x: cv2.contourArea(x))
    cv2.drawContours(thresh, [contours], -1, (255, 255, 0), 2)
    cv2.imshow("contours", img)
    cv2.waitKey(0)


if __name__=='__main__':
    image_path = '/home/popa/Documents/diplomski_rad/FingertipDetectionAndTrackingForPatternRecognitionOfAirWritingInVideos/fingertip_detection/runs/detect/exp3/crops/hand/2022-01-19-215301_3_segmented.jpg'
    image = get_image(image_path)
    signature(image)
    # get_segmented_hand()
    # get_yolo_pic()

    # test_cv2_thresh()