import cv2
import numpy as np

def get_countours(image):
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

if __name__=='__main__':
    image_path = '/home/popa/Documents/diplomski_rad/FingertipDetectionAndTrackingForPatternRecognitionOfAirWritingInVideos/segmentation_model_results/RefineNet/BCELoss_15_epochs/img_0_thresh_0.5.jpg'
    image = get_image(image_path)
    get_countours(image)