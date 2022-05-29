import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import json
import os

def empty_callback(x):
    pass


def round_up_to_odd(f):
    f = int(np.ceil(f))
    return f + 1 if f % 2 == 0 else f

def fillhole(input_image):
    im_flood_fill = input_image.copy()
    h, w = input_image.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    im_flood_fill = im_flood_fill.astype("uint8")
    cv.floodFill(im_flood_fill, mask, (0, 0), 255)
    im_flood_fill_inv = cv.bitwise_not(im_flood_fill)
    img_out = input_image | im_flood_fill_inv
    return img_out

def calculateScore(gt, u, n):
    """
        @:param gt - ground truth (wartość wzorcowa)
        @:param u - user score (wartość wyznaczona przez program)
        @:param n - number of images (liczba obrazów)
        @:return score_for_all_images - result of calculation
    """

    score = []
    for i in gt:
        dif_res = []
        for j in range(len(gt[i])):
            dif_res.append(abs(gt[i][j] - u[i][j]))
        # print(dif_res)
        score_one_image = float(sum(dif_res)) / float(sum(gt[i]))
        print("score: ", score_one_image)
        score.append(score_one_image)
        # print(score_one_image)
        dif_res.clear()
    score_for_all_images = float(sum(score)) / float(n)
    return score_for_all_images


# def sharpen_image(input_image):
#     kernel = np.array([[0, -1, 0],
#                        [-1, 5, -1],
#                        [0, -1, 0]])
#     image_sharp = cv.filter2D(src=input_image, ddepth=-1, kernel=kernel)
#     return image_sharp

def match_contours(filtered_contours, mask_color):
    mask = cv.cvtColor(mask_color, cv.COLOR_BGR2GRAY)
    mask = cv.medianBlur(mask, 7)

    result = 0
    data = []

    contours_mask, hierarchy_mask = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    cv.fillPoly(mask_color, contours_mask, (255, 255, 0))

    # cv.imshow('mask_7', mask_color)
    # cv.waitKey(0)
    print(len(contours_mask))

    # print(len(filtered_contours))
    # print(len(contours_mask))
    for idx, contour in enumerate(filtered_contours):
        ret = cv.matchShapes(contour, contours_mask[0], 3, 0.0)
        if ret < 0.25:
            print(idx, ret)
            result += 1
            data.append((idx, ret))
    return result, data

def choose_lower(data_1, data_2):
    for idx, x in enumerate(data_1):
        for idy, y in enumerate(data_2):
            if x is not None and y is not None:
                if x[0] == y[0]:
                    if x[1] < y[1]:
                        data_2[idy] = None
                        continue
                    if y[1] < x[1]:
                        data_1[idx] = None
    data_1[:] = list(filter(None, data_1))
    data_2[:] = list(filter(None, data_2))


def check_convex(data_1, data_2, filtered_contours, hull):
    for idx, x in enumerate(data_1):
        for idy, y in enumerate(data_2):
            if x is not None and y is not None:
                if x[0] == y[0]:
                    hull_area = cv.contourArea(hull[x[0]])
                    contour_area = cv.contourArea(filtered_contours[x[0]])
                    if hull_area - contour_area > 600:
                        data_2[idy] = None
                    else:
                        data_1[idx] = None
    data_1[:] = list(filter(None, data_1))
    data_2[:] = list(filter(None, data_2))



def main():

    results_all= {}
    for filename in os.listdir('./projekt/'):

        img = cv.imread(os.path.join('./projekt/',filename))
        # img = cv.imread('./projekt/img_007.jpg')
        img_scaled = cv.resize(img, fx=0.3, fy=0.3, dsize=None)
        gray_scaled = cv.cvtColor(img_scaled, cv.COLOR_BGR2GRAY)
        hsv = cv.cvtColor(img_scaled, cv.COLOR_BGR2HSV)
        img_scaled_pure = img_scaled.copy()

        results = [0]*11

        cv.namedWindow('img')
        cv.namedWindow('edges')
        cv.namedWindow('hsv')

        hsv_colors = []

        # red colour
        hsv_red1 = cv.inRange(hsv, (0, 100, 20), (10, 255, 255))
        hsv_red2 = cv.inRange(hsv, (160, 100, 20), (179, 255, 255))
        hsv_red = cv.bitwise_or(hsv_red1, hsv_red2)

        hsv_colors.append((hsv_red, 5))

        # blue colour
        hsv_blue = cv.inRange(hsv, (100, 100, 20), (135, 255, 255))
        hsv_colors.append((hsv_blue, 7))

        # green colour
        hsv_green = cv.inRange(hsv, (35, 80, 20), (85, 255, 255))
        hsv_colors.append((hsv_green, 6))

        # hsv yellow
        hsv_yellow = cv.inRange(hsv, (20, 100, 20), (35, 255, 255))
        hsv_colors.append((hsv_yellow, 9))

        # hsv white
        hsv_white = cv.inRange(hsv, (20, 0, 168), (200, 91, 204))
        hsv_colors.append((hsv_white, 8))

        hsv_all = cv.bitwise_or(hsv_blue, hsv_red)
        hsv_all = cv.bitwise_or(hsv_all, hsv_green)
        hsv_all = cv.bitwise_or(hsv_all, hsv_yellow)
        hsv_all = cv.bitwise_or(hsv_all, hsv_white)

        hsv_no_whites = cv.bitwise_or(hsv_blue, hsv_red)
        hsv_no_whites = cv.bitwise_or(hsv_no_whites, hsv_green)
        hsv_no_whites = cv.bitwise_or(hsv_no_whites, hsv_yellow)


        # shadow removal
        # https://stackoverflow.com/questions/44752240/how-to-remove-shadow-from-scanned-images-using-opencv
        rgb_planes = cv.split(img_scaled)

        result_planes = []
        result_norm_planes = []
        for plane in rgb_planes:
            dilated_img = cv.dilate(plane, np.ones((5, 5), np.uint8))
            bg_img = cv.medianBlur(dilated_img, 5)
            diff_img = 255 - cv.absdiff(plane, bg_img)
            norm_img = cv.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1)
            result_planes.append(diff_img)
            result_norm_planes.append(norm_img)

        result = cv.merge(result_planes)
        result_norm = cv.merge(result_norm_planes)

        gray = cv.cvtColor(result, cv.COLOR_BGR2GRAY)
        gray_norm = cv.cvtColor(result_norm, cv.COLOR_BGR2GRAY)


        # threshold = 8 #30
        threshold = 255
        erosions = 2
        dilations = 0
        kernel = np.ones((3, 3), np.uint8)
        kernel_prewitt_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)
        kernel_prewitt_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
        kernel_sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32) / 4.0
        kernel_sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32) / 4.0

        # C = 19
        C = 23
        block_size = 26 #72
        # a = 100
        a = 70
        # b = 160
        b = 210
        # b = 37

        hue_low = 68
        saturation_low = 11
        value_low = 177
        hue_high = 101
        saturation_high = 30
        value_high = 186

        cv.createTrackbar('hue_low', 'hsv', hue_low, 255, empty_callback)
        cv.createTrackbar('saturation_low', 'hsv', saturation_low, 255, empty_callback)
        cv.createTrackbar('value_low', 'hsv', value_low, 255, empty_callback)
        cv.createTrackbar('hue_high', 'hsv', hue_high, 255, empty_callback)
        cv.createTrackbar('saturation_high', 'hsv', saturation_high, 255, empty_callback)
        cv.createTrackbar('value_high', 'hsv', value_high, 255, empty_callback)


        cv.createTrackbar('threshold', 'img', threshold, 255, empty_callback)
        cv.createTrackbar('erosion', 'img', erosions, 10, empty_callback)
        cv.createTrackbar('dilation', 'img', dilations, 10, empty_callback)
        cv.createTrackbar('C', 'img', C, 30, empty_callback)
        cv.createTrackbar('block_size', 'img', block_size, 255, empty_callback)
        cv.createTrackbar('a', 'edges', a, 255, empty_callback)
        cv.createTrackbar('b', 'edges', b, 255, empty_callback)

        while True:
            threshold = cv.getTrackbarPos('threshold', 'img')
            erosions = cv.getTrackbarPos('erosion', 'img')
            dilations = cv.getTrackbarPos('dilation', 'img')
            C = cv.getTrackbarPos('C', 'img')
            block_size = cv.getTrackbarPos('block_size', 'img')
            a = cv.getTrackbarPos('a', 'edges')
            b = cv.getTrackbarPos('b', 'edges')

            hue_low = cv.getTrackbarPos('hue_low', 'hsv')
            saturation_low = cv.getTrackbarPos('saturation_low', 'hsv')
            value_low = cv.getTrackbarPos('value_low', 'hsv')
            hue_high = cv.getTrackbarPos('hue_high', 'hsv')
            saturation_high = cv.getTrackbarPos('saturation_high', 'hsv')
            value_high = cv.getTrackbarPos('value_high', 'hsv')

            mask1 = cv.inRange(hsv, (hue_low, saturation_low, value_low), (hue_high, saturation_high, value_high))

            edges = cv.Canny(result_norm, a, b)
            edges = cv.dilate(edges, kernel, iterations=1)
            edges_2 = cv.Canny(result_norm[:,:,2], a, b)
            edges_3 = cv.Canny(result[:,:,2], a, b)
            # edges = cv.Canny(hsv[:,:,2], a, b)
            filtered_y = cv.filter2D(img_scaled, ddepth=-1, kernel=kernel_sobel_y)
            filtered_x = cv.filter2D(img_scaled, ddepth=-1, kernel=kernel_sobel_x)

            filtered_xy = cv.bitwise_or(filtered_x, filtered_y)
            # filtered_xy = filtered_xy * 4

            # filtered_xy = cv.dilate(filtered_xy, kernel, iterations=dilations)

            # filtered_xy = cv.cvtColor(filtered_xy, cv.COLOR_BGR2GRAY)
            filtered_xy = cv.Canny(filtered_xy, a, b)


            # ret, filtered_xy = cv.threshold(filtered_xy, threshold, 255, cv.THRESH_BINARY)
            filtered_xy = cv.medianBlur(filtered_xy, 3)
            filtered_xy = cv.dilate(filtered_xy, kernel, iterations=dilations)
            # cv.imshow('filteredxy sobel', filtered_xy)
            # filtered_y = cv.Canny(filtered_y, 37, 87)

            # ret, thresh = cv.threshold(gray_norm, threshold, 255, cv.THRESH_BINARY)
            adaptive_threshold = cv.adaptiveThreshold(result_norm[:,:,0], 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,
                                                 round_up_to_odd(block_size), C)

            # adaptive_threshold = cv.dilate(adaptive_threshold, kernel, iterations=1)
            adaptive_threshold = cv.erode(adaptive_threshold, kernel, iterations=3)
            adaptive_threshold = cv.bitwise_not(adaptive_threshold)

            # cv.imshow('adaptive', adaptive_threshold)
            # adaptive_threshold = cv.medianBlur(adaptive_threshold, 21)

            # edges = cv.bitwise_or(edges, filtered_xy)
            edges = cv.bitwise_or(edges, adaptive_threshold)


            edges = cv.erode(edges, kernel, iterations=erosions)
            # edges = cv.dilate(edges, kernel, iterations=dilations)
            # edges_2 = cv.erode(edges_2, kernel, iterations=erosions)
            # edges_2 = cv.dilate(edges_2, kernel, iterations=dilations)
            # edges_3 = cv.erode(edges_3, kernel, iterations=erosions)
            # edges_3 = cv.dilate(edges_3, kernel, iterations=dilations)

            # edges = cv.GaussianBlur(edges, (5, 5), 0)

            # edges = cv.bitwise_or(edges, filtered_y)

            # erosion = cv.medianBlur(erosion, 5)
            edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, np.ones((7, 7), np.uint8))
            # edges_2 = cv.morphologyEx(edges_2, cv.MORPH_CLOSE, np.ones((21, 21), np.uint8))
            # edges_3 = cv.morphologyEx(edges_3, cv.MORPH_CLOSE, np.ones((21, 21), np.uint8))
            edges = cv.bitwise_or(edges, hsv_no_whites)



            # edges = cv.bitwise_or(edges, edges_2)
            # edges = cv.bitwise_or(edges, edges_3)



            edges_fill = fillhole(edges)


            cv.imshow('img', result)
            cv.imshow('edges', edges)
            # cv.imshow('edges_inv', edges_inv)
            cv.imshow('edges_fill', edges_fill)
            # cv.imshow('hsv', mask1)
            # cv.imshow('hsv_to_watch', mask1)
            # cv.imshow('sobel', filtered_xy)


            key_code = cv.waitKey(10)
            if key_code == 27:
                break

        # inv = cv.bitwise_not(erosion)

        contours, hierarchy = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)


        bounding_boxes = []
        filtered_contours = []
        biggest_contour_area = 0

        for contour in contours:
            contour_area = cv.contourArea(contour)
            if contour_area > biggest_contour_area:
                biggest_contour_area = contour_area

        contours_mask = np.zeros(img_scaled.shape[:2],dtype=np.uint8)
        hull_list = []
        for contour in contours:
            if 20000 > cv.contourArea(contour) >= biggest_contour_area/1.6:
                # cv.drawContours(img_scaled, contour, -1, (0, 255, 0), 3)
                filtered_contours.append(contour)
                x, y, w, h = cv.boundingRect(contour)
                bounding_boxes.append((x, y, w, h))
                cv.rectangle(img_scaled, (x, y), (x+w, y+h), (0, 0, 255), 2)
                hull = cv.convexHull(contour)
                hull_list.append(hull)

                # cv.fillConvexPoly(img_scaled, contour, (255, 255, 255))
        # cv.polylines(img_scaled, filtered_contours, False, (255, 255, 255), 3)
        cv.drawContours(img_scaled, filtered_contours, -1, (0, 255, 0), 3)
        cv.drawContours(img_scaled, hull_list, -1, (255, 0, 0), 3)
        cv.fillPoly(contours_mask, filtered_contours, 255)

        cv.imshow('contours', img_scaled)



        # 8x2 block
        mask_1_color = cv.imread('klocki/1_mask.jpg')
        mask_1_color_scaled = cv.resize(mask_1_color, fx=0.25, fy=0.25, dsize=None)
        print('8x2 blocks: \n')
        results[0], data_1 = match_contours(filtered_contours, mask_1_color_scaled)

        # tetris block
        mask_7_color = cv.imread('klocki/7_mask.jpg')
        mask_7_color_scaled = cv.resize(mask_7_color, fx=0.25, fy=0.25, dsize=None)
        print('tetris blocks: \n')
        results[1], data_2 = match_contours(filtered_contours, mask_7_color_scaled)

        # l block
        mask_2_color = cv.imread('klocki/2_mask.jpg')
        mask_3_color = cv.imread('klocki/3_mask.jpg')
        mask_2_color_scaled = cv.resize(mask_2_color, fx=0.25, fy=0.25, dsize=None)
        mask_3_color_scaled = cv.resize(mask_3_color, fx=0.25, fy=0.25, dsize=None)
        print('l blocks: \n')
        results[2], data_3 = match_contours(filtered_contours, mask_3_color_scaled)
        results[2], data_3_2 = match_contours(filtered_contours, mask_2_color_scaled)
        for idx, x in enumerate(data_3):
            for idy, y in enumerate(data_3_2):
                if x[0] == y[0]:
                    if data_3[idx][1] > data_3_2[idy][1]:
                        data_3[idx] = data_3_2[idy]



        # square blocks:
        mask_6_color = cv.imread('klocki/6_mask.jpg')
        mask_6_color_scaled = cv.resize(mask_6_color, fx=0.25, fy=0.25, dsize=None)
        mask_6_color_scaled = cv.rotate(mask_6_color_scaled, cv.ROTATE_90_COUNTERCLOCKWISE)
        print('square blocks: \n')
        results[3], data_4 = match_contours(filtered_contours, mask_6_color_scaled)

        # z and s blocks:
        mask_4_color = cv.imread('klocki/4_mask.jpg')
        mask_5_color = cv.imread('klocki/5_mask.jpg')
        mask_4_color_scaled = cv.resize(mask_4_color, fx=0.25, fy=0.25, dsize=None)
        mask_5_color_scaled = cv.resize(mask_5_color, fx=0.25, fy=0.25, dsize=None)
        print('z and s blocks: \n')

        results[4], data_5 = match_contours(filtered_contours, mask_5_color_scaled)
        results[4], data_5_2 = match_contours(filtered_contours, mask_4_color_scaled)
        for idx, x in enumerate(data_5):
            for idy, y in enumerate(data_5_2):
                if x[0] == y[0]:
                    if data_5[idx][1] > data_5_2[idy][1]:
                        data_5[idx] = data_5_2[idy]

        check_convex(data_2, data_4, filtered_contours, hull_list)
        check_convex(data_3, data_4, filtered_contours, hull_list)
        choose_lower(data_5, data_3)
        choose_lower(data_5, data_1)
        choose_lower(data_5, data_4)
        choose_lower(data_1, data_3)
        choose_lower(data_1, data_4)
        choose_lower(data_5, data_2)
        choose_lower(data_4, data_2)
        choose_lower(data_4, data_1)
        choose_lower(data_4, data_3)
        choose_lower(data_3, data_1)
        choose_lower(data_3, data_2)
        choose_lower(data_2, data_1)




        results[0] = len(data_1)
        results[1] = len(data_2)
        results[2] = len(data_3)
        results[3] = len(data_4)
        results[4] = len(data_5)


        for bb in bounding_boxes:

            tmp =  contours_mask[bb[1]:bb[1]+bb[3], bb[0]:bb[0]+bb[2]]

            hsvs = []
            # cv.imshow('tmp', tmp)
            for hsv in hsv_colors:
                temp_color = cv.bitwise_and(tmp, tmp, mask=cv.erode(hsv[0][bb[1]:bb[1]+bb[3], bb[0]:bb[0]+bb[2]],
                                                                    np.ones((3, 3)), iterations=3))
                # cv.imshow('temp_color', temp_color)
                # cv.imshow('hsv_colors', cv.erode(hsv[0][bb[1]:bb[1]+bb[3], bb[0]:bb[0]+bb[2]], np.ones((3, 3)),
                #                                  iterations=3))
                # cv.waitKey(0)
                if np.sum(temp_color) != 0:
                    hsvs.append(hsv[1])
            if len(hsvs) != 0:
                if len(hsvs) == 1:
                    results[hsvs[0]] += 1
                else:
                    results[10] += 1

            # cv.waitKey(0)

        # hsv_all = cv.bitwise_and(img_scaled, img_scaled, mask=hsv_colors[0][0])

        # cv.imshow('hsv_x', hsv_all)



        print('results: ', results)

        results_all[filename] = results

        cv.waitKey(0)
        cv.destroyAllWindows()

    with open('./gt_result.json', 'r') as f:
        gt = json.load(f)
    print("GT: ", gt)
    print("---------------")
    print("RES: ", results_all)
    score = calculateScore(gt, results_all, len(results_all))
    print("Score for all images: ", score)

    with open('./result.txt', 'a') as file:
        file.write("-----------------------------------------------------" + '\n')
        file.write("GT: " + str(gt) + '\n')
        file.write("RE: " + str(results_all) + '\n')
        file.write("Score: " + str(score) + '\n')
        file.write("-----------------------------------------------------" + '\n')

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
