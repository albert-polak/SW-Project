import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


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

    cv.imshow('mask_7', mask_color)
    cv.waitKey(0)
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


def check_convex(data_1, data_2, filtered_contours):
    for idx, x in enumerate(data_1):
        for idy, y in enumerate(data_2):
            if x is not None and y is not None:
                if x[0] == y[0]:
                    if cv.isContourConvex(filtered_contours[x[0]]):
                        data_2[idy] = None
                    else:
                        data_1[idx] = None
    data_1[:] = list(filter(None, data_1))
    data_2[:] = list(filter(None, data_2))



def main():
    img = cv.imread('projekt/img_004.jpg')
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
    erosions = 0
    dilations = 0
    kernel = np.ones((3, 3), np.uint8)
    kernel_prewitt_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32) / 3.0
    kernel_prewitt_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32) / 3.0
    kernel_sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    kernel_sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)

    # C = 19
    C = 30
    block_size = 26 #72
    # a = 138
    a = 100
    b = 148
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

    edges = hsv_all

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

        canny = cv.Canny(result_norm[:,:,0], a, b)

        filtered_y = cv.filter2D(result_norm, ddepth=-1, kernel=kernel_sobel_y)
        filtered_x = cv.filter2D(result_norm, ddepth=-1, kernel=kernel_sobel_x)
        filtered_xy = cv.bitwise_or(filtered_x, filtered_y)
        filtered_xy = cv.cvtColor(filtered_xy, cv.COLOR_BGR2GRAY)
        filtered_xy = cv.erode(filtered_xy, kernel, iterations=erosions)
        filtered_xy = cv.dilate(filtered_xy, kernel, iterations=1)
        ret, filtered_xy = cv.threshold(filtered_xy, 35, 255, cv.THRESH_BINARY)
        filtered_xy = cv.medianBlur(filtered_xy, 3)

        # edges = cv.bitwise_or(filtered_xy, hsv_all)
        edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, np.ones((3, 3), np.uint8))

        cv.imshow('hsv_all', edges)


        key_code = cv.waitKey(10)
        if key_code == 27:
            break

    # inv = cv.bitwise_not(erosion)

    contours, hierarchy = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)


    bounding_boxes = []
    filtered_contours = []
    for contour in contours:
        if 10000 > cv.contourArea(contour) >= 3500:
            # cv.drawContours(img_scaled, contour, -1, (0, 255, 0), 3)
            filtered_contours.append(contour)
            x, y, w, h = cv.boundingRect(contour)
            bounding_boxes.append((x, y, w, h))
            cv.rectangle(img_scaled, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv.drawContours(img_scaled, filtered_contours, -1, (0, 255, 0), 3)
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
    # results[2], data_3_2 = match_contours(filtered_contours, mask_2_color_scaled)
    # for idx, x in enumerate(data_3):
    #     for idy, y in enumerate(data_3_2):
    #         if x[0] == y[0]:
    #             if data_3[idx][1] > data_3_2[idy][1]:
    #                 data_3[idx] = data_3_2[idy]



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
    # results[4], data_5_2 = match_contours(filtered_contours, mask_4_color_scaled)
    # for idx, x in enumerate(data_5):
    #     for idy, y in enumerate(data_5_2):
    #         if x[0] == y[0]:
    #             if data_5[idx][1] > data_5_2[idy][1]:
    #                 data_5[idx] = data_5_2[idy]


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
    check_convex(data_2, data_4, filtered_contours)
    check_convex(data_3, data_4, filtered_contours)


    results[0] = len(data_1)
    results[1] = len(data_2)
    results[2] = len(data_3)
    results[3] = len(data_4)
    results[4] = len(data_5)


    for bb in bounding_boxes:

        tmp = img_scaled_pure[bb[1]:bb[1]+bb[3], bb[0]:bb[0]+bb[2], :]

        hsvs = []
        cv.imshow('tmp', tmp)
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

        cv.waitKey(0)

    # hsv_all = cv.bitwise_and(img_scaled, img_scaled, mask=hsv_colors[0][0])

    cv.imshow('hsv_x', hsv_all)



    print('results: ', results)

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
