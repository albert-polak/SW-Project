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


def main():
    img = cv.imread('projekt/img_001.jpg')
    img_scaled = cv.resize(img, fx=0.3, fy=0.3, dsize=None)
    gray_scaled = cv.cvtColor(img_scaled, cv.COLOR_BGR2GRAY)
    hsv = cv.cvtColor(img_scaled, cv.COLOR_BGR2HSV)

    results = [0]*11

    cv.namedWindow('img')
    cv.namedWindow('edges')

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


    threshold = 8 #30
    erosions = 0
    dilations = 1
    kernel = np.ones((3, 3), np.uint8)
    kernel_prewitt_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32) / 3.0
    kernel_prewitt_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32) / 3.0
    kernel_sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32) / 4.0
    kernel_sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32) / 4.0

    C = 19
    block_size = 72
    # a = 138
    a = 255
    b = 255
    # b = 37

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


        edges = cv.Canny(result_norm[:,:,0], a, b)
        edges_2 = cv.Canny(result_norm[:,:,2], a, b)
        edges_3 = cv.Canny(result[:,:,2], a, b)
        # edges = cv.Canny(hsv[:,:,2], a, b)
        filtered_y = cv.filter2D(img_scaled[:,:,0], ddepth=-1, kernel=kernel_sobel_y)
        filtered_x = cv.filter2D(img_scaled[:,:,0], ddepth=-1, kernel=kernel_sobel_x)

        filtered_xy = cv.bitwise_or(filtered_x, filtered_y)

        ret, thresh_filtered = cv.threshold(filtered_xy, threshold, 255, cv.THRESH_BINARY)
        filtered_xy = cv.dilate(thresh_filtered, kernel, iterations=dilations)
        # filtered_y = cv.Canny(filtered_y, 37, 87)

        # ret, thresh = cv.threshold(gray_norm, threshold, 255, cv.THRESH_BINARY)
        adaptive_threshold = cv.adaptiveThreshold(result_norm[:,:,0], 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,
                                             round_up_to_odd(block_size), C)

        # adaptive_threshold = cv.dilate(adaptive_threshold, kernel, iterations=1)
        adaptive_threshold = cv.erode(adaptive_threshold, kernel, iterations=3)
        adaptive_threshold = cv.bitwise_not(adaptive_threshold)

        # cv.imshow('adaptive', adaptive_threshold)
        # adaptive_threshold = cv.medianBlur(adaptive_threshold, 21)

        edges = cv.bitwise_or(edges, filtered_xy)
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



        # edges = cv.bitwise_or(edges, edges_2)
        # edges = cv.bitwise_or(edges, edges_3)



        edges_fill = fillhole(edges)


        cv.imshow('img', result)
        cv.imshow('edges', edges)
        # cv.imshow('edges_inv', edges_inv)
        cv.imshow('edges_fill', edges_fill)
        # cv.imshow('hsv', hsv[:,:,2])
        # cv.imshow('sobel', filtered_xy)


        key_code = cv.waitKey(10)
        if key_code == 27:
            break

    # inv = cv.bitwise_not(erosion)

    contours, hierarchy = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)



    filtered_contours = []
    for contour in contours:
        if 20000 > cv.contourArea(contour) >= 5000:
            # cv.drawContours(img_scaled, contour, -1, (0, 255, 0), 3)
            filtered_contours.append(contour)

    cv.drawContours(img_scaled, filtered_contours, -1, (0, 255, 0), 3)
    cv.imshow('contours', img_scaled)


    # 8x2 block
    mask_1_color = cv.imread('klocki/1_mask.jpg')
    print('8x2 blocks: \n')
    results[0], data_1 = match_contours(filtered_contours, mask_1_color)

    # tetris block
    mask_7_color = cv.imread('klocki/7_mask.jpg')
    print('tetris blocks: \n')
    results[1], data_2 = match_contours(filtered_contours, mask_7_color)

    # l block
    # mask_2_color = cv.imread('klocki/2_mask.jpg')
    mask_3_color = cv.imread('klocki/3_mask.jpg')
    print('l blocks: \n')
    # match_contours(filtered_contours, mask_2_color)
    results[2], data_3 = match_contours(filtered_contours, mask_3_color)

    # z and s blocks:
    # mask_4_color = cv.imread('klocki/4_mask.jpg')
    mask_5_color = cv.imread('klocki/5_mask.jpg')
    print('z and s blocks: \n')
    # match_contours(filtered_contours, mask_4_color)
    results[4], data_5 = match_contours(filtered_contours, mask_5_color)

    # square blocks:
    # mask_6_color = cv.imread('klocki/6_mask.jpg')
    # print('square blocks: \n')
    # results[3] = match_contours(filtered_contours, mask_6_color)

    choose_lower(data_5, data_3)
    choose_lower(data_5, data_1)


    results[4] = len(data_5)
    results[2] = len(data_3)
    results[0] = len(data_1)



    print('results: ', results)

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
