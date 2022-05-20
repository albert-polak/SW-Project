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


def sharpen_image(input_image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    image_sharp = cv.filter2D(src=input_image, ddepth=-1, kernel=kernel)
    return image_sharp

def main():
    img = cv.imread('projekt/img_001.jpg')
    img_scaled = cv.resize(img, fx=0.3, fy=0.3, dsize=None)
    gray_scaled = cv.cvtColor(img_scaled, cv.COLOR_BGR2GRAY)
    hsv = cv.cvtColor(img_scaled, cv.COLOR_BGR2HSV)

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
    dilations = 2
    kernel = np.ones((3, 3), np.uint8)
    kernel_prewitt_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32) / 3.0
    kernel_prewitt_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32) / 3.0
    kernel_sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32) / 4.0
    kernel_sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32) / 4.0

    C = 19
    block_size = 72
    a = 138
    # a = 36
    # b = 130
    b = 37

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
            # escape key pressed
            break

    # inv = cv.bitwise_not(erosion)

    contours, hierarchy = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    cv.fillPoly(img_scaled, contours, (255, 255, 255))
    # cv.drawContours(img_scaled, contours, -1, (0, 255, 0), 3)
    filtered_contours = []
    for contour in contours:
        if 20000 > cv.contourArea(contour) >= 500:
            # cv.drawContours(img_scaled, contour, -1, (0, 255, 0), 3)
            filtered_contours.append(contour)

    cv.drawContours(img_scaled, filtered_contours, -1, (0, 255, 0), 3)
    cv.imshow('contours', img_scaled)

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
