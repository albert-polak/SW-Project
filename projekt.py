import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def empty_callback(x):
    pass


def round_up_to_odd(f):
    f = int(np.ceil(f))
    return f + 1 if f % 2 == 0 else f


def main():
    img = cv.imread('projekt/img_003.jpg')
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
        bg_img = cv.medianBlur(dilated_img, 3)
        diff_img = 255 - cv.absdiff(plane, bg_img)
        norm_img = cv.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)

    result = cv.merge(result_planes)
    result_norm = cv.merge(result_norm_planes)

    gray = cv.cvtColor(result, cv.COLOR_BGR2GRAY)
    gray_norm = cv.cvtColor(result_norm, cv.COLOR_BGR2GRAY)



    threshold = 242
    erosions = 0
    dilations = 0
    kernel = np.ones((3, 3), np.uint8)
    C = 0
    block_size = 11
    a = 138
    b = 130

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

        ret, thresh = cv.threshold(gray_norm, threshold, 255, cv.THRESH_BINARY)
        adaptive_threshold = cv.adaptiveThreshold(gray_norm, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,
                                             round_up_to_odd(block_size), C)

        edges = cv.erode(edges, kernel, iterations=erosions)
        edges = cv.dilate(edges, kernel, iterations=dilations)
        edges = cv.GaussianBlur(edges, (5, 5), 0)


        # erosion = cv.medianBlur(erosion, 5)
        edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, np.ones((9, 9), np.uint8))



        cv.imshow('img', result)
        cv.imshow('edges', edges)

        key_code = cv.waitKey(10)
        if key_code == 27:
            # escape key pressed
            break

    # inv = cv.bitwise_not(erosion)

    contours, hierarchy = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # cv.drawContours(img_scaled, contours, -1, (0, 255, 0), 3)
    filtered_contours = []
    for contour in contours:
        if 10000 > cv.contourArea(contour) >= 500:
            # cv.drawContours(img_scaled, contour, -1, (0, 255, 0), 3)
            filtered_contours.append(contour)

    cv.drawContours(img_scaled, filtered_contours, -1, (0, 255, 0), 3)
    cv.imshow('contours', img_scaled)

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
