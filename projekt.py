import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def empty_callback(x):
    pass


def round_up_to_odd(f):
    f = int(np.ceil(f))
    return f + 1 if f % 2 == 0 else f


def main():
    img = cv.imread('projekt/img_015.jpg')
    img_scaled = cv.resize(img, fx=0.3, fy=0.3, dsize=None)

    hsv = cv.cvtColor(img_scaled, cv.COLOR_BGR2HSV)

    cv.namedWindow('img')

    rgb_planes = cv.split(img_scaled)

    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv.medianBlur(dilated_img, 5)
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

    cv.createTrackbar('threshold', 'img', threshold, 255, empty_callback)
    cv.createTrackbar('erosion', 'img', erosions, 10, empty_callback)
    cv.createTrackbar('dilation', 'img', dilations, 10, empty_callback)

    while True:
        threshold = cv.getTrackbarPos('threshold', 'img')
        erosions = cv.getTrackbarPos('erosion', 'img')
        dilations = cv.getTrackbarPos('dilation', 'img')

        ret, thresh = cv.threshold(gray_norm, threshold, 255, cv.THRESH_BINARY)

        dilation = cv.dilate(thresh, kernel, iterations=dilations)
        erosion = cv.erode(dilation, kernel, iterations=erosions)
        erosion = cv.medianBlur(erosion, 5)

        cv.imshow('img', result)
        cv.imshow('result_norm', result_norm)
        cv.imshow('gray', gray)
        cv.imshow('gray_norm', gray_norm)
        cv.imshow('threshold', erosion)

        key_code = cv.waitKey(10)
        if key_code == 27:
            # escape key pressed
            break

    inv = cv.bitwise_not(erosion)

    contours, hierarchy = cv.findContours(inv, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # cv.drawContours(img_scaled, contours, -1, (0, 255, 0), 3)

    for contour in contours:
        if cv.contourArea(contour) > 5000:
            cv.drawContours(img_scaled, contour, -1, (255, 255, 255), 3, lineType=cv.LINE_AA)

    cv.imshow('contours', img_scaled)

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
