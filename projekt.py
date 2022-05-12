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
    gray = cv.cvtColor(img_scaled, cv.COLOR_BGR2GRAY)
    hsv = cv.cvtColor(img_scaled, cv.COLOR_BGR2HSV)

    cv.namedWindow('img')
    cv.namedWindow('whites')

    threshold = 87
    threshold_whites = 125
    threshold_blues = 71

    C = -3
    d = 9
    sigmaColor = 145
    sigmaSpace = 75
    block_size = 11
    erosions = 0
    dilations = 0

    # in range variables
    hue_low = 68
    saturation_low = 11
    value_low = 177
    hue_high = 101
    saturation_high = 30
    value_high = 186



    kernel = np.ones((3, 3), np.uint8)

    # cv.createTrackbar('sigmaColor', 'img', sigmaColor, 255, empty_callback)
    # cv.createTrackbar('sigmaSpace', 'img', sigmaSpace, 255, empty_callback)
    cv.createTrackbar('erosion', 'img', erosions, 10, empty_callback)
    cv.createTrackbar('dilation', 'img', dilations, 10, empty_callback)
    cv.createTrackbar('threshold', 'img', threshold, 255, empty_callback)
    cv.createTrackbar('threshold_whites', 'img', threshold_whites, 255, empty_callback)
    cv.createTrackbar('threshold_blues', 'img', threshold_blues, 255, empty_callback)
    cv.createTrackbar('C', 'img', C, 30, empty_callback)
    cv.createTrackbar('block_size', 'img', block_size, 255, empty_callback)
    cv.createTrackbar('hue_low', 'whites', hue_low, 255, empty_callback)
    cv.createTrackbar('saturation_low', 'whites', saturation_low, 255, empty_callback)
    cv.createTrackbar('value_low', 'whites', value_low, 255, empty_callback)
    cv.createTrackbar('hue_high', 'whites', hue_high, 255, empty_callback)
    cv.createTrackbar('saturation_high', 'whites', saturation_high, 255, empty_callback)
    cv.createTrackbar('value_high', 'whites', value_high, 255, empty_callback)

    while True:

        erosions = cv.getTrackbarPos('erosion', 'img')
        dilations = cv.getTrackbarPos('dilation', 'img')
        threshold = cv.getTrackbarPos('threshold', 'img')
        threshold_whites = cv.getTrackbarPos('threshold_whites', 'img')
        threshold_blues = cv.getTrackbarPos('threshold_blues', 'img')
        C = cv.getTrackbarPos('C', 'img')
        block_size = cv.getTrackbarPos('block_size', 'img')

        hue_low = cv.getTrackbarPos('hue_low', 'whites')
        saturation_low = cv.getTrackbarPos('saturation_low', 'whites')
        value_low = cv.getTrackbarPos('value_low', 'whites')
        hue_high = cv.getTrackbarPos('hue_high', 'whites')
        saturation_high = cv.getTrackbarPos('saturation_high', 'whites')
        value_high = cv.getTrackbarPos('value_high', 'whites')

        # pyr = cv.pyrMeanShiftFiltering(img_scaled, 1, 30)

        mask1 = cv.inRange(hsv, (hue_low, saturation_low, value_low), (hue_high, saturation_high, value_high))
        ret, thresh_color = cv.threshold(img_scaled[:, :, 0], threshold, 255, cv.THRESH_BINARY)
        # ret2, thresh_whites = cv.threshold(img_scaled[:, :, 0], threshold_whites, 255, cv.THRESH_BINARY_INV)
        ret3, thresh_blues = cv.threshold(img_scaled[:, :, 2], threshold_blues, 255, cv.THRESH_BINARY)

        thresh_all = cv.bitwise_and(thresh_color, thresh_blues)
        thresh_all = cv.bitwise_and(thresh_all, cv.bitwise_not(mask1))
        # thresh_whites = cv.adaptiveThreshold(img_scaled[:, :, 0], 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,
        #                                      round_up_to_odd(block_size), C)

        dilation = cv.dilate(thresh_all, kernel, iterations=dilations)
        erosion = cv.erode(dilation, kernel, iterations=erosions)

        blur = cv.medianBlur(erosion, 3)

        # white threshold morphology
        # dilation_whites = cv.dilate(thresh_whites, kernel, iterations=dilations)
        # erosion_whites = cv.erode(dilation_whites, kernel, iterations=erosions)

        # blur_whites = cv.medianBlur(erosion_whites, 3)


        # blur = cv.bilateralFilter(blur, d, sigmaColor, sigmaSpace)
        # blur = cv.GaussianBlur(blur, (5, 5), 0)

        key_code = cv.waitKey(10)
        if key_code == 27:
            # escape key pressed
            break

        cv.imshow('img', blur)
        cv.imshow('thresh_color', thresh_color)
        # cv.imshow('thresh_whites', thresh_whites)
        cv.imshow('thresh_blues', thresh_blues)
        cv.imshow('whites', mask1)
        cv.imshow('whites_to_watch', mask1)


    inv = cv.bitwise_not(blur)

    contours, hierarchy = cv.findContours(inv, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(img_scaled, contours, -1, (0, 255, 0), 3)

    cv.imshow('contours', img_scaled)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
