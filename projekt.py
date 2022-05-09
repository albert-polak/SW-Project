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
    
    cv.namedWindow('img')

    threshold = 87

    d = 9
    sigmaColor = 145
    sigmaSpace = 75
    erosions = 0
    dilations = 0

    kernel = np.ones((3, 3), np.uint8)

    # cv.createTrackbar('sigmaColor', 'img', sigmaColor, 255, empty_callback)
    # cv.createTrackbar('sigmaSpace', 'img', sigmaSpace, 255, empty_callback)
    cv.createTrackbar('erosion', 'img', erosions, 10, empty_callback)
    cv.createTrackbar('dilation', 'img', dilations, 10, empty_callback)
    cv.createTrackbar('threshold', 'img', threshold, 255, empty_callback)





    while True:

        erosions = cv.getTrackbarPos('erosion', 'img')
        dilations = cv.getTrackbarPos('dilation', 'img')
        threshold = cv.getTrackbarPos('threshold', 'img')


        ret, thresh_whites = cv.threshold(img_scaled[:,:,0], threshold, 255, cv.THRESH_BINARY)


        dilation = cv.dilate(thresh_whites, kernel, iterations=dilations)
        erosion = cv.erode(dilation, kernel, iterations=erosions)


        blur = cv.medianBlur(erosion, 3)
        # cv.pyrMeanShiftFiltering()
        # blur = cv.bilateralFilter(blur, d, sigmaColor, sigmaSpace)
        # blur = cv.GaussianBlur(blur, (5, 5), 0)

        key_code = cv.waitKey(10)
        if key_code == 27:
            # escape key pressed
            break

        cv.imshow('img', blur)
        # cv.imshow('org', dst)
        cv.imshow('colour', img_scaled[:,:,0])

    inv = cv.bitwise_not(blur)

    contours, hierarchy = cv.findContours(inv, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(img_scaled, contours, -1, (0, 255, 0), 3)

    cv.imshow('contours', img_scaled)
    cv.waitKey(0)
    cv.destroyAllWindows()



if __name__ == "__main__":
    main()