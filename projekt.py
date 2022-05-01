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
    C = -3
    block_size = 11
    d = 9
    sigmaColor = 145
    sigmaSpace = 75

    # cv.createTrackbar('C', 'img', C, 30, empty_callback)
    # cv.createTrackbar('block_size', 'img', block_size, 255, empty_callback)
    cv.createTrackbar('d', 'img', d, 30, empty_callback)
    cv.createTrackbar('sigmaColor', 'img', sigmaColor, 255, empty_callback)
    cv.createTrackbar('sigmaSpace', 'img', sigmaSpace, 255, empty_callback)

    thresh_whites = cv.adaptiveThreshold(img_scaled[:, :, 0], 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,
                                         round_up_to_odd(block_size), C)

    while True:
        d = cv.getTrackbarPos('d', 'img')
        # C = cv.getTrackbarPos('C', 'img')
        # block_size = cv.getTrackbarPos('block_size', 'img')
        sigmaColor = cv.getTrackbarPos('sigmaColor', 'img')
        sigmaSpace = cv.getTrackbarPos('sigmaSpace', 'img')

        thresh_whites = cv.adaptiveThreshold(img_scaled[:, :, 0], 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,
                                             round_up_to_odd(block_size), C)

        blur = cv.bilateralFilter(thresh_whites, d, sigmaColor, sigmaSpace)


        key_code = cv.waitKey(10)
        if key_code == 27:
            # escape key pressed
            break

        cv.imshow('img', blur)
        # cv.imshow('org', dst)
        cv.imshow('colour', img_scaled[:, :, 0])

    cv.destroyAllWindows()



if __name__ == "__main__":
    main()