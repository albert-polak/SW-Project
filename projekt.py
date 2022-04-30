import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def empty_callback(x):
    pass


def round_up_to_odd(f):
    f = int(np.ceil(f))
    return f + 1 if f % 2 == 0 else f


def main():
    img = cv.imread('projekt/img_001.jpg')
    img_scaled = cv.resize(img, fx=0.3, fy=0.3, dsize=None)
    gray = cv.cvtColor(img_scaled, cv.COLOR_BGR2GRAY)
    
    cv.namedWindow('img')

    threshold = 87
    # C = 0
    C = -3
    block_size = 11
    # block_size = 9

    cv.createTrackbar('threshold', 'img', threshold, 255, empty_callback)
    cv.createTrackbar('C', 'img', C, 30, empty_callback)
    cv.createTrackbar('block_size', 'img', 3, 255, empty_callback)


    # ret, thresh_colours = cv.threshold(gray, 87, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C)
    thresh_whites1 = cv.adaptiveThreshold(img_scaled[:,:,0], 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,
                                         round_up_to_odd(block_size), C)

    # kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    # morph = cv.morphologyEx(thresh_whites, cv.MORPH_CLOSE, kernel)



    while True:

        threshold = cv.getTrackbarPos('threshold', 'img')
        C = cv.getTrackbarPos('C', 'img')

        block_size = cv.getTrackbarPos('block_size', 'img')


        thresh_whites = cv.adaptiveThreshold(img_scaled[:,:,0], 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,
                                             round_up_to_odd(block_size), C)
        # ret, thresh_whites = cv.threshold(gray, threshold, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C)

        # blur = cv.GaussianBlur(thresh_whites, (3, 3), 0)
        # blur = cv.medianBlur(thresh_whites, 3)
        blur1 = cv.bilateralFilter(thresh_whites1, 9, 75, 75)
        blur = cv.bilateralFilter(thresh_whites, 9, 75, 75)

        key_code = cv.waitKey(10)
        if key_code == 27:
            # escape key pressed
            break

        # dst = cv.addWeighted(thresh_colours, 1, thresh_whites, 1, 0)

        cv.imshow('img', blur)
        # cv.imshow('org', dst)
        cv.imshow('gray', blur1)
        cv.imshow('colour', img_scaled[:,:,0])

    cv.destroyAllWindows()



if __name__ == "__main__":
    main()