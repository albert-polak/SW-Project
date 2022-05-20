import cv2 as cv
import numpy as np

def emptycallback(x):
    pass


def make_mask():
    im_number = 1

    img = cv.imread(f'klocki/{im_number}.jpg')

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    cv.namedWindow('img')

    threshold = 88

    kernel = np.ones((3, 3), np.uint8)

    cv.createTrackbar('threshold', 'img', threshold, 255, emptycallback)

    while True:
        threshold = cv.getTrackbarPos('threshold', 'img')

        ret, thresh = cv.threshold(gray, threshold, 255, cv.THRESH_BINARY_INV)
        thresh = cv.medianBlur(thresh, 5)
        # thresh = cv.dilate(thresh, kernel, iterations=1)
        thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, np.ones((7, 7), np.uint8))

        key_code = cv.waitKey(10)
        if key_code == 27:
            # escape key pressed
            break

        cv.imshow('img', thresh)

    cv.imwrite(f'klocki/{im_number}_mask.jpg', thresh)

    cv.waitKey(0)
    cv.destroyAllWindows()



make_mask()