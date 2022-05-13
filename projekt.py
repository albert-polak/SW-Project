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

    hsv = cv.cvtColor(img_scaled, cv.COLOR_BGR2HSV)



    rgb_planes = cv.split(img_scaled)

    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv.medianBlur(dilated_img, 21)
        diff_img = 255 - cv.absdiff(plane, bg_img)
        norm_img = cv.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)

    result = cv.merge(result_planes)
    result_norm = cv.merge(result_norm_planes)

    gray = cv.cvtColor(result, cv.COLOR_BGR2GRAY)
    gray_norm = cv.cvtColor(result_norm, cv.COLOR_BGR2GRAY)

    while True:

        cv.imshow('result', result)
        cv.imshow('result_norm', result_norm)
        cv.imshow('gray', gray)
        cv.imshow('gray_norm', gray_norm)
        key_code = cv.waitKey(10)
        if key_code == 27:
            # escape key pressed
            break

    # ret, thresh_color = cv.threshold(img_scaled[:, :, 0], threshold, 255, cv.THRESH_BINARY)

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
