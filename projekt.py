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



    while True:


        key_code = cv.waitKey(10)
        if key_code == 27:
            # escape key pressed
            break



    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
