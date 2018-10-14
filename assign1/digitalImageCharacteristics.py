import numpy as np
import cv2
import matplotlib.pyplot as plt


def intensityHistogram():
    gray = cv2.imread("testIMG.jpg", 0)
    gray_1d = np.ravel(gray)
    plt.hist(gray_1d, 50)
    plt.show()


def dualIntensityHistograms():
    test1 = cv2.imread("testIMG.jpg", 0)
    test1_1d = np.ravel(test1)
    test2 = cv2.imread("testHistIMG.jpg", 0)
    test2_1d = np.ravel(test2)

    plt.hist(test2_1d, 50, color="blue")
    plt.hist(test1_1d, 50, color="red")
    plt.show()


if __name__ == "__main__":
    intensityHistogram()
    dualIntensityHistograms()
