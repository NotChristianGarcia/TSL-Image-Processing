"""
Simple plot stuff
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt


def intensityHistogram():
    gray = cv2.imread("testIMG.jpg", 0)
    gray1D = np.ravel(gray)
    plt.hist(gray1D, 50)
    plt.show()


def dualIntensityHistograms():
    test1 = cv2.imread("testIMG.jpg", 0)
    test11D = np.ravel(test1)
    test2 = cv2.imread("testHistIMG.jpg", 0)
    test21D = np.ravel(test2)

    plt.hist(test21D, 50, color="blue")
    plt.hist(test11D, 50, color="red")
    plt.show()


if __name__ == "__main__":
    intensityHistogram()
    dualIntensityHistograms()
