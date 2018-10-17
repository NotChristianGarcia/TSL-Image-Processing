"""
Image morphology consisting of thresholding, erosion, dilation
opening, closing, and some connected component labeling.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt


def intensityHistogramTest(plot=1):
    gray = cv2.imread("Particles.jpg", 0)
    gray1D = np.ravel(gray)
    if plot == 1:
        plt.hist(gray1D, 50)
        plt.show()
    return

def threshold(plot=0):
    gray = cv2.imread("Particles.jpg", 0)
    _, thresholdedImg = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    if plot == 1:
        plt.imshow(thresholdedImg, cmap="gray")
        plt.show()
    return thresholdedImg


def erosion(binaryImg, plot=0):
    kernel = np.ones((3, 3))
    erodedImg = cv2.erode(binaryImg, kernel, iterations=1)
    if plot == 1:
        plt.imshow(erodedImg, cmap="gray")
        plt.show()
    return erodedImg


def dilation(binaryImg, plot=0):
    kernel = np.ones((3, 3))
    dilatedImg = cv2.dilate(binaryImg, kernel, iterations=1)
    if plot == 1:
        plt.imshow(dilatedImg, cmap="gray")
        plt.show()
    return dilatedImg


def opening(binaryImg, plot=0):
    kernel = np.ones((3, 3))
    openedImg = cv2.morphologyEx(binaryImg, cv2.MORPH_OPEN, kernel)
    if plot == 1:
        plt.imshow(openedImg, cmap="gray")
        plt.show()
    return openedImg


def closing(binaryImg, plot=0):
    kernel = np.ones((3, 3))
    closedImg = cv2.morphologyEx(binaryImg, cv2.MORPH_CLOSE, kernel)
    if plot == 1:
        plt.imshow(closedImg, cmap="gray")
        plt.show()
    return closedImg


def connectedComponentLabeling(processedImg, plot=0):
    _, cclImg = cv2.connectedComponents(processedImg)
    components = 0
    for row in cclImg:
        maxOnRow = max(row)
        if maxOnRow > components:
            components = maxOnRow
    # -1 due to the "500 microns" bar at the bottom
    print(components-1)

    if plot == 1:
        # Cute plotting stuff that I completely stole.
        labelHue = np.uint8(179*cclImg/np.max(cclImg))
        blankCh = 255*np.ones_like(labelHue)
        labeledImg = cv2.merge([labelHue, blankCh, blankCh])
        # cvt to BGR for display
        labeledImg = cv2.cvtColor(labeledImg, cv2.COLOR_HSV2BGR)
        # set bg label to black
        labeledImg[labelHue == 0] = 0
        plt.imshow(labeledImg)
        plt.show()


if __name__ == "__main__":
    plotAll = 1
    #intensityHistogramTest()
    binaryImg = threshold(plotAll)
    erosion(binaryImg, plotAll)
    dilation(binaryImg, plotAll)
    opening(binaryImg, plotAll)
    closing(binaryImg, plotAll)
    connectedComponentLabeling(erosion(opening(binaryImg)), 1)
