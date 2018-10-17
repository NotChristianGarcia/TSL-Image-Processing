"""
Basic reading in of images. Color conversions, splits/indexing, and cuts.
"""

import time
import cv2
import matplotlib.pyplot as plt


def readRGB():
    """
    Part One
    Read an image as RGB and display it on the screen (use opencv, matplotlib)
    """
    imageBGR = cv2.imread("testIMG.jpg")
    imageRGB = cv2.cvtColor(imageBGR, cv2.COLOR_BGR2RGB)
    plt.imshow(imageRGB)
    plt.show()


def readGRAY():
    """
    Part Two
    Read an image as grayscale and display it
    """
    imageGray = cv2.imread("testIMG.jpg", cv2.IMREAD_GRAYSCALE)
    plt.imshow(imageGray, cmap='gray')
    plt.show()


def convertGRAY():
    """
    Part Three
    Read an image as RGB and convert it to grayscale, display it
    """
    imageBGR = cv2.imread("testIMG.jpg")
    imageGray = cv2.cvtColor(imageBGR, cv2.COLOR_BGR2GRAY)
    plt.imshow(imageGray, cmap='gray')
    plt.show()


def splitRGB():
    """
    Part Four
    Read an image as RGB and separate the image's color channels (use split() or
    numpy indexing - Which method is computationally faster?)
    """
    imageBGR = cv2.imread("testIMG.jpg")
    imageRGB = cv2.cvtColor(imageBGR, cv2.COLOR_BGR2RGB)

    startOne = time.clock()
    r1, g1, b1 = cv2.split(imageRGB)
    elapsedOne = time.clock() - startOne

    startTwo = time.clock()
    r2 = imageRGB[:, :, 0]
    g2 = imageRGB[:, :, 1]
    b2 = imageRGB[:, :, 2]
    elapsedTwo = time.clock() - startTwo

    print("Using split function: {}\nUsing indexing: {}".format(elapsedOne, elapsedTwo))
    if elapsedOne < elapsedTwo:
        print("Split function is faster than indexing")
    elif elapsedTwo < elapsedOne:
        print("Indexing is faster than split function")

    red = imageRGB.copy()
    red[:, :, 1] = 0
    red[:, :, 2] = 0
    plt.imshow(red)
    plt.show()

    green = imageRGB.copy()
    green[:, :, 0] = 0
    green[:, :, 2] = 0
    plt.imshow(green)
    plt.show()

    blue = imageRGB.copy()
    blue[:, :, 0] = 0
    blue[:, :, 1] = 0
    plt.imshow(blue)
    plt.show()


def imageCut():
    """
    Part Five
    Cut out a rectangular section of the image and display it on screen (use numpy indexing)
    """
    imageBGR = cv2.imread("testIMG.jpg")
    imageRGB = cv2.cvtColor(imageBGR, cv2.COLOR_BGR2RGB)
    cutImage = imageRGB[40:350, 200:620, :]
    plt.imshow(cutImage)
    plt.show()


if __name__ == "__main__":
    readRGB()
    readGRAY()
    convertGRAY()
    splitRGB()
    imageCut()
