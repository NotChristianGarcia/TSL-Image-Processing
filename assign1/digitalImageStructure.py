import cv2
import matplotlib.pyplot as plt
import time


# Part One
# Read an image as RGB and display it on the screen (use opencv, matplotlib)
def readRGB():
    bgr_image = cv2.imread("testIMG.jpg")
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb_image)
    plt.show()


# Part Two
# Read an image as grayscale and display it
def readGRAY():
    gray_image = cv2.imread("testIMG.jpg", cv2.IMREAD_GRAYSCALE)
    plt.imshow(gray_image, cmap='gray')
    plt.show()


# Part Three
# Read an image as RGB and convert it to grayscale, display it
def convertGRAY():
    bgr_image = cv2.imread("testIMG.jpg")
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    plt.imshow(gray_image, cmap='gray')
    plt.show()


# Part Four
# Read an image as RGB and separate the image's color channels (use split() or
# numpy indexing - Which method is computationally faster?)
def splitRGB():
    bgr_image = cv2.imread("testIMG.jpg")
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    start_one = time.clock()
    r1,g1,b1 = cv2.split(rgb_image)
    elapsed_one = time.clock() - start_one

    start_two = time.clock()
    r2 = rgb_image[:, :, 0]
    g2 = rgb_image[:, :, 1]
    b2 = rgb_image[:, :, 2]
    elapsed_two = time.clock() - start_two

    print("Using split function: {}\nUsing indexing: {}".format(elapsed_one, elapsed_two))
    if elapsed_one < elapsed_two:
        print("Split function is faster than indexing")
    elif elapsed_two < elapsed_one:
        print("Indexing is faster than split function")

    red = rgb_image.copy()
    red[:, :, 1] = 0
    red[:, :, 2] = 0
    plt.imshow(red)
    plt.show()

    green = rgb_image.copy()
    green[:, :, 0] = 0
    green[:, :, 2] = 0
    plt.imshow(green)
    plt.show()

    blue = rgb_image.copy()
    blue[:, :, 0] = 0
    blue[:, :, 1] = 0
    plt.imshow(blue)
    plt.show()


# Part Five
# Cut out a rectangular section of the image and display it on screen (use numpy indexing)
def imageCut():
    bgr_image = cv2.imread("testIMG.jpg")
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    cut_image = rgb_image[40:350, 200:620, :]
    plt.imshow(cut_image)
    plt.show()


if __name__ == "__main__":
    readRGB()
    readGRAY()
    convertGRAY()
    splitRGB()
    imageCut()
