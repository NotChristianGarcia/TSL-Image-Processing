import os
import numpy as np
from matplotlib import pyplot as plt

trainPath = os.path.normpath('cifar-100/train')
testPath = os.path.normpath('cifar-100/test')

cifarLabelList = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm'
]


def unpickle(file):
    import pickle
    with open(file, 'rb') as pickledFile:
        dataSet = pickle.load(pickledFile, encoding='bytes')
    return dataSet


def get_data(data_path, as_str=True):
    dataset = unpickle(data_path)
    imgs = np.array(dataset[b'data'], dtype=np.uint8)
    labels = np.array(dataset[b'fine_labels'], dtype=np.uint8).reshape(-1, 1)
    
    if as_str:
        trainLblsStr = np.array([cifarLabelList[i[0]] for i in labels])
        return imgs, labels, trainLblsStr
    return imgs, labels


class knn:
    def __init__(self, numK, trainImgs, numLabels):
        self.k = numK
        self.trainImgs = trainImgs
        self.numLabels = numLabels

    def predict(self, testImgs):
        items = len(testImgs)
        predictions = np.zeros(items)

        for i in range(items):
            distances = np.reshape(np.sqrt(np.sum(np.square(self.trainImgs - testImgs[i, :]), axis=1)), (-1, 1))

            # Along with the distance stack the labels so that we can vote easily
            distance_label = np.hstack((distances, self.numLabels))

            # Simple majoridataLabels voting based on the minimum distance
            sorted_distance = distance_label[distance_label[:, 0].argsort()]
            k_sorted_distance = sorted_distance[:self.k, :]
            (labels, occurence) = np.unique(k_sorted_distance[:, 1], return_counts=True)
            label = labels[occurence.argsort()[0]]
            predictions[i] = label

            print("{} of {}.".format(i+1, items))
        return predictions

    def checkAccuracy(self, testLabelNum, predLabelNum, testStart, testEnd):
        correct = 0
        dataSize = len(predLabelNum)
        for i, offsetI in enumerate(range(testStart, testEnd)):
            if testLabelNum[offsetI] == predLabelNum[i]:
                correct += 1
        print("{} correct in test of {}.".format(correct, dataSize))
        return correct


def showImg(testImgs, predLabelNum, testStart, testEnd):
    for i, offsetI in enumerate(range(testStart, testEnd)):
        j = testImgs[0][offsetI]
        r = j[:1024].reshape(32, 32)
        g = j[1024:2048].reshape(32, 32)
        b = j[2048:].reshape(32, 32)
        rgb = np.dstack([r, g, b])
        plt.imshow(rgb)
        plt.title("real:pred - {}:{}".format(testImgs[2][offsetI], cifarLabelList[int(predLabelNum[i])]))
        plt.show()


if __name__ == "__main__":
    testStart = 0
    testEnd = 500
    k = 3
    trainData = get_data(trainPath)
    testData = get_data(testPath)
    KNN = knn(k, trainData[0], trainData[1])
    predictions = KNN.predict(testData[0][testStart:testEnd, :])
    KNN.checkAccuracy(testData[1], predictions, testStart, testEnd)
    #showImg(testData, predictions, testStart, testEnd)
