import os
import numpy as np
from matplotlib import pyplot as plt

img_dataset_train_path = os.path.normpath('cifar-100/train')
img_dataset_test_path = os.path.normpath('cifar-100/test')

CIFAR100_LABELS_LIST = [
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

def display_img(name, img, show=True, cmap='gray'):
    figure = plt.figure()
    axes = plt.axes()
    axes.set_title(name)
    axes.imshow(img, cmap=cmap)
    if show:
        plt.show()

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_data(data_path, as_str=True):
    dataset = unpickle(data_path)
    imgs = np.array(dataset[b'data'], dtype=np.uint8)
    labels = np.array(dataset[b'fine_labels'], dtype=np.uint8).reshape(-1, 1)

    if as_str:
        train_lbls_str = np.array([CIFAR100_LABELS_LIST[i[0]] for i in labels])
        return imgs, labels, train_lbls_str

    return imgs, labels

class knn:
    def __init__(self, k):
        self.k = k

    def train(self, X, y):
        self.testData = X
        self.dataLabels = y

    def predict(self, X):
        items = X.shape[0]
        print(X.shape[0])
        predictions = np.zeros(items)

        for i in range(items):
            distances = np.reshape(np.sqrt(np.sum(np.square(self.testData - X[i, :]), axis=1)), (-1, 1))
 
            # Along with the distance stack the labels so that we can vote easily
            distance_label = np.hstack((distances, self.dataLabels))

            # Simple majoridataLabels voting based on the minimum distance
            sorted_distance = distance_label[distance_label[:, 0].argsort()]
            k_sorted_distance = sorted_distance[:self.k, :]
            (labels, occurence) = np.unique(k_sorted_distance[:, 1], return_counts=True)
            label = labels[occurence.argsort()[0]]
            predictions[i] = label
            
            print("{} of {}.".format(i+1, items))

        return predictions

def showImg(testImgs, predLabelNum, start, end):
    # Assume image size is 32 x 32. First 1024 px is r, next 1024 px is g, last 1024 px is b from the (r,g b) channel
    k = 0
    for i in range(start, end):
        j = testImgs[0][i]
        r = j[:1024].reshape(32, 32)
        g = j[1024:2048].reshape(32, 32)
        b = j[2048:].reshape(32, 32)
        rgb = np.dstack([r, g, b])
        plt.imshow(rgb)
        plt.title("real:pred - {}:{}".format(testImgs[2][i], CIFAR100_LABELS_LIST[int(predLabelNum[k])]))
        plt.show()
        k = k + 1

if __name__ == "__main__":
    start = 400
    end = 500
    trainData = get_data(img_dataset_train_path)
    testData = get_data(img_dataset_test_path)
    KNN = knn(3)
    KNN.train(trainData[0], trainData[1])
    predictions = KNN.predict(testData[0][400:450,:])
    showImg(testData, predictions, start, end)
