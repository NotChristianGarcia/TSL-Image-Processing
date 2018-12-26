"""
Stuff
"""
import os
import pickle
import numpy as np

CODE_DIR = os.path.dirname(__file__)
PATH_DICT = {
    "train": CODE_DIR + "/cifar-100/train",
    "test": CODE_DIR + "/cifar-100/test"
    }

LABEL_LIST = [
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

def unpickle(data_path):
    with open(data_path, "rb") as pickled_file:
        data = pickle.load(pickled_file, encoding="bytes")
    return data

def get_data(data_type, as_str=True):
    data_path = PATH_DICT[str.lower(data_type)]

    data = unpickle(data_path)
    images = np.array(data[b"data"])
    labels = np.array(data[b"fine_labels"]).reshape(-1, 1)

    if as_str:
        train_labels_str = np.array([LABEL_LIST[i[0]] for i in labels])
        return images, labels, train_labels_str
    return images, labels

def kfolds(k, data_type="train"):
    data_path = PATH_DICT[str.lower(data_type)]
    data = unpickle(data_path)

    spacing = int(np.floor(len(data[b"data"])/k))
    image_folds = np.ndarray((k, spacing, 3072))
    label_folds = np.ndarray((k, spacing))
    for i in range(k):
        image_folds[i] = np.array(data[b"data"][i*spacing:((i+1)*spacing)])
        label_folds[i] = np.array(data[b"fine_labels"][i*spacing:((i+1)*spacing)])
    return image_folds, label_folds
