import numpy as np
import sklearn.datasets
from scipy.io import loadmat
from scipy.spatial import distance_matrix
import os


DATASETS = [_ for _ in os.listdir('glenn_data') if _.endswith('.mat')]


def load_dataset(nm):
    DATA = loadmat(f'data/{nm}')
    y = DATA['d']
    y = np.ravel(y, order='C')
    print(nm, DATA['A'].shape, y.shape)
    return {'name': nm, 'X': DATA['A'], 'y': y}


def get_checkerboard(low=0, high=4, size=(1000, 2), seed=234):
    np.random.seed(seed)
    X = np.random.uniform(low, high, size=size)
    y = np.floor(X).sum(axis=1) % 2
    return {
        'name': f'{X.shape[1]}-d checkerboard',
        'X': X, 'y': y,
    }


def get_mnist_small():
    d = sklearn.datasets.load_digits()
    return {
        'name': 'sklearn digits',
        'X': d['data'],
        'y': d['target'],
    }


def get_iris():
    d = sklearn.datasets.load_iris()
    return {'name': 'iris', 'X': d['data'], 'y': d['target']}


def remove_close_points(dataset):
    # if points are too close in each coordinate they can cause problems
    # e.g. in iris, there are two points with coordinates
    # [6.4, 2.8, 5.6, 2.1]
    # [6.4, 2.8, 5.6, 2.2]
    # which caused magintude to behave poorly
    X = dataset['X']
    y = dataset['y']

    dist_mtx = distance_matrix(X, X)
    min_dist = dist_mtx[np.where(dist_mtx > 0)].min()

    coords_mean = np.zeros(shape=(X.shape[0], X.shape[0]))
    for i in range(coords_mean.shape[0]):
        for j in range(coords_mean.shape[1]):
            coords_mean[i, j] = np.mean(np.abs(X[i] - X[j]))

    indexes = np.where((coords_mean > 0) & (coords_mean < 0.5*min_dist))[0]
    indexes_to_remove = indexes[1:]
    X = np.delete(X, indexes_to_remove, axis=0)
    y = np.delete(y, indexes_to_remove, axis=0)

    dataset['X'] = X
    dataset['y'] = y

    return dataset


datasets = [
    load_dataset(nm)
    for nm
    in DATASETS
    if nm != 'censusdata.mat'
]
other_datasets = [get_checkerboard(), get_mnist_small(), get_iris()]
datasets = other_datasets + datasets
