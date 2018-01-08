# coding:utf-8


from mxnet import nd
from mxnet import image
import numpy as np


def find_range(location, s):
    """
    :param x:
    :param y:
    :param location:
    :return:
    """
    x_min, y_min, x_max, y_max = location
    if x_min < 0 or y_min < 0 or x_max < 0 or y_max < 0:
        return -1
    x_center = 0.5 * (x_max + x_min)
    y_center = 0.5 * (y_max + y_min)
    ceil = 1. / s
    row = int(y_center / ceil)
    columns = int(x_center / ceil)
    index = (row * s + columns)
    return index


def translate_locat(x_min, y_min, x_max, y_max):
    """


    :param x_min:
    :param y_min:
    :param x_max:
    :param y_max:
    :return: x_center, y_center,w,h
    """

    center_x = 0.5 * (x_max + x_min)
    center_y = 0.5 * (y_max + y_min)
    w = (x_max - x_min)
    h = (y_max - y_min)

    return center_x, center_y, w, h


def translate_y(label, s, b, c):
    """

    :param y:
    :param s:
    :param b:
    :param c:
    :return:
    """
    y_ = label.asnumpy()
    labels = y_[:, 0] + 1
    location = y_[:, 1:]
    batch = len(label)
    new_y = np.zeros(shape=[batch, s * s * (b * 5 + c)])
    for i, locat in enumerate(location):

        labels_ = np.zeros(shape=(s * s, c))
        preds_ = np.zeros(shape=(s * s, b))
        location_ = np.zeros(shape=(s * s, b, 4))
        index = find_range(locat, s)

        if index == -1:
            labels_[:, 0] = 1
            labels_ = labels_.reshape((s * s * c,))
            preds_ = preds_.reshape((s * s * b,))
            location_ = location_.reshape((s * s * b * 4,))
            new_y[i] = (np.concatenate([labels_, preds_, location_], axis=0))
            continue
        for index_ in range(s * s):
            if index_ != index:
                labels_[index_][0] = 1

        labels_[index][int(labels[i])] = 1

        x_min, y_min, x_max, y_max = locat
        x, y, w, h = translate_locat(x_min, y_min, x_max, y_max)
        w, h = np.sqrt(w), np.sqrt(h)
        ceil = 1 / s
        x, y = round(x % ceil, 4), round(y % ceil, 4)
        for j, b_ in enumerate(preds_[index]):
            if b_ != 1:
                preds_[index][j] = 1
                location_[index][j] = [x, y, w, h]
                break


        labels_ = labels_.reshape((s * s * c,))
        preds_ = preds_.reshape((s * s * b,))
        location_ = location_.reshape((s * s * b * 4,))
        new_y[i] = (np.concatenate([labels_, preds_, location_], axis=0))
    return new_y


def deal_output(y: nd.NDArray, s, b, c):
    """

    :param y:
    :param s:
    :param b:
    :param c:
    :return:
    """
    label = y[:, 0:s * s * c]
    preds = y[:, s * s * c: s * s * c + s * s * b]
    location = y[:, s * s * c + s * s * b:]
    label = nd.reshape(label, shape=(-1, s * s, c))
    location = nd.reshape(location, shape=(-1, s * s, b, 4))
    return label, preds, location


def process_image(fname, data_shape, rgb_mean, rgb_std):
    with open(fname, 'rb') as f:
        im = image.imdecode(f.read())
    data = image.imresize(im, data_shape, data_shape)
    data = (data.astype('float32') - rgb_mean) / rgb_std
    return data.transpose((2, 0, 1)).expand_dims(axis=0), im
