# coding:utf-8

import matplotlib as mpl
mpl.rcParams['figure.dpi']= 75
import matplotlib.pyplot as plt

def box_to_rect(box, color="blue", linewidth=3):
    """

    :param box: (x_min,y_min,x_max,y_max)
    :param color:
    :param linewidth:
    :return:
    """
    """convert an anchor box to a matplotlib rectangle"""
    return plt.Rectangle(
        (box[0], box[1]), box[2]-box[0], box[3]-box[1],
        fill=False, edgecolor=color, linewidth=linewidth)