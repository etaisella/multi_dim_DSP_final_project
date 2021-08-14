import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def xyData2BinaryImage(x, y):
    # transform column to row if necessary
    if x.shape[1] == 1:
        x = x.T

    # scale data to integers
    x_scale = 0.01
    y_scale = 10.0

    x_indices = np.round((x / x_scale)).astype(int)
    y_indices = np.round((y / y_scale)).astype(int)

    y_indices = max(y_indices) - y_indices

    # create image
    image = np.zeros((np.max(y_indices)+1, np.max(x_indices)+1))
    image[y_indices, x_indices] = 255

    scale = (x_scale, y_scale)

    return image, scale, max(y_indices)