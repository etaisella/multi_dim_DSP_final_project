import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def xyData2BinaryImage(x, y):
    fig = plt.figure()
    new_plot = fig.add_subplot(111)
    new_plot.scatter(x, y, color='black', marker='.')
    new_plot.axis('off')
    canvas = FigureCanvas(fig)
    fig.canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape((int(height), int(width), 3))
    image = np.dot(image[..., :3], [0.299, 0.587, 0.114]) # rgb 2 gray
    image = 255 - image
    image[image > 0] = 255
    return image