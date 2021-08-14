from make_chirps import *
from utilities import xyData2BinaryImage
import numpy as np
from skimage.transform import hough_line, hough_line_peaks
import matplotlib.pyplot as plt
from matplotlib import cm

def hough_test():
    sigmas = [0, 0.5, 1, 2, 2.5, 2.7, 2.8, 2.9, 3, 3.5, 4, 5, 6]
    data = make_chirps(amp=1, mu=0, sigmas=sigmas, second_chirp=True)

    try:
        sample = data['sigma'].index(2.9)
    except ValueError:
        print('Sigma not found!')
        return

    fs = data['fs'][sample]
    S = np.array(data['spec'][sample])
    snr = np.array(data['snr'][sample])
    linear = np.array(data['linear'][sample])
    f_step, t_step = S.shape

    # Build axes
    t = np.linspace(0, len(data['signal'][sample]) / fs, t_step)
    f = np.linspace(0, fs / 2, f_step)

    # Get stft points
    points = get_points(S)
    x = np.reshape(t[points[:, 1]], [-1, 1])
    y = f[points[:, 0]]

    # get image of two chirps
    img, scale, yshift = xyData2BinaryImage(x, y)

    # format original chirp line to fit image
    linear[:, :, 0] = linear[:, :, 0] / scale[0]
    linear[:, :, 1] = linear[:, :, 1] / scale[1]
    linear[:, :, 1] = yshift - linear[:, :, 1]

    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
    h, theta, d = hough_line(img, theta=tested_angles)

    plt.imshow(img, cmap=cm.gray)

    slopes = np.zeros(2)
    y_intercepts = np.zeros(2)

    for _, angle, dist, i in zip(*hough_line_peaks(h, theta, d), range(2)):
        (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
        slopes[i] = np.tan(angle + np.pi / 2)
        y_intercepts[i] = y0-slopes[i]*x0

    height, width = img.shape
    X = np.arange(width)
    Y = np.arange(height)

    y_line0 = slopes[0] * X + y_intercepts[0]
    y_line1 = slopes[1] * X + y_intercepts[1]

    # plot hough results
    plt.plot(X, y_line0, color='red', linewidth=2, label='Hough Output')
    plt.plot(X, y_line1, color='red', linewidth=2)

    # plot original chirps
    plt.plot(linear[0, :, 0], linear[0, :, 1], color='blue', linewidth=2, label='Real Linear Chirp')
    plt.plot(linear[1, :, 0], linear[1, :, 1], color='blue', linewidth=2,)

    xticks_labels = np.around(np.linspace(np.min(x), np.max(x), num=10), 2)
    yticks_labels = np.linspace(np.max(y), np.min(y), num=10).astype(int)
    xticks = np.linspace(0, img.shape[1], num=10)
    yticks = np.linspace(0, img.shape[0], num=10)

    plt.xticks(xticks, labels=xticks_labels)
    plt.yticks(yticks, labels=yticks_labels)

    plt.title("Hough transform on two chirp signals + white noise")
    plt.xlabel("Time intervals")
    plt.ylabel("Frequencies (Hz)")
    plt.legend()

    plt.tight_layout()
    plt.show()

hough_test()
