import numpy as np
from scipy import signal
from skimage import morphology

def checkScorePerLine(img, x_line, y_line):
    height, width = img.shape
    x_line = np.clip(x_line, 0, width-1)
    y_line = np.clip(y_line, 0, height-1)
    hits = img[np.around(y_line).astype(int), x_line.astype(int)] > 0
    return np.sum(hits) / x_line.size

def xyData2BinaryImage(x, y):
    # transform column to row if necessary
    if x.shape[1] == 1:
        x = x.T

    # scale data to integers
    x_scale = 1e-7
    y_scale = 2e6

    x_indices = np.round((x / x_scale)).astype(int)
    y_indices = np.round((y / y_scale)).astype(int)

    yshift = max(y_indices)
    y_indices = yshift - y_indices

    # create image
    image = np.zeros((max(y_indices)+1, np.max(x_indices)+1))
    image[y_indices, x_indices] = 255

    for i in range(3):
        image = morphology.binary_dilation(image)

    scale = (x_scale, y_scale)

    return image, scale, yshift

def getSlopeAndInterceptFromPoints(x1, y1, x2, y2):
    slope = (y2 - y1) / (x2 - x1)
    intercept = y2 - slope * x2
    return slope, intercept

def calcCRE(slope_true, yIntercept_true, slope_tested, yIntercept_tested, min_x_val, max_x_val, samples=1000):
    xValues = np.linspace(min_x_val, max_x_val, samples)
    yValues_true = xValues * slope_true + yIntercept_true
    yValues_test = xValues * slope_tested + yIntercept_tested
    diff = np.absolute(yValues_true - yValues_test)
    diffRatio = diff / yValues_true

    errors = diffRatio > 0.01
    numErrors = np.sum(errors)
    return (samples-numErrors) / samples * 100

def calcSNR(sig, noise):

    # Calculate mean
    # _, avg_signal_p = signal.welch(sig)
    # _, avg_noise_p = signal.welch(noise)
    avg_signal_p = np.mean(np.power(sig, 2))
    avg_noise_p = np.mean(np.power(noise, 2))

    # Power SNR
    # snr_p = np.mean(avg_signal_p) / np.mean(avg_noise_p)
    snr_p = avg_signal_p / avg_noise_p

    # db
    snr = 10*np.log10(snr_p)

    return snr


def extractTimeFrequencyCurve(S, fs, T):

    # Initiate points array
    points = np.zeros([S.shape[1], 2], dtype=int)

    # Find frequencies
    for i, row in enumerate(S.T):  # Same as col in S
        points[i, 1] = i
        points[i, 0] = np.argmax(row)

    # Build axes
    f_step, t_step = S.shape
    t = np.linspace(0, T, t_step)
    f = np.linspace(0, fs/2, f_step)

    # Get Time & Frequency
    X = np.reshape(t[points[:, 1]], [-1, 1])
    y = f[points[:, 0]]

    return X, y

def medianFilter(signal, N_med=9):

    # Pad signal
    new_signal = np.zeros_like(signal)
    padded_signal = np.append(np.repeat(signal[0], N_med//2), signal)
    padded_signal = np.append(padded_signal, np.repeat(signal[-1], N_med//2))

    # Build signal based on median filter
    for i in range(len(signal)):
        sorted_short_signal = np.sort(padded_signal[i:i+N_med])
        new_signal[i] = sorted_short_signal[N_med//2]

    return new_signal
