import numpy as np

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

def getSlopeAndInterceptFromPoints(x1, y1, x2, y2):
    slope = (y2 - y1) / (x2 - x1)
    intercept = y2 - slope * x2
    return slope, intercept

def calcCRE(slope1, yIntercept1, slope2, yIntercept2, min_x_val, max_x_val, samples=1000):
    xValues = np.linspace(min_x_val, max_x_val, samples)
    yValues1 = xValues * slope1 + yIntercept1
    yValues2 = xValues * slope2 + yIntercept2
    diffRatio = yValues1/yValues2
    errors = (diffRatio < 0.99) + (diffRatio > 1.01)
    numErrors = np.sum(errors)
    return numErrors / samples
