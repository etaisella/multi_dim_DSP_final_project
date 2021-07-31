import numpy as np
import sys
from matplotlib import pyplot as plt

from sklearn import linear_model, datasets

def RANSAC_fit(x, y, n_iterations=None, threshold=None, min_inliers=None):
    # calculate number of iterations
    if n_iterations == None:
        iters = (x.size**2) / 2
    else:
        iters = n_iterations

    # calculate threshold
    if threshold == None:
        # calculate Median absolute deviation
        thresh = 0.01*np.median(np.abs(y - np.median(y)))
    else:
        thresh = threshold

    # calculate min_inliers
    if min_inliers == None:
        min_inl = int(0.9*x.size)
    else:
        min_inl = min_inliers

    seen_pairs = np.zeros((x.size, x.size))
    best_model = np.zeros(2)
    best_distance = sys.float_info.max

    for i in range(int(iters)):
        # draw random sample
        while(1):
            random_pair = np.random.choice(x.size, 2, replace=False)
            if seen_pairs[random_pair[0], random_pair[1]] != True:
                seen_pairs[random_pair[0], random_pair[1]] = True
                break;

        # calculate distance from model
        x1 = np.ones_like(x) * x[random_pair[0]]
        x2 = np.ones_like(x) * x[random_pair[1]]
        y1 = np.ones_like(x) * y[random_pair[0]]
        y2 = np.ones_like(x) * y[random_pair[1]]

        diff_x = x2 - x1
        diff_y = y2 - y1

        if y.shape[0] != 1:
            y_col = np.zeros_like(x)
            y_col[:, 0] = y[:]
        else:
            y_col = y

        numerator = np.abs((diff_x*(y1 - y_col) - diff_y*(x1 - x)))
        denominator = np.sqrt(diff_x*diff_x + diff_y*diff_y)
        distance = numerator / denominator

        # calculate threshold
        #if threshold == None:
        #    # calculate Median absolute deviation
        #    thresh = np.median(np.abs(distance - np.median(distance)))
        #else:
        #    thresh = threshold

        # calculate amount of inliers
        num_inliers = (distance < thresh).sum()

        # determine best model so far
        if num_inliers > min_inl:
            distance_from_inliers = distance[distance < thresh].mean()
            if distance_from_inliers < best_distance:
                best_model = random_pair
                best_distance = distance_from_inliers

    # calculate a and b (ax + b = y) according to best pair
    x1 = x[best_model[0]]
    y1 = y[best_model[0]]
    x2 = x[best_model[1]]
    y2 = y[best_model[1]]

    a = (y1 - y2) / (x1 - x2)
    b = y1 - x1*a

    return a, b

def RANSAC_test():
    n_samples = 1000
    n_outliers = 50

    X, y, coef = datasets.make_regression(n_samples=n_samples, n_features=1,
                                          n_informative=1, noise=10,
                                          coef=True, random_state=0)

    # Add outlier data
    np.random.seed(0)
    X[:n_outliers] = 3 + 0.5 * np.random.normal(size=(n_outliers, 1))
    y[:n_outliers] = -3 + 10 * np.random.normal(size=n_outliers)

    # Fit line using all data
    lr = linear_model.LinearRegression()
    lr.fit(X, y)

    # Robustly fit linear model with RANSAC algorithm
    ransac = linear_model.RANSACRegressor()
    ransac.fit(X, y)
    a,b = RANSAC_fit(X, y, n_iterations=5000)
    print("a: %d\n", a)
    print("b: %d\n", b)
    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)

    # Predict data of estimated models
    line_X = np.arange(X.min(), X.max())[:, np.newaxis]
    line_y = lr.predict(line_X)
    line_y_ransac = ransac.predict(line_X)

    our_prediction = a*line_X + b

    # Compare estimated coefficients
    print("Estimated coefficients (true, linear regression, RANSAC):")
    print(coef, lr.coef_, ransac.estimator_.coef_)

    lw = 2

    plt.scatter(X[inlier_mask], y[inlier_mask], color='yellowgreen', marker='.',
                label='Inliers')
    plt.scatter(X[outlier_mask], y[outlier_mask], color='gold', marker='.',
                label='Outliers')
    plt.plot(line_X, our_prediction, color='red', linewidth=lw, label='Our RANSAC')
    plt.plot(line_X, line_y, color='navy', linewidth=lw, label='Linear regressor')
    plt.plot(line_X, line_y_ransac, color='cornflowerblue', linewidth=lw,
             label='sklearn RANSAC')
    plt.legend(loc='lower right')
    plt.xlabel("Input")
    plt.ylabel("Response")
    plt.show()

#RANSAC_test()