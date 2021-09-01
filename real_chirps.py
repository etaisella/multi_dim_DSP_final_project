import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utilities import *
from RANSAC import RANSAC_fit
from sklearn import linear_model
from scipy.signal import stft

def test_real_chirps():

    # Load dataframe
    df = pd.read_pickle('data/recordings.pkl')

    # Convert to numpy
    Ts_full = df.to_numpy()
    Ts_recorded = Ts_full[:, 500:3000]

    # Known parameters
    fs = 500e3
    T = 5e-3
    
    # Select random sample
    sample = np.random.randint(len(Ts_recorded))
    ts = Ts_recorded[sample] - np.mean(Ts_recorded[sample])
    f, t, S = stft(ts, fs=500e3)
    f_step, t_step = S.shape
    St = np.abs(S)

    # Real chirp
    linear = np.array([[0, 100e3], [T, 20e3]])

    # Extract time frequency curve
    X, y_no_med = extractTimeFrequencyCurve(St, fs, T)
    y = medianFilter(y_no_med, N_med=9)
    line_X = np.linspace(X.min(), X.max(), num = 200)[:, np.newaxis]

    # # Our RANSAC
    # a, b = RANSAC_fit(X, y, n_iterations=200, threshold=0.4e-4)
    # our_prediction = a*line_X + b

    # Sklearn RANSAC
    ransac = linear_model.RANSACRegressor()
    ransac.fit(X, y)
    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)
    line_y_ransac = ransac.predict(line_X)
    b = ransac.estimator_.intercept_
    a = ransac.estimator_.coef_
    print(f'RANSAC coef: {int(a)}, Real coef: {int((linear[0, 1] - linear[1, 1])/(linear[0, 0] - linear[1, 0]))}')

    # Sklearn Linear Regressor
    lr = linear_model.LinearRegression()
    lr.fit(X, y)
    line_y_lin_regres = lr.predict(line_X)

    # calculate errors
    chirpSlope, chirpIntercept = getSlopeAndInterceptFromPoints(linear[0, 0], linear[0, 1], linear[1, 0], linear[1, 1])
    cre = calcCRE(chirpSlope, chirpIntercept, a, b, 0, T)

    # Plots
    fig, axs = plt.subplots(2)

    # Plot STFT
    axs[0].pcolormesh(t*1e3, f*1e-3, St, shading='gouraud', vmax=0.05)
    axs[0].set(xlabel='Time [ms]', ylabel='Frequency [kHz]')

    # Inlires & Outlires
    axs[1].scatter(X[inlier_mask]*1e3, y[inlier_mask]*1e-3, color='yellowgreen', marker='.', label='Inliers')
    axs[1].scatter(X[outlier_mask]*1e3, y[outlier_mask]*1e-3, color='gold', marker='.', label='Outliers')

    # # Our RANSAC
    # axs[1].plot(line_X*1e3, our_prediction*1e-3, color='red', linewidth=1, label='Our RANSAC')

    # Sklearn RANSAC
    axs[1].plot(line_X*1e3, line_y_ransac*1e-3, color='cornflowerblue', linewidth=2, label='sklearn RANSAC')

    # Sklearn Linear Regressor
    axs[1].plot(line_X*1e3, line_y_lin_regres*1e-3, color='black', linewidth=2, label='sklearn Linear Regressor')

    # Real Linear Chirp
    axs[1].plot(linear[:, 0]*1e3, linear[:, 1]*1e-3, linewidth=1, label='Real Linear Chirp')

    axs[1].set(xlabel='Time [ms]', ylabel='Frequency [kHz]')
    axs[1].legend()

    plt.tight_layout()
    plt.show()
