import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utilities import *
from RANSAC import RANSAC_fit
from sklearn import linear_model
import feather

def real_chirps():
    
    # Load dataframe
    df = pd.read_pickle('stfts.pkl')

    # Convert to numpy
    X = df['specs'].to_numpy()
    
    S = []
    for x in X:
        S.append(list(x))
    S = np.array(S)

    del X

    # Build axes
    n_samples, f_step, t_step = S.shape
    t = np.linspace(0, 16e-3, t_step)
    f = np.linspace(0, 250e3, f_step)
    St = S[0, :f_step//2, :t_step//2]

    # Extract time frequency curve
    X, y_no_med = extractTimeFrequencyCurve(St, fs=f[f_step//2]*2, T=t[t_step//2])
    y = medianFilter(y_no_med, N_med=9)

    # Our RANSAC
    line_X = np.linspace(X.min(), X.max(), num = 200)[:, np.newaxis]
    a, b = RANSAC_fit(X, y, n_iterations=5000, threshold=0.4e-4)
    our_prediction = a*line_X + b

    # Sklearn RANSAC
    ransac = linear_model.RANSACRegressor()
    ransac.fit(X, y)
    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)
    line_y_ransac = ransac.predict(line_X)

    # Sklearn Linear Regressor
    lr = linear_model.LinearRegression()
    lr.fit(X, y)
    line_y_lin_regres = lr.predict(line_X)

    # Plots
    fig, axs = plt.subplots(2)

    # Plot STFT
    axs[0].pcolormesh(t[:t_step//2]*1e3, f[:f_step//2]*1e-3, St, shading='gouraud')
    axs[0].set(xlabel='Time [ms]', ylabel='Frequency [kHz]')

    # Inlires & Outlires
    axs[1].scatter(X[inlier_mask]*1e3, y[inlier_mask][:f_step//2]*1e-3, color='yellowgreen', marker='.', label='Inliers')
    axs[1].scatter(X[outlier_mask]*1e3, y[outlier_mask][:f_step//2]*1e-3, color='gold', marker='.', label='Outliers')

    # Our RANSAC
    axs[1].plot(line_X*1e3, our_prediction*1e-3, color='red', linewidth=1, label='Our RANSAC')

    # Sklearn RANSAC
    axs[1].plot(line_X*1e3, line_y_ransac[:f_step//2]*1e-3, color='cornflowerblue', linewidth=2, label='sklearn RANSAC')

    # Sklearn Linear Regressor
    axs[1].plot(line_X*1e3, line_y_lin_regres*1e-3, color='black', linewidth=2, label='sklearn Linear Regressor')

    # # Real Linear Chirp
    # axs[1].plot(linear[:, 0]*1e3, linear[:, 1]*1e-3, linewidth=1, label='Real Linear Chirp')

    axs[1].set(xlabel='Time [ms]', ylabel='Frequency [kHz]')
    axs[1].legend()

    plt.tight_layout()
    plt.show()


# real_chirps()