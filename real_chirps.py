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
    del df

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
    y = medianFilter(y_no_med.copy(), N_med=99)
    line_X = np.linspace(X.min(), X.max(), num=200)[:, np.newaxis]

    # Our RANSAC
    a, b = RANSAC_fit(X, y)
    our_prediction = a*line_X + b

    # Our RANSAC - without median
    a_no_med, b_no_med = RANSAC_fit(X, y_no_med)
    our_prediction_no_med = a_no_med*line_X + b_no_med

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

    # calculate errors
    chirpSlope, chirpIntercept = getSlopeAndInterceptFromPoints(linear[0, 0], linear[0, 1], linear[1, 0], linear[1, 1])
    cre = calcCRE(chirpSlope, chirpIntercept, a, b, 0, T)
    print(f'Sample: {sample} | CRE: {cre} [%]')

    # Plot STFT
    plt.pcolormesh(t*1e3, f*1e-3, St, shading='gouraud', vmax=0.05)
    plt.xlabel('Time [ms]')
    plt.ylabel('Frequency [kHz]')
    plt.tight_layout()
    plt.show()

    # Inlires & Outlires
    plt.scatter(X[inlier_mask]*1e3, y[inlier_mask]*1e-3,
                color='yellowgreen', marker='.', label='Inliers')
    plt.scatter(X[outlier_mask]*1e3, y[outlier_mask]*1e-3,
                color='gold', marker='.', label='Outliers')

    # Our RANSAC
    plt.plot(line_X*1e3, our_prediction*1e-3,
                color='green', linewidth=1, label='RANSAC')

    # Our RANSAC - no median
    plt.plot(line_X*1e3, our_prediction_no_med*1e-3,
                color='blue', linewidth=1, label='RANSAC WITHOUT MEDIAN')

    # Sklearn Linear Regressor
    plt.plot(line_X*1e3, line_y_lin_regres*1e-3,
                color='red', linewidth=2, label='LSM')

    # Real Linear Chirp
    plt.plot(linear[:, 0]*1e3, linear[:, 1]*1e-3,
                linewidth=1, label='Real Linear Chirp')

    plt.xlabel('Time [ms]')
    plt.ylabel('Frequency [kHz]')
    plt.legend()

    plt.tight_layout()
    plt.show()


def test_real_chirps_cre():

    # Load dataframe
    df = pd.read_pickle('data/recordings_full.pkl')

    # Convert to numpy
    Ts_full = df.to_numpy()
    Ts_recorded = Ts_full[:, 500:3000]
    del df

    # Known parameters
    fs = 500e3
    T = 5e-3

    cres = {'lsm': [],
            'ransac': [],
            'ransac_no_median': []}
    n_samples = len(Ts_recorded)

    for sample in range(n_samples):
        ts = Ts_recorded[sample] - np.mean(Ts_recorded[sample])
        f, t, S = stft(ts, fs=500e3)
        f_step, t_step = S.shape
        St = np.abs(S)

        # Real chirp
        linear = np.array([[0, 100e3], [T, 20e3]])

        # Extract time frequency curve
        X, y_no_med = extractTimeFrequencyCurve(St, fs, T)
        y = medianFilter(y_no_med.copy(), N_med=99)
        line_X = np.linspace(X.min(), X.max(), num=200)[:, np.newaxis]

        # Our RANSAC
        a, b = RANSAC_fit(X, y)
        our_prediction = a*line_X + b

        # Our RANSAC - without median
        a_no_med, b_no_med = RANSAC_fit(X, y_no_med)
        our_prediction_no_med = a_no_med*line_X + b_no_med

        # Sklearn Linear Regressor
        lr = linear_model.LinearRegression()
        reg = lr.fit(X, y)
        line_y_lin_regres = lr.predict(line_X)

        # calculate errors
        chirpSlope, chirpIntercept = getSlopeAndInterceptFromPoints(linear[0, 0], linear[0, 1], linear[1, 0], linear[1, 1])
        cre_ransac = calcCRE(chirpSlope, chirpIntercept, a, b, 0, T)
        cres['ransac'].append(cre_ransac)
        cre_ransac_no_med = calcCRE(chirpSlope, chirpIntercept, a_no_med, b_no_med, 0, T)
        cres['ransac_no_median'].append(cre_ransac_no_med)
        cre_lsm = calcCRE(chirpSlope, chirpIntercept, reg.coef_, reg.intercept_, 0, T)
        cres['lsm'].append(cre_lsm)

        if (sample + 1) % 100 == 0:
            print(f'Finished {sample+1} samples...')
            # print(f'Sample {sample+1} : CRE RANSAC = {cre_ransac} [%] | CRE RANSAC WITHOUT MEDIAN = {cre_ransac_no_med} [%] | CRE LSM = {cre_lsm} [%]')

    p_over_50_lsm = np.around(np.sum(np.array(cres['lsm']) > 30)/n_samples*100, 2)
    p_over_50_ransac = np.around(np.sum(np.array(cres['ransac']) > 30)/n_samples*100, 2)
    p_over_50_ransac_no_median = np.around(np.sum(np.array(cres['ransac_no_median']) > 30)/n_samples*100, 2)

    print(f'CRE\'s over 30%: RANSAC = {p_over_50_ransac} [%] | RANSAC WITHOUT MEDIAN = {p_over_50_ransac_no_median} [%] | LSM = {p_over_50_lsm} [%]')
    print(f'Number of samples = {n_samples}')
