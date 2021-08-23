import numpy as np
import matplotlib.pyplot as plt
from utilities import *
from scipy.signal import stft, chirp, spectrogram
import itertools
from RANSAC import RANSAC_fit
from sklearn import linear_model
import json


def make_chirps(amp=1, mu=0, sigmas=[0], second_chirp=False):

    # Define parameters
    T = [1e-4]
    freqs = list(itertools.combinations([1300e6, 2300e6], 2))
    combs = list(itertools.product(T, freqs))

    # Initiate chirps dict
    chirps = {'fs': [],
              'T': [],
              'linear': [],
              'signal': [],
              'snr': [],
              'sigma': [],
              'spec': []
             }
    
    for sigma in sigmas:
        for i, ((T), (f0, f1)) in enumerate(combs):

            # Define signal
            fs = int(4800e6)
            t = np.linspace(0, T, int(T*fs))
            
            # Chirp
            signal = amp*chirp(t, f0=f0, f1=f1, t1=T, method='linear')

            # Linear Signal
            linear = [[0, f0], [T, f1]]

            # add another, fixed chirp to test hough
            if second_chirp:
                scnd_chirp = amp*chirp(t, f0=3000, f1=5000, t1=T, method='linear')
                signal = signal + scnd_chirp
                linear = [linear, [[0, 3000], [T, 5000]]]

            # Add noise
            noise = np.random.normal(mu, sigma, signal.shape)
            noisy_signal = signal + noise

            # Calculate SNR
            snr = np.around(calcSNR(signal, noisy_signal), 2)

            # Get stft
            _, _, Zxx = stft(noisy_signal, fs=fs, nperseg=2048, noverlap=64)

            # Append to dict
            chirps['fs'].append(fs)
            chirps['T'].append(T)
            chirps['linear'].append(linear)
            chirps['signal'].append(noisy_signal.tolist())
            chirps['snr'].append(snr.tolist())
            chirps['sigma'].append(sigma)
            chirps['spec'].append(np.abs(Zxx).tolist())
    
    return chirps


def chirp_test(data, graph=False, CRE=True):
    ''' Test '''

    # # Get sample
    # try:
    #     sample = data['sigma'].index(sigma)
    # except ValueError:
    #     print('Sigma not found!')
    #     return
    if CRE:
        cres = []
        snrs = []

    for sample in range(len(data['signal'])):

        fs = data['fs'][sample]
        T = data['T'][sample]
        S = np.array(data['spec'][sample])
        snr = np.array(data['snr'][sample])
        linear = np.array(data['linear'][sample])
        f_step, t_step = S.shape

        # Build axes
        t = np.linspace(0, T, t_step)
        f = np.linspace(0, fs/2, f_step)

        # Extract time frequency curve
        X, y = extractTimeFrequencyCurve(S, fs, T)
        y = medianFilter(y, N_med=9)

        # Our RANSAC
        line_X = np.arange(X.min(), X.max())[:, np.newaxis]
        a, b = RANSAC_fit(X, y, n_iterations=5000)
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

        if CRE:
            # calculate errors
            chirpSlope, chirpIntercept = getSlopeAndInterceptFromPoints(linear[0, 0], linear[0, 1], linear[1, 0], linear[1, 1])
            cre = calcCRE(chirpSlope, chirpIntercept, a, b, 0, len(data['signal'][sample]) / fs)
            cres.append(cre)
            snrs.append(snr)
            print(f'CRE of {snr} [db] = {cre} [%]')
        
        if graph:
            # Plots
            fig, axs = plt.subplots(2)

            # Plot STFT
            # axs[0].title(f'fs {fs} | SNR {snr}')
            axs[0].pcolormesh(t*1e6, f*1e-6, S, shading='gouraud')
            axs[0].set(xlabel=r'$Time [\mu s]$', ylabel='Frequency [MHz]')
            # axs[0].colorbar()
            # axs[0].show()

            # Inlires & Outlires
            axs[1].scatter(X[inlier_mask]*1e6, y[inlier_mask]*1e-6, color='yellowgreen', marker='.', label='Inliers')
            axs[1].scatter(X[outlier_mask]*1e6, y[outlier_mask]*1e-6, color='gold', marker='.', label='Outliers')

            # Our RANSAC
            axs[1].plot(line_X*1e6, our_prediction*1e-6, color='red', linewidth=2, label='Our RANSAC')

            # Sklearn RANSAC
            axs[1].plot(line_X*1e6, line_y_ransac*1e-6, color='cornflowerblue', linewidth=2, label='sklearn RANSAC')

            # Sklearn Linear Regressor
            axs[1].plot(line_X*1e6, line_y_lin_regres*1e-6, color='black', linewidth=2, label='sklearn Linear Regressor')

            # Real Linear Chirp
            axs[1].plot(linear[:, 0]*1e6, linear[:, 1]*1e-6, linewidth=1, label='Real Linear Chirp')

            axs[1].set(xlabel=r'$Time [\mu s]$', ylabel='Frequency [MHz]')
            axs[1].legend()

            plt.tight_layout()
            plt.show()

    if CRE:
        plt.plot(snrs, cres, '.')
        plt.plot(snrs, cres, '--')
        plt.xlabel('SNR [db]')
        plt.ylabel('CRE [%]')
        plt.show()



''' Make Data '''
# # Define sigmas
# sigmas = [0, 0.5, 1, 2, 2.5, 2.7, 2.8, 2.9, 3, 3.5, 4, 5, 6]

# # # Make chirps
# data = make_chirps(amp=1, mu=0, sigmas=sigmas)
# # # Save json
# # json_data = json.dumps(data)
# # jsonFile = open("data.json", "w")
# # jsonFile.write(json_data)
# # jsonFile.close()
# # Test
# chirp_test(data, graph=True, CRE=False)

''' Test Median Filter '''
# sigmas = [5]
# data = make_chirps(amp=1, mu=0, sigmas=sigmas)

# sample = 0
# fs = data['fs'][sample]
# T = data['T'][sample]
# S = np.array(data['spec'][sample])
# snr = np.array(data['snr'][sample])
# linear = np.array(data['linear'][sample])
# f_step, t_step = S.shape

# # Get stft points
# X, y = extractTimeFrequencyCurve(S, fs, T)

# # Medeian filter
# y_median = medianFilter(y.copy(), N_med=9)

# # Plot
# plt.plot(X, y)
# plt.plot(X, y_median)
# plt.show()

''' Test CRE '''
# sigmas = np.arange(20)
# data = make_chirps(amp=1, mu=0, sigmas=sigmas)
# chirp_test(data, graph=False, CRE=True)
