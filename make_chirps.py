import numpy as np
import matplotlib.pyplot as plt
from utilities import calcCRE, getSlopeAndInterceptFromPoints
from scipy.signal import stft, chirp, spectrogram
import itertools
from RANSAC import RANSAC_fit
from sklearn import linear_model
import json


def make_chirps(amp=1, mu=0, sigmas=[0], second_chirp=False):

    # Define parameters
    T = [5]
    freqs = list(itertools.combinations([5000, 1000, 10000, 3000], 2))
    combs = list(itertools.product(T, freqs))

    # Initiate chirps dict
    chirps = {'fs': [],
              'linear': [],
              'signal': [],
              'snr': [],
              'sigma': [],
              'spec': []
             }
    
    for sigma in sigmas:
        for i, ((T), (f0, f1)) in enumerate(combs):

            # Define signal
            fs = int(max(f1, f0) * 2.5)
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
            snr = np.around(SNR(signal, noisy_signal), 2)

            # Get stft
            _, _, Zxx = stft(noisy_signal, fs=fs, nfft=256)

            # Append to dict
            chirps['fs'].append(fs)
            chirps['linear'].append(linear)
            chirps['signal'].append(noisy_signal.tolist())
            chirps['snr'].append(snr.tolist())
            chirps['sigma'].append(sigma)
            chirps['spec'].append(np.abs(Zxx).tolist())
    
    return chirps


def SNR(signal, noise):

    # Calculate mean
    avg_signal_p = np.mean(np.power(signal, 2))
    avg_noise_p = np.mean(np.power(noise, 2))

    # Power SNR
    snr_p = avg_signal_p / avg_noise_p

    # db
    snr = 10*np.log10(snr_p)

    return snr


def get_points(S):
    
    points = np.zeros([S.shape[1], 2], dtype=int)

    for i, row in enumerate(S.T): # Same as col in S
        points[i, 1] = i
        points[i, 0] = np.argmax(row)
    
    return points


def chirp_test(data, sigma):
    ''' Test '''

    # Get sample
    try:
        sample = data['sigma'].index(sigma)
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
    f = np.linspace(0, fs/2, f_step)

    # Get stft points
    points = get_points(S)
    X = np.reshape(t[points[:, 1]], [-1, 1])
    y = f[points[:, 0]]

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

    # Plots
    fig, axs = plt.subplots(2)

    # Plot STFT
    # axs[0].title(f'fs {fs} | SNR {snr}')
    axs[0].pcolormesh(t, f, S, shading='gouraud')
    axs[0].set(xlabel='Time [s]', ylabel='Frequency [Hz]')
    # axs[0].colorbar()
    # axs[0].show()

    # Inlires & Outlires
    axs[1].scatter(X[inlier_mask], y[inlier_mask], color='yellowgreen', marker='.', label='Inliers')
    axs[1].scatter(X[outlier_mask], y[outlier_mask], color='gold', marker='.', label='Outliers')

    # Our RANSAC
    axs[1].plot(line_X, our_prediction, color='red', linewidth=2, label='Our RANSAC')

    # Sklearn RANSAC
    axs[1].plot(line_X, line_y_ransac, color='cornflowerblue', linewidth=2, label='sklearn RANSAC')

    # Sklearn Linear Regressor
    axs[1].plot(line_X, line_y_lin_regres, color='black', linewidth=2, label='sklearn Linear Regressor')

    # Real Linear Chirp
    axs[1].plot(linear[:, 0], linear[:, 1], linewidth=1, label='Real Linear Chirp')

    axs[1].set(xlabel='Time [s]', ylabel='Frequency [Hz]')
    axs[1].legend()

    chirpSlope, chirpIntercept = getSlopeAndInterceptFromPoints(linear[0, 0], linear[0, 1], linear[1, 0], linear[1, 1])
    CRE = calcCRE(chirpSlope, chirpIntercept, a, b, 0, len(data['signal'][sample]) / fs)

    plt.tight_layout()
    plt.show()




''' Make Data '''
# Define sigmas
sigmas = [0, 0.5, 1, 2, 2.5, 2.7, 2.8, 2.9, 3, 3.5, 4, 5, 6]

# Make chirps
data = make_chirps(amp=1, mu=0, sigmas=sigmas)

# # Save json
# json_data = json.dumps(data)
# jsonFile = open("data.json", "w")
# jsonFile.write(json_data)
# jsonFile.close()

# Test
chirp_test(data, sigma=2.7)
