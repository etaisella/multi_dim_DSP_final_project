import numpy as np
import matplotlib.pyplot as plt
from utilities import *
from scipy.signal import stft, chirp, spectrogram
import itertools
from RANSAC import RANSAC_fit
from sklearn import linear_model
import json
from matplotlib import cm

def make_2slope_chirp(amp=1, mu=0, sigmas=[0]):
    # Define parameters
    T = 1e-4
    fs = int(4800e6)
    t_half = np.linspace(0, T/2, int((T/2)*fs))
    t_full = np.linspace(0, T, int(T*fs))

    # chirp
    signal1 = amp * chirp(t_half, f0=1300e6, f1=1500e6, t1=(T / 2), method='linear')
    signal2 = amp * chirp(t_half, f0=1500e6, f1=2300e6, t1=(T / 2), method='linear')
    unified_signal = np.zeros(signal1.size + signal2.size)
    unified_signal[:signal1.size] = signal1[:]
    unified_signal[signal1.size:] = signal2[:]

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
        linear = [[[0, 1300e6], [T/2, 1500e6]], [[T/2, 1500e6], [T, 2300e6]]]

        # Add noise
        noise = np.random.normal(mu, sigma, unified_signal.shape)
        noisy_signal = unified_signal + noise

        # Calculate SNR
        snr = np.around(calcSNR(unified_signal, noise), 2)

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
    
    for i, sigma in enumerate(sigmas):
        for ((T), (f0, f1)) in combs:

            # Define signal
            fs = int(4800e6)
            t = np.linspace(0, T, int(T*fs))
            
            # Chirp
            signal = amp*chirp(t, f0=f0, f1=f1, t1=T, method='linear')

            # Linear Signal
            linear = [[0, f0], [T, f1]]

            # add another, fixed chirp to test hough
            if second_chirp:
                scnd_chirp = amp*chirp(t, f0=2000e6, f1=1500e6, t1=T, method='linear')
                signal = signal + scnd_chirp
                linear = [linear, [[0, 2000e6], [T, 1500e6]]]

            # Add noise
            noise = np.random.normal(mu, sigma, signal.shape)
            noisy_signal = signal + noise

            # Calculate SNR
            snr = np.around(calcSNR(signal, noise), 2)

            # Get stft
            # _, _, Zxx = stft(noisy_signal, fs=fs, nperseg=2048, noverlap=64)
            _, _, Zxx = stft(noisy_signal, fs=fs, nperseg=2048, noverlap=2048-64)

            # Append to dict
            chirps['fs'].append(fs)
            chirps['T'].append(T)
            chirps['linear'].append(linear)
            chirps['signal'].append(noisy_signal.tolist())
            chirps['snr'].append(snr.tolist())
            chirps['sigma'].append(sigma)
            chirps['spec'].append(np.abs(Zxx).tolist())

            if (i+1) % 50 == 0:
                print(f'Finished {i+1} samples...')
    
    return chirps


def test_chirps(data, graph=False, CRE=True, plot_time_freq_curve=False, plot_only_predictions=False, plot_article_figures=False):
    if CRE:
        cres = {'lsm': [],
                'ransac': [],
                'ransac_no_median': []}
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
        X, y_no_med = extractTimeFrequencyCurve(S, fs, T)
        y = medianFilter(y_no_med.copy(), N_med=99)

        if plot_time_freq_curve:
            plt.title("Time / Frequency curve, SNR: %1.2f [dB]" % snr)
            plt.plot(X * 1e6, y_no_med * 1e-6, color='red', label='Peak frequency before Median filter')
            plt.plot(X*1e6, y * 1e-6, color='blue', label='Peak frequency after Median filter')
            plt.xlabel(r'$Time [\mu s]$')
            plt.ylabel('Frequency [MHz]')
            plt.legend()
            plt.show()

        # Our RANSAC - with median
        line_X = np.linspace(X.min(), X.max(), num=200)[:, np.newaxis]
        a, b = RANSAC_fit(X, y, n_iterations=5000, threshold=0.4e-4)
        our_prediction = a*line_X + b

        # Our RANSAC - without median
        a_no_med, b_no_med = RANSAC_fit(X, y_no_med, n_iterations=5000, threshold=0.4e-4)
        our_prediction_no_med = a_no_med*line_X + b_no_med

        # Sklearn RANSAC
        ransac = linear_model.RANSACRegressor()
        ransac.fit(X, y)
        inlier_mask = ransac.inlier_mask_
        outlier_mask = np.logical_not(inlier_mask)
        line_y_ransac = ransac.predict(line_X)
        a_ransac = ransac.estimator_.coef_
        b_ransac = ransac.estimator_.intercept_

        # Sklearn Linear Regressor
        lr = linear_model.LinearRegression()
        reg = lr.fit(X, y)
        line_y_lin_regres = lr.predict(line_X)

        if CRE:
            # calculate errors
            chirpSlope, chirpIntercept = getSlopeAndInterceptFromPoints(linear[0, 0], linear[0, 1], linear[1, 0], linear[1, 1])
            cre_ransac = calcCRE(chirpSlope, chirpIntercept, a, b, 0, T)
            cres['ransac'].append(cre_ransac)
            cre_ransac_no_med = calcCRE(chirpSlope, chirpIntercept, a_no_med, b_no_med, 0, T)
            cres['ransac_no_median'].append(cre_ransac_no_med)
            cre_lsm = calcCRE(chirpSlope, chirpIntercept, reg.coef_, reg.intercept_, 0, T)
            cres['lsm'].append(cre_lsm)
            snrs.append(snr)
            print(f'SNR = {snr} [db] | CRE RANSAC = {cre_ransac} [%] | CRE RANSAC WITHOUT MEDIAN = {cre_ransac_no_med} [%] | CRE LSM = {cre_lsm} [%]')

        if graph:
            # Plots
            fig, axs = plt.subplots(2)

            # Plot STFT
            axs[0].set_title('STFT of noisy chirp')
            axs[0].pcolormesh(t*1e6, f*1e-6, S, shading='gouraud')
            axs[0].set(xlabel=r'$Time [\mu s]$', ylabel='Frequency [MHz]')

            # Inlires & Outlires
            axs[1].set_title('Linear models')
            axs[1].scatter(X[inlier_mask]*1e6, y[inlier_mask]*1e-6,
                           color='yellowgreen', marker='.', label='Inliers')
            axs[1].scatter(X[outlier_mask]*1e6, y[outlier_mask]
                           * 1e-6, color='gold', marker='.', label='Outliers')

            # Our RANSAC - with median
            axs[1].plot(line_X*1e6, our_prediction*1e-6,
                        color='green', linewidth=1, label='RANSAC')

            # Our RANSAC - without median
            axs[1].plot(line_X*1e6, our_prediction_no_med*1e-6,
                        color='blue', linewidth=1, label='RANSAC NO MEDIAN')

            # # Sklearn RANSAC
            # axs[1].plot(line_X*1e6, line_y_ransac*1e-6,
            #             color='cornflowerblue', linewidth=2, label='sklearn RANSAC')

            # Sklearn Linear Regressor
            axs[1].plot(line_X*1e6, line_y_lin_regres*1e-6, color='red',
                        linewidth=2, label='LSM')

            # Real Linear Chirp
            axs[1].plot(linear[:, 0]*1e6, linear[:, 1]*1e-6,
                        linewidth=1, label='Real Linear Chirp')

            axs[1].set(xlabel=r'$Time [\mu s]$', ylabel='Frequency [MHz]')
            axs[1].legend()

            plt.tight_layout()
            plt.show()

        if plot_article_figures:
            fig = plt.figure()

            # plot 3d STFT
            ax = fig.add_subplot(111, projection='3d')
            X_mesh, Z_mesh = np.meshgrid(t * 1e6, f * 1e-6)
            ax.plot_surface(X_mesh, Z_mesh, S, cmap=cm.coolwarm)
            ax.set(xlabel=r'$Time [\mu s]$', ylabel='Frequency [MHz]')
            plt.tight_layout()
            plt.show()

            # plot STFT flat
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.pcolormesh(t * 1e6, f * 1e-6, S, shading='gouraud')
            ax.set(xlabel=r'$Time [\mu s]$', ylabel='Frequency [MHz]')
            plt.tight_layout()
            plt.show()

            # plot time frequency curve
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(X * 1e6, y_no_med * 1e-6, color='red', label='Peak frequency before Median filter')
            ax.set(xlabel=r'$Time [\mu s]$', ylabel='Frequency [MHz]')
            plt.ylim(0, 2500)
            plt.xlim(0, 100)
            plt.autoscale(False)
            plt.tight_layout()
            plt.show()

        if plot_only_predictions:
            # Inlires & Outlires
            # plt.set_title('Linear models')
            plt.scatter(X[inlier_mask]*1e6, y[inlier_mask]*1e-6,
                        color='yellowgreen', marker='.', label='Inliers')
            plt.scatter(X[outlier_mask]*1e6, y[outlier_mask]*1e-6,
                        color='gold', marker='.', label='Outliers')

            # Our RANSAC - with median
            plt.plot(line_X*1e6, our_prediction*1e-6,
                     color='green', linewidth=1, label='RANSAC')

            # Our RANSAC - without median
            plt.plot(line_X*1e6, our_prediction_no_med*1e-6,
                     color='blue', linewidth=1, label='RANSAC NO MEDIAN')

            # Sklearn Linear Regressor
            plt.plot(line_X*1e6, line_y_lin_regres*1e-6, color='red',
                     linewidth=2, label='LSM')

            # Real Linear Chirp
            plt.plot(linear[:, 0]*1e6, linear[:, 1]*1e-6,
                     linewidth=1, label='Real Linear Chirp')

            plt.xlabel(r'$Time [\mu s]$')
            plt.ylabel('Frequency [MHz]')
            plt.legend()

            plt.tight_layout()
            plt.show()

    if CRE:
        plt.plot(snrs, cres['ransac'], 'g.', label='RANSAC')
        plt.plot(snrs, cres['ransac'], 'g--')
        plt.plot(snrs, cres['ransac_no_median'], 'b.', label='RANSAC NO MEDIAN')
        plt.plot(snrs, cres['ransac_no_median'], 'b--')
        plt.plot(snrs, cres['lsm'], 'r.', label='LSM')
        plt.plot(snrs, cres['lsm'], 'r--')
        plt.xlabel('SNR [db]')
        plt.ylabel('CRE [%]')
        plt.legend()
        plt.show()



# def test_chirps(data, graph=False, CRE=True, plot_time_freq_curve=False, plot_article_figures=False):
#     if CRE:
#         cres = []
#         snrs = []

#     for sample in range(len(data['signal'])):

#         fs = data['fs'][sample]
#         T = data['T'][sample]
#         S = np.array(data['spec'][sample])
#         snr = np.array(data['snr'][sample])
#         linear = np.array(data['linear'][sample])
#         f_step, t_step = S.shape

#         # Build axes
#         t = np.linspace(0, T, t_step)
#         f = np.linspace(0, fs/2, f_step)

#         # Extract time frequency curve
#         X, y_no_med = extractTimeFrequencyCurve(S, fs, T)
#         y = medianFilter(y_no_med, N_med=9)

#         if plot_time_freq_curve:
#             plt.title("Time / Frequency curve, SNR: %1.2f [dB]" % snr)
#             plt.plot(X * 1e6, y_no_med * 1e-6, color='red', label='Peak frequency before Median filter')
#             plt.plot(X*1e6, y * 1e-6, color='blue', label='Peak frequency after Median filter')
#             plt.xlabel(r'$Time [\mu s]$')
#             plt.ylabel('Frequency [MHz]')
#             plt.legend()
#             plt.show()

#         # Our RANSAC
#         line_X = np.linspace(X.min(), X.max(), num = 200)[:, np.newaxis]
#         a, b = RANSAC_fit(X, y, n_iterations=100, threshold=0.4e-4)
#         our_prediction = a*line_X + b

#         # Sklearn RANSAC
#         ransac = linear_model.RANSACRegressor()
#         ransac.fit(X, y)
#         inlier_mask = ransac.inlier_mask_
#         outlier_mask = np.logical_not(inlier_mask)
#         line_y_ransac = ransac.predict(line_X)

#         # Sklearn Linear Regressor
#         lr = linear_model.LinearRegression()
#         lr.fit(X, y)
#         line_y_lin_regres = lr.predict(line_X)

#         if CRE:
#             # calculate errors
#             chirpSlope, chirpIntercept = getSlopeAndInterceptFromPoints(linear[0, 0], linear[0, 1], linear[1, 0], linear[1, 1])
#             cre = calcCRE(chirpSlope, chirpIntercept, a, b, 0, len(data['signal'][sample]) / fs)
#             cres.append(cre)
#             snrs.append(snr)
#             print(f'SNR = {snr} [db] | CRE = {cre} [%]')
        
#         if graph:
#             # Plots
#             fig, axs = plt.subplots(2)

#             # Plot STFT
#             # axs[0].title(f'fs {fs} | SNR {snr}')
#             axs[0].pcolormesh(t*1e6, f*1e-6, S, shading='gouraud')
#             axs[0].set(xlabel=r'$Time [\mu s]$', ylabel='Frequency [MHz]')
#             # axs[0].colorbar()
#             # axs[0].show()

#             # Inlires & Outlires
#             axs[1].scatter(X[inlier_mask]*1e6, y[inlier_mask]*1e-6, color='yellowgreen', marker='.', label='Inliers')
#             axs[1].scatter(X[outlier_mask]*1e6, y[outlier_mask]*1e-6, color='gold', marker='.', label='Outliers')

#             # Our RANSAC
#             axs[1].plot(line_X*1e6, our_prediction*1e-6, color='red', linewidth=1, label='Our RANSAC')

#             # Sklearn RANSAC
#             axs[1].plot(line_X*1e6, line_y_ransac*1e-6, color='cornflowerblue', linewidth=2, label='sklearn RANSAC')

#             # Sklearn Linear Regressor
#             axs[1].plot(line_X*1e6, line_y_lin_regres*1e-6, color='black', linewidth=2, label='sklearn Linear Regressor')

#             # Real Linear Chirp
#             axs[1].plot(linear[:, 0]*1e6, linear[:, 1]*1e-6, linewidth=1, label='Real Linear Chirp')

#             axs[1].set(xlabel=r'$Time [\mu s]$', ylabel='Frequency [MHz]')
#             axs[1].legend()

#             plt.tight_layout()
#             plt.show()

#         if plot_article_figures:
#             fig = plt.figure()

#             # plot 3d STFT
#             ax = fig.add_subplot(111, projection='3d')
#             X_mesh, Z_mesh = np.meshgrid(t * 1e6, f * 1e-6)
#             ax.plot_surface(X_mesh, Z_mesh, S, cmap=cm.coolwarm)
#             ax.set(xlabel=r'$Time [\mu s]$', ylabel='Frequency [MHz]')
#             plt.tight_layout()
#             plt.show()

#             # plot STFT flat
#             fig = plt.figure()
#             ax = fig.add_subplot(111)
#             ax.pcolormesh(t * 1e6, f * 1e-6, S, shading='gouraud')
#             ax.set(xlabel=r'$Time [\mu s]$', ylabel='Frequency [MHz]')
#             plt.tight_layout()
#             plt.show()

#             # plot time frequency curve
#             fig = plt.figure()
#             ax = fig.add_subplot(111)
#             ax.plot(X * 1e6, y_no_med * 1e-6, color='red', label='Peak frequency before Median filter')
#             ax.set(xlabel=r'$Time [\mu s]$', ylabel='Frequency [MHz]')
#             plt.ylim(0, 2500)
#             plt.xlim(0, 100)
#             plt.autoscale(False)
#             plt.tight_layout()
#             plt.show()

#     if CRE:
#         plt.plot(snrs, cres, '.')
#         plt.plot(snrs, cres, '--')
#         plt.xlabel('SNR [db]')
#         plt.ylabel('CRE [%]')
#         plt.show()




