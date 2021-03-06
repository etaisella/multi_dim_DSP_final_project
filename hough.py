from make_chirps import *
from utilities import xyData2BinaryImage, checkScorePerLine
import numpy as np
from skimage.transform import hough_line, hough_line_peaks
import matplotlib.pyplot as plt
from matplotlib import cm

def hough_test(plot_article_figures=False, CRE=False):
    sigmas = np.arange(2, 24)
    data = make_2slope_chirp(amp=1, mu=0, sigmas=sigmas)

    try:
        sample = data['sigma'].index(9)
    except ValueError:
        print('Sigma not found!')
        return

    T = [1e-4]
    fs = data['fs'][sample]
    S = np.array(data['spec'][sample])
    snr = np.array(data['snr'][sample])
    linear = np.array(data['linear'][sample])
    f_step, t_step = S.shape

    # Build axes
    t = np.linspace(0, len(data['signal'][sample]) / fs, t_step)
    f = np.linspace(0, fs / 2, f_step)

    # Get stft points
    x, y_no_med = extractTimeFrequencyCurve(S, fs, T)
    y = y_no_med
    #y = medianFilter(y_no_med, N_med=3)

    # get image of two chirps
    img, scale, yshift = xyData2BinaryImage(x, y)
    im_height, im_width = img.shape

    # format original chirp line to fit image
    linear[:, :, 0] = linear[:, :, 0] / scale[0]
    linear[:, :, 1] = linear[:, :, 1] / scale[1]
    linear[:, :, 1] = yshift - linear[:, :, 1]

    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
    h, theta, d = hough_line(img, theta=tested_angles)

    slopes = np.zeros(2)
    y_intercepts = np.zeros(2)

    for _, angle, dist, i in zip(*hough_line_peaks(h, theta, d), range(2)):
        (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
        slopes[i] = np.tan(angle + np.pi / 2)
        y_intercepts[i] = y0-slopes[i]*x0

    # get intersection point
    x_intersection = (y_intercepts[0] - y_intercepts[1]) / (slopes[1] - slopes[0])

    if plot_article_figures:
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(5,3))

        ax1.imshow(img, cmap=cm.gray)
        ax1.set_title("Original Image")
        ax1.set_axis_off()

        angle_step = 0.5 * np.diff(theta).mean()
        d_step = 0.5 * np.diff(d).mean()
        bounds = [np.rad2deg(theta[0] - angle_step),
                  np.rad2deg(theta[-1] + angle_step),
                  d[-1] + d_step, d[0] - d_step]

        ax2.imshow(np.log(1 + h), cmap=cm.gray,extent=bounds, aspect='auto')
        ax2.set_title("Hough Transform")
        ax2.set_xlabel("Angle [degrees]")
        ax2.set_ylabel("Distance from origin [pixels]")

        ax3.imshow(img, cmap=cm.gray)
        ax3.set_title("Lines corresponsing to Hough transform Maximas")
        x_vals = np.arange(im_width)
        y_vals1 = slopes[0] * x_vals + y_intercepts[0]
        y_vals2 = slopes[1] * x_vals + y_intercepts[1]
        ax3.plot(x_vals, y_vals1, color='red', linewidth=3)
        ax3.plot(x_vals, y_vals2, color='red', linewidth=3)
        ax3.set_axis_off()

        plt.show()

    plt.imshow(img, cmap=cm.gray)

    # check line locations
    X1 = np.arange(int(x_intersection))
    X2 = X1 + int(x_intersection)
    y_line0_left = slopes[0] * X1 + y_intercepts[0]
    y_line0_right = slopes[0] * X2 + y_intercepts[0]
    y_line1_left = slopes[1] * X1 + y_intercepts[1]
    y_line1_right = slopes[1] * X2 + y_intercepts[1]

    score_line0_left = checkScorePerLine(img, X1, y_line0_left)
    score_line1_left = checkScorePerLine(img, X1, y_line1_left)
    #score_line0_right = checkScorePerLine(img, X2, y_line0_right)
    #score_line1_right = checkScorePerLine(img, X2, y_line1_right)

    if (score_line0_left > score_line1_left):
        y_line0 = y_line0_left
        y_line1 = y_line1_right
        slopes_0, y_intercepts_0 = slopes[0], y_intercepts[0]
        slopes_1, y_intercepts_1 = slopes[1], y_intercepts[1]
    else:
        y_line0 = y_line1_left
        y_line1 = y_line0_right
        slopes_0, y_intercepts_0 = slopes[1], y_intercepts[1]
        slopes_1, y_intercepts_1 = slopes[0], y_intercepts[0]

    # plot hough results
    plt.plot(X1, y_line0, color='red', linewidth=3, label='Hough Output')
    plt.plot(X2, y_line1, color='red', linewidth=3)

    # plot original chirps
    unified_linear = np.zeros((2,1,4))
    plt.plot(linear[0, :, 0], linear[0, :, 1], color='blue', linewidth=2, label='Real Linear Chirp')
    plt.plot(linear[1, :, 0], linear[1, :, 1], color = 'blue', linewidth = 2)

    xticks_labels = np.around(np.linspace(np.min(x)*1e6, np.max(x)*1e6, num=10), 2)
    yticks_labels = np.linspace(np.max(y)*1e-6, np.min(y)*1e-6, num=10).astype(int)
    xticks = np.linspace(0, img.shape[1], num=10)
    yticks = np.linspace(0, img.shape[0], num=10)

    plt.xticks(xticks, labels=xticks_labels)
    plt.yticks(yticks, labels=yticks_labels)

    #plt.title("Hough transform on two chirp signals + white noise")
    plt.xlabel(r'$Time [\mu s]$')
    plt.ylabel("Frequencies [Hz]")
    plt.legend()

    plt.tight_layout()
    plt.show()

    if CRE:
        chirpSlope, chirpIntercept = getSlopeAndInterceptFromPoints(linear[0, 0, 0], linear[0, 0, 1], linear[0, 1, 0], linear[0, 1, 1])
        cre_0 = calcCRE(chirpSlope, chirpIntercept, slopes_0, y_intercepts_0, 0, T)
        chirpSlope, chirpIntercept = getSlopeAndInterceptFromPoints(linear[1, 0, 0], linear[1, 0, 1], linear[1, 1, 0], linear[1, 1, 1])
        cre_1 = calcCRE(chirpSlope, chirpIntercept, slopes_1, y_intercepts_1, 0, T)
        print(f'CRE 0 = {cre_0} [%] | CRE 1 = {cre_1} [%] | MEAN CRE\'s = {(cre_0 + cre_1) / 2}')

    if plot_article_figures:
        # plot original image
        plt.imshow(img, cmap=cm.gray)
        plt.xticks(xticks, labels=xticks_labels)
        plt.yticks(yticks, labels=yticks_labels)
        plt.xlabel(r'$Time [\mu s]$')
        plt.ylabel("Frequencies [Hz]")
        plt.show()

def hough_cre_test():
    
    n_sigmas = 15
    cre_0_mean = np.zeros(n_sigmas)
    cre_1_mean = np.zeros(n_sigmas)

    for test in range(100):

        sigmas = np.linspace(1, 20, n_sigmas)
        data = make_2slope_chirp(amp=1, mu=0, sigmas=sigmas)
        
        cres = {'cre_0': [],
                'cre_1': [],
                'cre_mean': []}
        snrs = []


        for sample in range(len(data['signal'])):

            T = [1e-4]
            fs = data['fs'][sample]
            S = np.array(data['spec'][sample])
            snr = np.array(data['snr'][sample])
            linear = np.array(data['linear'][sample])
            f_step, t_step = S.shape

            # Build axes
            t = np.linspace(0, len(data['signal'][sample]) / fs, t_step)
            f = np.linspace(0, fs / 2, f_step)

            # Get stft points
            x, y_no_med = extractTimeFrequencyCurve(S, fs, T)
            y = y_no_med
            #y = medianFilter(y_no_med, N_med=3)

            # get image of two chirps
            img, scale, yshift = xyData2BinaryImage(x, y)
            im_height, im_width = img.shape

            # format original chirp line to fit image
            linear[:, :, 0] = linear[:, :, 0] / scale[0]
            linear[:, :, 1] = linear[:, :, 1] / scale[1]
            linear[:, :, 1] = yshift - linear[:, :, 1]

            tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
            h, theta, d = hough_line(img, theta=tested_angles)

            slopes = np.zeros(2)
            y_intercepts = np.zeros(2)

            for _, angle, dist, i in zip(*hough_line_peaks(h, theta, d), range(2)):
                (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
                slopes[i] = np.tan(angle + np.pi / 2)
                y_intercepts[i] = y0-slopes[i]*x0

            # get intersection point
            if slopes[1] == slopes[0]:
                slopes[1] = slopes[1] + 0.001
            x_intersection = (y_intercepts[0] - y_intercepts[1]) / (slopes[1] - slopes[0])

            # check line locations
            try:
                X1 = np.arange(int(x_intersection))
                X2 = X1 + int(x_intersection)
            except OverflowError:
                X1 = np.arange(len(y_intercepts))
                X2 = np.arange(len(y_intercepts))
            
            y_line0_left = slopes[0] * X1 + y_intercepts[0]
            y_line0_right = slopes[0] * X2 + y_intercepts[0]
            y_line1_left = slopes[1] * X1 + y_intercepts[1]
            y_line1_right = slopes[1] * X2 + y_intercepts[1]

            score_line0_left = checkScorePerLine(img, X1, y_line0_left)
            score_line1_left = checkScorePerLine(img, X1, y_line1_left)
            #score_line0_right = checkScorePerLine(img, X2, y_line0_right)
            #score_line1_right = checkScorePerLine(img, X2, y_line1_right)

            if (score_line0_left > score_line1_left):
                y_line0 = y_line0_left
                y_line1 = y_line1_right
                slopes_0, y_intercepts_0 = slopes[0], y_intercepts[0]
                slopes_1, y_intercepts_1 = slopes[1], y_intercepts[1]
            else:
                y_line0 = y_line1_left
                y_line1 = y_line0_right
                slopes_0, y_intercepts_0 = slopes[1], y_intercepts[1]
                slopes_1, y_intercepts_1 = slopes[0], y_intercepts[0]

            chirpSlope, chirpIntercept = getSlopeAndInterceptFromPoints(linear[0, 0, 0], linear[0, 0, 1], linear[0, 1, 0], linear[0, 1, 1])
            # cre_0 = calcCRE(chirpSlope, chirpIntercept, slopes[0], y_intercepts[0], linear[0, 0, 0], linear[0, -1, 0])
            cre_0 = calcCRE(chirpSlope, chirpIntercept, slopes_0, y_intercepts_0, 0, T, samples=500)
            cres['cre_0'].append(cre_0)
            chirpSlope, chirpIntercept = getSlopeAndInterceptFromPoints(linear[1, 0, 0], linear[1, 0, 1], linear[1, 1, 0], linear[1, 1, 1])
            # cre_1 = calcCRE(chirpSlope, chirpIntercept, slopes[1], y_intercepts[1], linear[1, 0, 0], linear[1, -1, 0])
            cre_1 = calcCRE(chirpSlope, chirpIntercept, slopes_1, y_intercepts_1, 0, T, samples=500)
            cres['cre_1'].append(cre_1)
            # print(f'CRE 0 = {cre_0} [%] | CRE 1 = {cre_1} [%] | MEAN CRE\'s = {(cre_0 + cre_1) / 2}')
            snrs.append(snr)
            cres['cre_mean'].append((cre_0 + cre_1) / 2)

        if (test + 1) % 10 == 0:
            print(f'Finished {test+1} tests...')

        # Add cres
        cre_0_mean += np.array(cres['cre_0'])
        cre_1_mean += np.array(cres['cre_1'])

    cre_0_mean = cre_0_mean / (test+1)
    cre_1_mean = cre_1_mean / (test+1)

    plt.plot(snrs, cre_0_mean, 'g.', label='HOUGH Left line')
    plt.plot(snrs, cre_0_mean, 'g--')
    plt.plot(snrs, cre_1_mean, 'r.', label='HOUGH Right line')
    plt.plot(snrs, cre_1_mean, 'r--')
    plt.plot(snrs, (cre_0_mean + cre_1_mean)/2, 'b.', label='HOUGH Mean')
    plt.plot(snrs, (cre_0_mean + cre_1_mean)/2, 'b--')
    plt.xlabel('SNR [db]')
    plt.ylabel('CRE [%]')
    plt.legend()
    plt.show()
