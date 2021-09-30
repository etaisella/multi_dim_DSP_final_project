from RANSAC import RANSAC_test
from make_chirps import make_chirps, test_chirps, chirp_cre_test
from real_chirps import test_real_chirps, test_real_chirps_cre
from hough import hough_test, hough_cre_test
import numpy as np
import getopt, sys
import pickle

# list of command line arguments
argumentList = sys.argv[1:]

# Options
options = "ht:"

# Long options
long_options = ["Help", "Test"]

if __name__ == "__main__":
    try:
        # Parsing argument
        arguments, values = getopt.getopt(argumentList, options, long_options)

        # checking each argument
        for currentArgument, currentValue in arguments:

            if currentArgument in ("-h", "--Help"):
                print("Command line arguments:\n"
                      "-h, -help: Print help\n"
                      "-t, -test: Run test for a specific component, possible options:\n"
                      "hough - Conducts hough test\n"
                      "hough_cre - Conducts hough cre test\n"
                      "ransac - Conducts RANSAC test\n"
                      "chirp - Conducts chirp test\n"
                      "chirp_cre - Conducts chirp cre test\n"
                      "recorded_chirp - Conducts recorded chirp test - Example\n"
                      "recorded_chirp_cre - Conducts recorded chirp cre test")

            elif currentArgument in ("-t", "--Test"):
                if currentValue == "hough":
                    hough_test(plot_article_figures=False, CRE=True)
                elif currentValue == "hough_cre":
                    hough_cre_test()
                elif currentValue == "ransac":
                    RANSAC_test()
                elif currentValue == "chirp":
                    # Make new data
                    sigmas = np.linspace(1, 20, 15)
                    data = make_chirps(sigmas=sigmas)
                    test_chirps(data, plot_time_freq_curve=True)
                elif currentValue == "chirp_cre":
                    # Make CRE chirp test
                    chirp_cre_test()
                elif currentValue == "recorded_chirp":
                    test_real_chirps()
                elif currentValue == "recorded_chirp_cre":
                    test_real_chirps_cre()

    except getopt.error as err:
        # output error, and return with an error code
        print(str(err))
