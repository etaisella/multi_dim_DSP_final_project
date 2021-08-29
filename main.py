from RANSAC import RANSAC_test
from make_chirps import make_chirps, test_chirps
from real_chirps import test_real_chirps
from hough import hough_test
import numpy as np
import getopt, sys

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
                      "ransac - Conducts RANSAC test\n"
                      "chirp - Conducts chirp test\n"
                      "recorded_chirp - Conducts recrded chirp test")

            elif currentArgument in ("-t", "--Test"):
                if currentValue == "hough":
                    hough_test()
                elif currentValue == "ransac":
                    RANSAC_test()
                elif currentValue == "chirp":
                    sigmas = np.arange(2, 20)
                    data = make_chirps(amp=1, mu=0, sigmas=sigmas)
                    test_chirps(data, graph=False, CRE=True, plot_time_freq_curve=False)
                elif currentValue == "recorded_chirp"
                    test_real_chirps()

    except getopt.error as err:
        # output error, and return with an error code
        print(str(err))