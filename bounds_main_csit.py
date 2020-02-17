import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def _yopt_lower(r_s, lam_x, lam_y):
    yopt = np.minimum((lam_x*(2**r_s-1)+np.log(lam_y/lam_x))/(lam_x-lam_y), 0)
    _idx = np.where(lam_x <= lam_y)
    yopt[_idx] = 0
    return yopt

def g(y, r_s, lam_x, lam_y):
    return np.exp(lam_y*y) - np.exp(lam_x*(y-(2**r_s-1)))

def lower_bound_main_csit(r_s, r_c, lam_x, lam_y):
    yopt = _yopt_lower(r_s, lam_x, lam_y)
    print(yopt)
    return g(yopt, r_s, lam_x, lam_y)


def _yopt_upper(r_s, lam_x, lam_y):
    yopt = (lam_x*(2**r_s-1)+np.log(lam_y/lam_x))/(lam_x-lam_y)
    return yopt

def h(y, r_s, lam_x, lam_y):
    return 1 - np.exp(lam_x*(y-(2**r_s-1))) + np.exp(lam_y*y)

def upper_bound_main_csit(r_s, r_c, lam_x, lam_y):
    yopt = _yopt_upper(r_s, lam_x, lam_y)
    return np.minimum(h(yopt, r_s, lam_x, lam_y), 1)

def independent_main_csit(r_s, r_c, lam_x, lam_y):
    return 1.-(lam_y*np.exp(-lam_x*(2**r_s-1)))/(lam_x+lam_y)

def export_results(results, filename):
    data = pd.DataFrame.from_dict(results)
    data.to_csv(filename, sep="\t", index=False)

def main(r_s, r_c, lam_x, lam_y, snr_eve_db, variable="snr"):
    if variable == "snr":
        snr_db = np.arange(-5, 16, .5)
        xvar = snr_db
        filename = f"secrecy_outage_main_csit-eve_{snr_eve_db:.1f}-rs_{r_s}-lx_{lam_x}-ly_{lam_y}.dat"
    elif variable == "r_s":
        snr_db = 5
        r_s = np.logspace(-3, 1, 30)
        xvar = r_s
        filename = f"secrecy_outage_main_csit-eve_{snr_eve_db:.1f}-bob_{snr_db:.1f}-lx_{lam_x}-ly_{lam_y}.dat"
    elif variable == "snr_eve":
        snr_db = 15
        snr_eve_db = np.arange(-30, 21, .5)
        xvar = snr_eve_db
        filename = f"secrecy_outage_main_csit-bob_{snr_db:.1f}-rs_{r_s}-lx_{lam_x}-ly_{lam_y}.dat"
    snr_bob = 10**(snr_db/10)
    snr_eve = 10**(snr_eve_db/10)
    lam_xt = lam_x/snr_bob
    lam_yt = lam_y/(snr_eve*2**r_s)
    print(lam_xt>lam_yt)
    lower = lower_bound_main_csit(r_s, r_c, lam_xt, lam_yt)
    upper = upper_bound_main_csit(r_s, r_c, lam_xt, lam_yt)
    indep = independent_main_csit(r_s, r_c, lam_xt, lam_yt)
    results = {variable: xvar, "upper": upper, "lower": lower, "indep": indep}
    export_results(results, filename=filename)
    plt.semilogy(xvar, lower)
    plt.semilogy(xvar, upper)
    plt.semilogy(xvar, indep)
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", dest="r_s", default=0.1, type=float)
    parser.add_argument("-c", dest="r_c", default=0.5, type=float)
    parser.add_argument("-x", dest="lam_x", default=1, type=float)
    parser.add_argument("-y", dest="lam_y", default=1, type=float)
    parser.add_argument("-e", dest="snr_eve_db", type=float, default=0)
    parser.add_argument("--variable", default="snr", type=str)
    params = vars(parser.parse_args())
    main(**params)
