import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def _yopt_lower(r_s, r_c, lam_x, lam_y):
    yopt = np.minimum((lam_x*(2**r_s-1)+np.log(lam_y/lam_x))/(lam_x-lam_y), 2**r_s-2**(r_s+r_c))
    _idx = np.where(lam_x <= lam_y)
    yopt[_idx] = 2**r_s-2**(r_s+r_c) 
    return yopt

def g1(y, r_s, r_c, lam_x, lam_y):
    return np.exp(lam_y*y) - np.exp(-lam_x*(2**r_s-1-y))

def g2(y, r_s, r_c, lam_x, lam_y):
    return np.exp(lam_y*y) - np.exp(-lam_x*(2**(r_s+r_c)-1))

def lower_bound_no_csit(r_s, r_c, lam_x, lam_y):
    yopt = _yopt_lower(r_s, r_c, lam_x, lam_y)
    _g1 = g1(yopt, r_s, r_c, lam_x, lam_y)
    _g2 = g2(0, r_s, r_c, lam_x, lam_y)
    return np.maximum(_g1, _g2)


def _yopt_upper(r_s, r_c, lam_x, lam_y):
    yopt = np.minimum((lam_x*(2**r_s-1)+np.log(lam_y/lam_x))/(lam_x-lam_y), 2**r_s-2**(r_s+r_c))
    return yopt

def h1(y, r_s, r_c, lam_x, lam_y):
    return 1 - np.exp(lam_x*(y-(2**r_s-1))) + np.exp(lam_y*y)

def h2(y, r_s, r_c, lam_x, lam_y):
    return 1 - np.exp(-lam_x*(2**(r_s+r_c)-1)) + np.exp(lam_y*y)

def upper_bound_no_csit(r_s, r_c, lam_x, lam_y):
    yopt = _yopt_upper(r_s, r_c, lam_x, lam_y)
    _h1 = h1(yopt, r_s, r_c, lam_x, lam_y)
    _h2 = h2(2**r_s - 2**(r_s+r_c), r_s, r_c, lam_x, lam_y)
    return np.minimum(np.minimum(_h1, _h2), 1)

def independent_no_csit(r_s, r_c, lam_x, lam_y):
    def fx(x, lam):
        return 1. - np.exp(-lam*x)
    alpha = 2**r_s-2**(r_s+r_c)
    int1 = fx(2**(r_s+r_c)-1, lam_x)
    int21 = np.exp(lam_y*alpha) - (lam_y*np.exp(alpha*(lam_x+lam_y)-lam_x*(2**r_s-1)))/(lam_x+lam_y)
    int22 = fx(2**(r_s+r_c)-1, lam_x)*np.exp(lam_y*alpha)
    return int1 + int21 - int22

def export_results(results, **kwargs):
    filename = "secrecy_outage_no_csit-eve_{snr_eve_db:.1f}-rs_{r_s}-rc_{r_c}-lx_{lam_x}-ly_{lam_y}.dat".format(**kwargs)
    data = pd.DataFrame.from_dict(results)
    data.to_csv(filename, sep="\t", index=False)

def main(r_s, r_c, lam_x, lam_y, snr_eve_db):
    snr_db = np.arange(-5, 16)
    snr_bob = 10**(snr_db/10)
    snr_eve = 10**(snr_eve_db/10)
    lam_xt = lam_x/snr_bob
    lam_yt = lam_y/(snr_eve*2**r_s)
    lower = lower_bound_no_csit(r_s, r_c, lam_xt, lam_yt)
    upper = upper_bound_no_csit(r_s, r_c, lam_xt, lam_yt)
    indep = independent_no_csit(r_s, r_c, lam_xt, lam_yt)
    results = {"snr": snr_db, "upper": upper, "lower": lower, "indep": indep}
    export_results(results, snr_eve_db=snr_eve_db, lam_x=lam_x, lam_y=lam_y,
                   r_c=r_c, r_s=r_s)
    plt.semilogy(snr_db, lower)
    plt.semilogy(snr_db, upper)
    plt.semilogy(snr_db, indep)
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", dest="r_s", default=0.1, type=float)
    parser.add_argument("-c", dest="r_c", default=0.5, type=float)
    parser.add_argument("-x", dest="lam_x", default=1, type=float)
    parser.add_argument("-y", dest="lam_y", default=2, type=float)
    parser.add_argument("-e", dest="snr_eve_db", type=float, default=0)
    params = vars(parser.parse_args())
    main(**params)
