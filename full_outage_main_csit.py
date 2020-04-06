import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def g1(x, r_s, r_c, lam_x, lam_y):
    return np.exp(lam_y*(2**r_s-1-x)) - np.exp(-lam_x*x)

def g2(x, r_s, r_c, lam_x, lam_y):
    return np.exp(lam_y*(2**r_s-2**(r_s+r_c))) - np.exp(-lam_x*x)

def lower_bound_main_csit_full(r_s, r_c, lam_x, lam_y):
    xopt = np.minimum((lam_y*(2**r_s-1)+np.log(lam_y/lam_x))/(lam_y-lam_x), 2**(r_s+r_c)-1)
    _g1 = g1(xopt, r_s, r_c, lam_x, lam_y)
    _g2 = np.exp(lam_y*(2**r_s - 2**(r_s+r_c)))
    return np.maximum(_g1, _g2)

def upper_bound_main_csit_full(r_s, r_c, lam_x, lam_y):
    xopt = np.minimum((lam_y*(2**r_s-1)+np.log(lam_y/lam_x))/(lam_y-lam_x), 2**(r_s+r_c)-1)
    upper = 1. - np.exp(-lam_x*xopt) + np.exp(lam_y*(2**r_s-1-xopt))
    return np.minimum(upper, 1)

def independent_main_csit_full(r_s, r_c, lam_x, lam_y):
    s = 2**(r_s)-1
    t = 2**(r_s+r_c)-1
    _part1 = 1. - np.exp(-lam_x*s)
    _part2 = -(lam_x*np.exp(-lam_x*s-(lam_x+lam_y)*t)*(np.exp((lam_x+lam_y)*s)-np.exp((lam_x+lam_y)*t)))/(lam_x+lam_y)
    _part3 = np.exp(lam_y*(s-t))*np.exp(-lam_x*t)
    return _part1 + _part2 + _part3

def export_results(results, **kwargs):
    filename = "full_secrecy_outage_main_csit-eve_{snr_eve_db:.1f}-rs_{r_s}-rc_{r_c}-lx_{lam_x}-ly_{lam_y}.dat".format(**kwargs)
    data = pd.DataFrame.from_dict(results)
    data.to_csv(filename, sep="\t", index=False)

def main(r_s, r_c, lam_x, lam_y, snr_eve_db, export=False):
    snr_db = np.linspace(-5, 15)  #np.arange(-5, 16)
    snr_bob = 10**(snr_db/10)
    snr_eve = 10**(snr_eve_db/10)
    lam_xt = lam_x/snr_bob
    lam_yt = lam_y/(snr_eve*2**r_s)
    lower = lower_bound_main_csit_full(r_s, r_c, lam_xt, lam_yt)
    upper = upper_bound_main_csit_full(r_s, r_c, lam_xt, lam_yt)
    indep = independent_main_csit_full(r_s, r_c, lam_xt, lam_yt)
    if export:
        results = {"snr": snr_db, "upper": upper, "lower": lower, "indep": indep}
        export_results(results, snr_eve_db=snr_eve_db, lam_x=lam_x, lam_y=lam_y,
                       r_c=r_c, r_s=r_s)
    plt.semilogy(snr_db, lower, label="Lower Bound")
    plt.semilogy(snr_db, upper, label="Upper Bound")
    plt.semilogy(snr_db, indep, label="Independent")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", dest="r_s", default=0.1, type=float)
    parser.add_argument("-c", dest="r_c", default=1.0, type=float)
    parser.add_argument("-x", dest="lam_x", default=1, type=float)
    parser.add_argument("-y", dest="lam_y", default=1, type=float)
    parser.add_argument("-e", dest="snr_eve_db", type=float, default=0)
    parser.add_argument("--export", action="store_true")
    params = vars(parser.parse_args())
    main(**params)
