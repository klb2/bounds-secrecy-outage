"""Calculations of the alternative bounds for dependent Rayleigh fading
channels with perfect main CSIT.

This module contains different functions to calculate the bounds on the secrecy
outage probability according to the alternative definition for dependent
Rayleigh fading channels with perfect main CSIT.


Copyright (C) 2020 Karl-Ludwig Besser

This program is used in the article:
Karl-Ludwig Besser and Eduard Jorswieck, "Bounds on the Secrecy Outage
Probability for Dependent Fading Channels", IEEE Transactions on
Communications, vol. 69, no. 1, pp. 443-456, Jan. 2021

License:
This program is licensed under the GPLv3 license. If you in any way use this
code for research that results in publications, please cite our original
article listed above.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.
See the GNU General Public License for more details.

Author: Karl-Ludwig Besser, Technische Universität Braunschweig
"""

import numpy as np
import matplotlib.pyplot as plt

def g1(x, r_s, r_c, lam_x, lam_y):
    return np.minimum(np.exp(lam_y*(2**r_s-1-x)), 1) - np.exp(-lam_x*x)

def g2(x, r_s, r_c, lam_x, lam_y):
    return np.exp(lam_y*(2**r_s-2**(r_s+r_c))) - np.exp(-lam_x*x)

def lower_bound_main_csit_full(r_s, r_c, lam_x, lam_y):
    s = 2**r_s - 1
    t = 2**(r_s+r_c)-1
    xopt = np.clip((lam_y*s+np.log(lam_y/lam_x))/(lam_y-lam_x), s, t)
    _g0 = 1. - np.exp(-lam_x*s)
    _g1 = g1(xopt, r_s, r_c, lam_x, lam_y)
    _g2 = np.exp(lam_y*(s-t))
    return np.maximum(np.maximum(_g1, _g2), _g0)

def upper_bound_main_csit_full(r_s, r_c, lam_x, lam_y):
    s = 2**r_s - 1
    t = 2**(r_s+r_c)-1
    xopt = np.clip((lam_y*s+np.log(lam_y/lam_x))/(lam_y-lam_x), s, t)
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
    import pandas as pd
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
