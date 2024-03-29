"""Calculations of the bounds for dependent Rayleigh fading channels with
only statistical CSIT.

This module contains different functions to calculate the bounds on the secrecy
outage probability for dependent Rayleigh fading channels with only statistical
CSIT.


Copyright (C) 2020 Karl-Ludwig Besser

This program is used in the article:
Karl-Ludwig Besser and Eduard Jorswieck, "Bounds on the Secrecy Outage
Probability for Dependent Fading Channels", IEEE Transactions on
Communications, vol. 69, no. 1, pp. 443-456, Jan. 2021.

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

def g1(y, r_s, r_c, lam_x, lam_y):
    return np.exp(lam_y*y) - np.exp(-lam_x*(2**r_s-1-y))

def g2(y, r_s, r_c, lam_x, lam_y):
    return np.exp(lam_y*y) - np.exp(-lam_x*(2**(r_s+r_c)-1))

def lower_bound_no_csit(r_s, r_c, lam_x, lam_y):
    yopt = np.minimum((lam_x*(2**r_s-1)+np.log(lam_y/lam_x))/(lam_x-lam_y), 2**r_s-2**(r_s+r_c))
    _g1 = g1(yopt, r_s, r_c, lam_x, lam_y)
    _g2 = 1. - np.exp(-lam_x*(2**(r_s+r_c)-1))
    return np.maximum(_g1, _g2)

def upper_bound_no_csit(r_s, r_c, lam_x, lam_y):
    yopt = np.minimum((lam_x*(2**r_s-1)+np.log(lam_y/lam_x))/(lam_x-lam_y), 2**r_s-2**(r_s+r_c))
    upper = 1. - np.exp(-lam_x*(2**r_s-1-yopt)) + np.exp(lam_y*yopt)
    return np.minimum(upper, 1)

def independent_no_csit(r_s, r_c, lam_x, lam_y):
    s = 2**(r_s)-1
    t = 2**(r_s+r_c)-1
    _part1 = 1. - np.exp(-lam_x*t)
    _part2 = (lam_x*np.exp(lam_y*(s-t)-lam_x*t))/(lam_x+lam_y)
    return _part1 + _part2

def export_results(results, **kwargs):
    import pandas as pd
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
    parser.add_argument("-c", dest="r_c", default=1.0, type=float)
    parser.add_argument("-x", dest="lam_x", default=1, type=float)
    parser.add_argument("-y", dest="lam_y", default=1, type=float)
    parser.add_argument("-e", dest="snr_eve_db", type=float, default=0)
    params = vars(parser.parse_args())
    main(**params)
