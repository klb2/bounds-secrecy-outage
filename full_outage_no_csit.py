"""Calculations of the alternative bounds for dependent Rayleigh fading
channels with statistical CSIT.

This module contains different functions to calculate the bounds on the secrecy
outage probability according to the alternative definition for dependent
Rayleigh fading channels with only statistical CSIT.


Copyright (C) 2020 Karl-Ludwig Besser

This program is used in the article:
Karl-Ludwig Besser and Eduard Jorswieck, "Bounds on the Secrecy Outage
Probability for Dependent Fading Channels", IEEE Transactions on
Communications, 2020.

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

def cdf_xt(xt, lam_xt):
    return np.maximum(1-np.exp(-lam_xt*xt), 0)

def cdf_yt(yt, lam_yt):
    return np.minimum(np.exp(lam_yt*yt), 1)

def w_copula(a, b):
    return np.maximum(a+b-1, 0)

def m_copula(a, b):
    return np.minimum(a, b)

def prod_copula(a, b):
    return a*b

def dual_copula(a, b, copula):
    return a + b - copula(a, b)


def outage_probability(r_s, r_c, lam_x, lam_y, copula):
    copula = copula.lower()
    if copula.startswith("low"):
        copula = m_copula
    elif copula.startswith("up"):
        copula = w_copula
    elif copula.startswith("prod") or copula.startswith("ind"):
        copula = prod_copula
    s = 2**r_s - 1.
    t = 2**(r_s+r_c) - 1.
    fxt = cdf_xt(t, lam_x)
    fyt = cdf_yt(s-t, lam_y)
    return dual_copula(fxt, fyt, copula)

def lower_bound_no_csit_full(*args, **kwargs):
    return outage_probability(*args, **kwargs, copula="lower")

def upper_bound_no_csit_full(*args, **kwargs):
    return outage_probability(*args, **kwargs, copula="upper")

def independent_no_csit_full(*args, **kwargs):
    return outage_probability(*args, **kwargs, copula="indep")


def export_results(results, **kwargs):
    import pandas as pd
    filename = "full_secrecy_outage_no_csit-eve_{snr_eve_db:.1f}-rs_{r_s}-rc_{r_c}-lx_{lam_x}-ly_{lam_y}.dat".format(**kwargs)
    data = pd.DataFrame.from_dict(results)
    data.to_csv(filename, sep="\t", index=False)

def main(r_s, r_c, lam_x, lam_y, snr_eve_db, export=False):
    snr_db = np.linspace(-5, 15)  #np.arange(-5, 16)
    snr_bob = 10**(snr_db/10)
    snr_eve = 10**(snr_eve_db/10)
    lam_xt = lam_x/snr_bob
    lam_yt = lam_y/(snr_eve*2**r_s)
    lower = outage_probability(r_s, r_c, lam_xt, lam_yt, copula="lower")
    upper = outage_probability(r_s, r_c, lam_xt, lam_yt, copula="upper")
    indep = outage_probability(r_s, r_c, lam_xt, lam_yt, copula="indep")
    s = 2**r_s - 1.
    t = 2**(r_s+r_c) - 1.
    fxt = cdf_xt(t, lam_xt)
    fyt = cdf_yt(s-t, lam_yt)*np.ones_like(snr_db)
    if export:
        results = {"snr": snr_db, "upper": upper, "lower": lower,
                   "indep": indep, "event2": fxt, "event3": fyt}
        export_results(results, snr_eve_db=snr_eve_db, lam_x=lam_x, lam_y=lam_y,
                   r_c=r_c, r_s=r_s)
    plt.semilogy(snr_db, lower, label="Lower Bound")
    plt.semilogy(snr_db, upper, label="Upper Bound")
    plt.semilogy(snr_db, indep, label="Independent")
    plt.semilogy(snr_db, fxt, label="Event 2")
    plt.semilogy(snr_db, fyt, label="Event 3")
    plt.xlabel("SNR Bob [dB]")
    plt.ylabel("Outage Probability")
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
