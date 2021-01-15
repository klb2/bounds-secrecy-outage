"""Calculations of the bounds on the eps-outage secrecy rate for dependent
Rayleigh fading channels with perfect main CSIT.

This module contains different functions to calculate the bounds on the
eps-outage secrecy rate for dependent Rayleigh fading channels with perfect
main CSIT.


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
from scipy import optimize
import matplotlib.pyplot as plt

from bounds_main_csit import (lower_bound_main_csit, upper_bound_main_csit,
                              independent_main_csit, export_results)

def limit_eps_rs0(lam_x, lam_y, snr_bob, snr_eve, function="lower"):
    _lam_xt = lam_x/snr_bob
    _lam_yt = lam_y/snr_eve
    if function == "indep":
        return _lam_xt/(_lam_xt+_lam_yt)
    elif function == "lower":
        if _lam_yt < _lam_xt:
            return np.exp(_lam_yt*np.log(_lam_yt/_lam_xt)/(_lam_xt-_lam_yt)) - np.exp(_lam_xt*np.log(_lam_yt/_lam_xt)/(_lam_xt-_lam_yt))
        else:
            return 0.
    elif function == "upper":
        if _lam_xt >= _lam_yt:
            return 1.
        else:
            return 1.+np.exp(_lam_yt*np.log(_lam_yt/_lam_xt)/(_lam_xt-_lam_yt)) - np.exp(_lam_xt*np.log(_lam_yt/_lam_xt)/(_lam_xt-_lam_yt))


def find_rate_to_eps(eps_target, r_c, lam_x, lam_y, snr_bob, snr_eve, function="lower"):
    _limit = limit_eps_rs0(lam_x, lam_y, snr_bob, snr_eve, function)
    if eps_target < _limit:
        return 0.
    if function == "lower":
        function = lower_bound_main_csit
    elif function == "indep":
        function = independent_main_csit
    elif function == "upper":
        function = upper_bound_main_csit
    lam_xt = lam_x/snr_bob
    sol = optimize.root_scalar(
            lambda r_s: function(r_s, r_c, lam_xt, lam_y/(snr_eve*2**r_s))-eps_target,
            bracket=(0, 10))
    #print(sol)
    return sol.root

def main(r_c, lam_x, lam_y, snr_db, snr_eve_db):
    snr_bob = 10**(snr_db/10)
    snr_eve = 10**(snr_eve_db/10)
    #eps = np.linspace(0.1, .8, 10)
    eps = np.logspace(-4, 0, 250, endpoint=False)
    names = ["lower", "indep", "upper"]
    rate = {_name: [find_rate_to_eps(_eps, r_c, lam_x, lam_y, snr_bob, snr_eve,
                                     function=_name) for _eps in eps]
            for _name in names}
    fig, ax = plt.subplots()
    for _name, _rates in rate.items():
        ax.loglog(eps, _rates, label=_name)
    ax.legend()
    filename = "eps_outage_sec_rates-main_csit-lx{}-ly{}-snrx{}-snry{}.dat".format(lam_x, lam_y, snr_db, snr_eve_db)
    rate['eps'] = eps
    export_results(rate, filename)
    print(rate)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", dest="r_c", default=0.5, type=float)
    parser.add_argument("-x", dest="lam_x", default=1, type=float)
    parser.add_argument("-y", dest="lam_y", default=1, type=float)
    parser.add_argument("-b", dest="snr_db", type=float, default=5)
    parser.add_argument("-e", dest="snr_eve_db", type=float, default=0)
    params = vars(parser.parse_args())
    main(**params)
    plt.show()
