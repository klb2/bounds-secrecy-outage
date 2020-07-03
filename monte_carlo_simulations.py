import functools

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from bounds_main_csit import lower_bound_main_csit, upper_bound_main_csit, independent_main_csit

def monte_carlo(func):
    @functools.wraps(func)
    def wrapper_monte_carlo(r_s, lam_x, lam_y, snr_bob, snr_eve, num_samples):
        outages = []
        lam_yt = lam_y/(snr_eve*2**r_s)
        for _snr_bob in snr_bob:
            lam_xt = lam_x/_snr_bob
            u1, u2 = func(r_s, lam_xt, lam_yt, num_samples)
            yt = inv_cdf_yt(u2, lam=lam_yt)
            xt = inv_cdf_xt(u1, lam=lam_xt)
            x = xt/_snr_bob
            y = -yt/(2**r_s*snr_eve)
            cs = secrecy_capacity(x, y, _snr_bob, snr_eve)
            _eps = np.count_nonzero(cs < r_s)/len(cs)
            outages.append(_eps)
        return outages
    return wrapper_monte_carlo

def secrecy_capacity(x, y, snr_x, snr_y):
    cap_bob = np.log2(1 + snr_x*x)
    cap_eve = np.log2(1 + snr_y*y)
    return np.maximum(cap_bob - cap_eve, 0)

def sample_copula_lower_main_csit(r_s=1, lam_xt=1, lam_yt=1, num_samples=1000):
    t = lower_bound_main_csit(r_s, 1, lam_xt, lam_yt)
    u1 = np.random.rand(num_samples)
    u2 = np.copy(u1)
    idx_counter = np.where(u1 > t)
    u2[idx_counter] = 1.-u1[idx_counter]+t
    return u1, u2

def sample_copula_upper_main_csit(r_s=1, lam_xt=1, lam_yt=1, num_samples=1000):
    t = upper_bound_main_csit(r_s, 1, lam_xt, lam_yt)
    u1 = np.random.rand(num_samples)
    u2 = np.copy(u1)
    idx_counter = np.where(u1 < t)
    u2[idx_counter] = t-u1[idx_counter]
    return u1, u2

def inv_cdf_xt(u, lam=1):
    return -np.log(1-u)/lam

def inv_cdf_yt(u, lam=1):
    return np.log(u)/lam

def main(r_s, r_c, lam_x, lam_y, snr_eve_db, num_samples=1000):
    snr_db = np.arange(-5, 16, .5)
    snr_bob = 10**(snr_db/10)
    snr_eve = 10**(snr_eve_db/10)
    lam_xt = lam_x/snr_bob
    lam_yt = lam_y/(snr_eve*2**r_s)
    # Analytical Bounds
    lower = lower_bound_main_csit(r_s, r_c, lam_xt, lam_yt)
    upper = upper_bound_main_csit(r_s, r_c, lam_xt, lam_yt)
    indep = independent_main_csit(r_s, r_c, lam_xt, lam_yt)
    #results = {"snr": snr_db, "upper": upper, "lower": lower, "indep": indep}
    #export_results(results, filename=filename)
    plt.semilogy(snr_db, lower)
    plt.semilogy(snr_db, upper)
    plt.semilogy(snr_db, indep)

    monte_carlo_outages = {}
    monte_carlo_outages["lower"] = monte_carlo_lower_bound(r_s, lam_x, lam_y,
                                                           snr_bob, snr_eve,
                                                           num_samples)
    monte_carlo_outages["upper"] = monte_carlo_upper_bound(r_s, lam_x, lam_y,
                                                           snr_bob, snr_eve,
                                                           num_samples)
    monte_carlo_outages["indep"] = monte_carlo_indep(r_s, lam_x, lam_y,
                                                     snr_bob, snr_eve,
                                                     num_samples)
    plt.semilogy(snr_db, monte_carlo_outages["lower"], 'o', label="MC Lower")
    plt.semilogy(snr_db, monte_carlo_outages["upper"], 'o', label="MC Upper")
    plt.semilogy(snr_db, monte_carlo_outages["indep"], 'o', label="MC Indep")
    plt.legend()

monte_carlo_lower_bound = monte_carlo(sample_copula_lower_main_csit)
monte_carlo_upper_bound = monte_carlo(sample_copula_upper_main_csit)

@monte_carlo
def monte_carlo_indep(r_s=1, lam_xt=1, lam_yt=1, num_samples=1000):
    u1 = np.random.rand(num_samples)
    u2 = np.random.rand(num_samples)
    return u1, u2

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", dest="r_s", default=0.1, type=float)
    parser.add_argument("-c", dest="r_c", default=0.5, type=float)
    parser.add_argument("-x", dest="lam_x", default=1, type=float)
    parser.add_argument("-y", dest="lam_y", default=1, type=float)
    parser.add_argument("-e", dest="snr_eve_db", type=float, default=0)
    parser.add_argument("-n", dest="num_samples", type=int, default=10000)
    params = vars(parser.parse_args())
    main(**params)
    plt.show()
