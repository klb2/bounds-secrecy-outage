# Bounds on the Secrecy Outage Probability

This repository is part of the publication "Bounds on the Secrecy Outage
Probability for Dependent Fading Channels" (Karl-Ludwig Besser and Eduard
Jorswieck, Submitted to IEEE Transactions on Communications, 2020,
[arXiv:2004.06644](https://arxiv.org/abs/2004.06644)).

The idea is to give an interactive version of the calculations to the reader
such that one can easily reproduce the plots presented in the paper as well as
changing parameters. One can also use this framework as a baseline and adjust
it to their own needs, e.g., for different channel models.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gl/klb2%2Fbounds-secrecy-outage/master?filepath=Bounds%20Secrecy%20Outage%20Probability.ipynb)

## File List
The following files are provided in this repository:

* [Bounds Secrecy Outage Probability.ipynb](https://mybinder.org/v2/gl/klb2%2Fbounds-secrecy-outage/master?filepath=Bounds%20Secrecy%20Outage%20Probability.ipynb):
  Jupyter notebook containing the results for Rayleigh fading presented in the
  paper.
* `Bounds_Secrecy_Outage.nb`: Mathematica notebook that contains all
  calculations for Rayleigh fading.
* `bounds_main_csit.py` and `bounds_no_csit.py`: Python modules that contain
  the functions for Rayleigh fading for the scenarios of perfect main CSIT and
  only statistical CSIT.
* `monte_carlo_simulations_main_csit.py` and
  `monte_carlo_simulations_no_csit.py`: Python modules that contain the
  functions to estimate the secrecy outage probability using Monte Carlo
  simulations.
* `bounds_secrecy_rate_main_csit.py`: Python module to calculate the eps-outage
  secrecy rate for Rayleigh fading channels with perfect main CSIT.
* `full_outage_main_csit.py` and `full_outage_no_csit.py`: Python modules that
  contain the functions for Rayleigh fading for the alternative, pessimistic
  secrecy outage definition.


## Usage
The calculations are provided in a Mathematica notebook and a Python/Jupyter
notebook.  
If you do not have a Jupyter installation, you can also run it online using a
service like Binder.
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gl/klb2%2Fbounds-secrecy-outage/master?filepath=Bounds%20Secrecy%20Outage%20Probability.ipynb)

### Local Installation
If you want to run the files locally, make sure you have 
[Python3](https://www.python.org/downloads/) installed on your computer.

You can install the required packages (including Jupyter) by running
```
pip3 install -r requirements.txt
jupyter nbextension enable --py widgetsnbextension
```
This will install all the needed packages which are listed in the requirements 
file. The second line enables the interactive controls in the Jupyter
notebooks.

Afterwards, you can start the notebook by running
```
jupyter notebook 'Bounds Secrecy Outage Probability.ipynb'
```

The present code was developed and tested with the following versions:
- Python 3.8
- Jupyter 1.0
- numpy 1.18
- scipy 1.4


## Acknowledgements
This research was supported in part by the Deutsche Forschungsgemeinschaft
(DFG) under grant JO 801/23-1.


## License and Referencing
This program is licensed under the GPLv3 license. If you in any way use this
code for research that results in publications, please cite our original
article listed above.
