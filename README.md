# Bounds on the Secrecy Outage Probability

This repository is part of the publication "Bounds on the Secrecy Outage
Probability for Dependent Fading Channels" (Karl-Ludwig Besser and Eduard
Jorswieck, Submitted to IEEE Transactions on Information Forensics and
Security, 2020, [arXiv:XXX](https://arxiv.org/abs/XXX)).

The idea is to give an interactive version of the calculations to the reader
such that one can easily reproduce the plots presented in the paper as well as
changing parameters. One can also use this framework as a baseline and adjust
it to their own needs, e.g., for different channel models.

## File List
The following files are provided in this repository:

* [Bounds Secrecy Outage Probability.ipynb](https://mybinder.org):
  Jupyter notebook containing the results for Rayleigh fading presented in the
  paper.
* `Bounds_Secrecy_Outage.nb`: Mathematica notebook that contains all
  calculations for Rayleigh fading.
* `bounds_main_csit.py` and `bounds_no_csit.py`: Python modules that contain
  the functions for Rayleigh fading for the scenarios of main CSIT and no CSIT.


## Usage
The calculations are provided in a Mathematica notebook and a Python/Jupyter
notebook.  
If you do not have a Jupyter installation, you can also run it online.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org)

### Local Installation
If you want to run the files locally, make sure you have 
[Python3](https://www.python.org/downloads/) installed on your computer.

You can install the requires packages (including Jupyter) by running
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


## Acknowledgements
This research was supported in part by the Deutsche Forschungsgemeinschaft
(DFG) under grant JO 801/23-1.


## License and Referencing
This program is licensed under the GPLv3 license. If you in any way use this
code for research that results in publications, please cite our original
article listed above.
