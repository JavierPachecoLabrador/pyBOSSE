# pyBOSSE

## Synopsis

This project contains experimental *Python* code for the *BOSSE v1.0: the Biodiversity Observing System Simulation Experiment* (pyBOSSE) model using simulating temporally dynamic scenes of species maps, vegeation traits and associated remote sensing imagery for the development of remote sensing methods for the quantification of plant diversity and biodiveristy-ecosyste function relationships. 

pyBOSSE incorporates the following packages: an updated version of **pyGNDiv** code (https://github.com/JavierPachecoLabrador/pyGNDiv-master) computing normalized plant functional diversity metrics from spectral and plant trait variables, **pyet** for the computation of potential evapotranspiration (https://github.com/pyet-org/pyet), and **NLMpy** (https://github.com/tretherington/nlmpy) for the generation of random landscapes.

The BOSSE model is implemented in a class, after the class and the scene to be simulated are initialized, the class can provide simulations of plant trait maps, ecosystem functions and remote sensing signals at any time step of the meteorological time series. This repository contains a *Jupyter* tutorial script (**tutorial_bosse_v1_0.ipynb**) to show how to run pyBOSSE, and the *Python* script (**ManuscriptFigures.py**) reproducing the figures of the associated manuscript.

A reduced set of meteorological data necessary to run pyBOSSE are provide in the folder ./BOSSE_inputs/Meteo_ERA5Land/, a larger dataset useful to run a larger variety of simulations can be found in https://doi.org/10.5281/zenodo.14717038

## Installation
The following Python libraries will be required:
    - python=3.11
    - dask
	- git+https://github.com/JavierPachecoLabrador/pyGNDiv-master
    - h5netcdf
    - joblib
    - jupyter
    - matplotlib
    - netCDF4
    - notebook
    - numpy
	- nlmpy
    - pandas=2.1.1
    - pyet
	- python-rle
    - python-wget
    - scipy
    - scikit-learn=1.3.1
    - scikit-image
    - seaborn
    - regex
    - shapely
    - pickledb
    - pip
    - xarray

With `conda`, you can create a complete environment with
```bash
conda env create -f environment.yml
```

or with pip:
```bash
pip install -r requirements.txt
```

if there were any issues during the installation of the **pyGNDiv** package, try either:
	```bash
	pip install git+https://github.com/JavierPachecoLabrador/pyGNDiv-master
	```
or
	```bash
	git clone https://github.com/JavierPachecoLabrador/pyGNDiv-master.git
	cd pyGNDiv-master/
	python setup.py install
	```

Then you call the BOSSE model by inserting the folder containing code and datasets as follows:
```python
sys.path.insert(0, path_bosse)
from BOSSE.bosse import BosseModel
```

## Basic Contents
### Code /BOSSE
- *bosse.py* script containing the BOSSE Class 
- Ancillary functions of the BOSSE model

### Model inputs /BOSSE_inputs
- /Meteo_ERA5Land/*.nc: NetCDF files containint the ERA5-Land hourly meteorological time series
- /Sensors/: Folder containing the spectral response functions of different sensors

### Model submoldels /BOSSE_models
- /GMMtraits: Gaussian Mixture Model generating correlated random samples of foliar traits
- /Inter_rss: 2D interpolator to predict soil resistance to evaporation for soil pore space as a funciton of soil moisture and an empirical factor
- /NNeF_nlyr: 2-layer neural network predicting ecosystem functions
- /NNF_nlyr: 2-layer neural network predicting hyperspectral fluorescence radiance
- /NNR_nlyr_Hy: 2-layer neural network predicting hyperspectral reflectance factors
- /NNT_nlyr2: 2-layer neural network predicting land surface temperature
- /NNRinv_nlyr2_*: 2-layer neural network retrieving optical traits from the reflectance factors of different sensors
- /PFTdist: Relative abundances of the different plant functional types in each Köppen Climatic Zone

## Main Scientific References
- Pacheco-Labrador, J., Gomarasca, U., Pabon-Moreno, D. E., Li, W., Migliavacca, M., Jung, M., and Duveiller, G. (submitted). BOSSE v1.0: the Biodiversity Observing System Simulation Experiment. Geoscientific Model Developpment.

- Pacheco-Labrador, J., Migliavacca, M., Ma, X., Mahecha, M.D., Carvalhais, N., Weber, U., Benavides, R., Bouriaud, O., Barnoaiea, I., Coomes, D.A., Bohn, F.J., Kraemer, G., Heiden, U., Huth, A., & Wirth, C. (2022). Challenging the link between functional and spectral diversity with radiative transfer modeling and data. Remote Sensing of Environment, 280, 113170. https://doi.org/10.1016/j.rse.2022.113170

- Gomarasca, U., Duveiller, G., Pacheco-Labrador, J., Ceccherini, G., Cescatti, A., Girardello, M., Nelson, J.A., Reichstein, M., Wirth, C., & Migliavacca, M. (2024). Satellite remote sensing reveals the footprint of biodiversity on multiple ecosystem functions across the NEON eddy covariance network. Environmental Research: Ecology, 3, 045003. https://dx.doi.org/10.1088/2752-664X/ad87f9

- Pacheco-Labrador, J., de Bello, F., Migliavacca, M., Ma, X., Carvalhais, N., & Wirth, C. (2023). A generalizable normalization for assessing plant functional diversity metrics across scales from remote sensing. Methods in Ecology and Evolution, 14, 2123-2136. https://doi.org/10.1111/2041-210X.14163

## Contributors
- **Javier Pacheco-Labrador** <javier.pacheco@csic.es> main developer

## License
pyBOSSE: the Biodiversity Observing System Simulation Experiment

Copyright 2025 Javier Pacheco-Labrador.
    
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.  If not, see <http://www.gnu.org/licenses/>.
