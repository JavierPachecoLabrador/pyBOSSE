import os
import sys
import pandas as pd
import numpy as np

# Set paths
parent_foler = os.path.abspath(
    os.path.join(os.path.abspath(
        os.path.join(os.getcwd(), os.pardir)), os.pardir))
# Defione as global the pyBOSSE paths
path_bosse = (parent_foler + '//pyBOSSE')
path_inputs= path_bosse + '//BOSSE_inputs//'
path_models= path_bosse + '//BOSSE_models//'

import netCDF4 as nc
sys.path.insert(0, path_bosse)
from BOSSE.bosse import BosseModel
from BOSSE.helpers import (set_up_paths_and_inputs, benchmark_bosse_speed)


# %% Main
output_folder = parent_foler + '//tutorial_bosse_v1_0_figures//'

(inputs_, paths_) = set_up_paths_and_inputs(None, output_folder,
                                            create_out_folder=False,
                                            pth_root=path_bosse,
                                            pth_inputs=path_inputs,
                                            pth_models=path_models)

# Run the tests
# Number of different Scenes per climatic zone and spatial pattern
n_samples = 1
# Number of different dates simulated per Scene
n_days = 12

df_ = benchmark_bosse_speed(BosseModel, inputs_, paths_,
                            out_fname=output_folder + 'SpeedTest',
                            n_samples=n_samples, n_days=n_days)

df_.to_csv(parent_foler + f'//BOSSE_BenchmarkTest_ns{n_samples}_nd{n_days}.csv',
           sep=';')