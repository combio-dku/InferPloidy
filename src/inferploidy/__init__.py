# __init__.py
# Copyright (c) 2021 (syoon@dku.edu) and contributors
# https://github.com/combio-dku/MarkerCount/tree/master
print('https://github.com/combio-dku/InferPloidy')

from .inferploidy import run_infercnv, run_inferploidy, plot_cnv
from .load_data import load_anndata, load_sample_data, load_scoda_processed_sample_data
