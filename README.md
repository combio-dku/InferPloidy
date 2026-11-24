# InferPloidy

![PyPI](https://img.shields.io/pypi/v/inferploidy) [![Downloads](https://pepy.tech/badge/hicat)](https://pepy.tech/project/inferploidy) [![Downloads/Month](https://pepy.tech/badge/hicat/month)](https://pepy.tech/project/inferploidy)


## Brief introduction
- InferPloidy is a CNV-based, ploidy annotation tool for single-cell RNA-seq data.
- It works with the CNV-estimates obtained from [infercnv](https://github.com/broadinstitute/inferCNV/wiki), or [infercnvpy](https://github.com/icbi-lab/infercnvpy) .

## Cite InferPloidy
- "InferPloidy enables CNV guided tumor cell stratification and biomarker discovery from single cell RNA sequencing", [Communications Biology Nov 2026](https://doi.org/10.1038/s42003-026-11298-2) 

<div align="center">
  <img src="https://github.com/combio-dku/inferploidy/blob/main/images/inferploidy.png" style="width:80%;"/>
</div>

## Installation using pip, importing inferploidy in Python

InferPloidy can be installed using pip command. With python3 installed in your system, simply use the follwing command in a terminal.

`pip install inferploidy`

Once it is installed using pip, you can import two functions using the following python command.

`from inferploidy import run_infercnv, run_inferploidy`

## Example usage in Jupyter notebook

[`inferPloidy_example.ipynb`](https://colab.research.google.com/github/combio-dku/InferPloidy/blob/main/notebooks/inferploidy_example.ipynb) is example code in Jupyter notebook, where you can see how to import and run InferPloidy. 

To run HiCAT, you need the pre-installed python packages `numpy`, `pandas`, `scikit-learn`, `scipy`, `scikit-network`, `infercnvpy` and , `hicat`.
`hicat` is used to annotate cell-type to collect reference cells for infercnv.
All of them can be installed simply using `pip` command.

## Contact
Send email to syoon@dku.edu for any inquiry on the usages.

