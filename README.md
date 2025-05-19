# InferPloidy
Ploidy inference from CNV estimates

## Brief introduction
- InferPloidy is a CNV-based, ploidy annotation tool for single-cell RNA-seq data.
- It works with the CNV-estimates obtained from [infercnv](https://github.com/icbi-lab/infercnvpy) .

## Cite HiCAT
- "InferPloidy: A fast ploidy inference tool accurately classifies cells with abnormal CNVs in large single-cell RNA-seq datasets", available at [bioRxiv](https://doi.org/10.1101/2025.03.13.643178) 

## Installation using pip, importing HiCAT in Python

InferPloidy can be installed using pip command. With python3 installed in your system, simply use the follwing command in a terminal.

`pip install inferploidy`

Once it is installed using pip, you can import two functions using the following python command.

`from inferploidy import run_infercnv, run_inferploidy`

## Example usage in Jupyter notebook

`HiCAT_example_py_v02.ipynb` is example code of HiCAT in Jupyter notebook, where you can see how to import and run HiCAT. For quick overveiw of the usage of HiCAT, simply click `HiCAT_example_py_v02.ipynb` above in the file list.

To run the example, download the Jupyter notebook file, maker DB in `.tsv` file and a sample single-cell RNA-Seq data with `.h5ad` extension (It is one of the data we used in our paper mentioned above). Just follow the instruction below.

1. Download all the files in ZIP format.
2. Decompress the files into a desired folder.
3. Decompress 'Melanoma_5K_rev.h5ad.zip'
4. Run jupyter notebook and open the jupyter notebook file `HiCAT_example_py_v02.ipynb`
5. You can run the codes step-by-step and can see the intermediate and final results.

To run HiCAT, you need the pre-installed python packages `Numpy`, `Pandas`, `sklearn`, `scipy`, and `scikit-network`.
`seaborn` and `matplotlib` are required only to show the results, not for the HiCAT itself.
All of them can be installed simply using `pip` command.

## Using HiCAT in R

(Installed using pip) You also can import and use HiCAT in R, for which you need the R package `reticulate`.
First, import HiCAT using the following command

`library(reticulate)`  
`mkrcnt <- import("MarkerCount.hicat")`

Then, you can call the HiCAT functions as follows.

`df_res <- mkrcnt$HiCAT( .. arguments .. )` 

The arguments to pass and the return value are the same as those in python.
R example of HiCAT is in R script `HiCAT_example.R`
Tested in linux Mint with R version 4.0.5. (numpy v1.26.4, pandas v2.2.1, scipy v1.12.0, scikit-network v0.33.1)

## Contact
Send email to syoon@dku.edu for any inquiry on the usages.

