# IFAC-VAE-Imputation
Demonstration code for missing data imputation using Variational Autoencoders (VAE)

Copyright 2017, 2018 JT McCoy, RS Kroon and L Auret

Paper presented at IFAC MMM2018 (http://ifac-mmm.csu.edu.cn/), paper link https://doi.org/10.1016/j.ifacol.2018.09.406.

## Overview
This code demonstrates missing data imputation using a VAE for two data sets:
1. A synthetic nonlinear system, adapted from the 2D system in https://github.com/oduerr/dl_tutorial/blob/master/tensorflow/vae/vae_demo-2D.ipynb
2. A simulated milling circuit, described in Wakefield et al., 2018 (https://doi.org/10.1016/j.mineng.2018.02.007). The simulation (in MATLAB and Simulink) is available at https://github.com/ProcessMonitoringStellenboschUniversity/ME-milling-circuit

Imputation on each dataset can be performed for two levels of corruption: light corruption, representing approximately 20% of records with a single missing value, or heavy corruption, with <10% of records complete, approximately 80% of records one or two missing values, and approximately 10% of records three or four missing values.

The VAE is trained on complete records, and imputation performed by Markov chain Monte Carlo as described in Rezende et al., 2014 (https://arxiv.org/abs/1401.4082). VAE imputation is compared to imputation by mean replacement.

## Running the code
Download all files into a directory, and run the main.py file for a default implementation. Adjust the hyperparameters, specified in the main.py file, to change the data files, VAE network dimensions and training parameters and imputation iterations.

The VAE is initialised using the class defined in autoencoders.py, which is based on the implementations by https://github.com/twolffpiggott and https://github.com/jmetzen, then trained on complete records. Missing values are then imputed, and some visualisations of the training progress (in terms of ELBO value), imputation iterations and samples from the posterior and prior distributions plotted.
