"""
Main file to run VAE for missing data imputation.
Presented at IFAC MMM2018 by JT McCoy, RS Kroon and L Auret.

Based on implementations
of VAEs from:
    https://github.com/twolffpiggott/autoencoders
    https://jmetzen.github.io/2015-11-27/vae.html
    https://github.com/lazyprogrammer/machine_learning_examples/blob/master/unsupervised_class3/vae_tf.py
    https://github.com/deep-learning-indaba/practicals2017/blob/master/practical5.ipynb

VAE is designed to handle real-valued data, not binary data, so the source code
has been adapted to work only with Gaussians as the output of the generative
model (p(x|z)).
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from autoencoders import TFVariationalAutoencoder
import pandas as pd
import random

'''
==============================================================================
'''
# DEFINE HYPERPARAMETERS

# select data source:
# can be "mill" or "X"
data_source = "X"

# select corruption level:
# can be "light" or "heavy"
corr_level = "light"

# Path to uncorrupted data:
DataPath = data_source + "data.csv"
# Path to corrupted data:
CorruptDataPath = data_source + "datacorrupt" + corr_level + ".csv"

# VAE network size:
Decoder_hidden1 = 20
Decoder_hidden2 = 20
Encoder_hidden1 = 20
Encoder_hidden2 = 20

# dimensionality of latent space:
latent_size = 5

# training parameters:
training_epochs = 500
batch_size = 250
learning_rate = 0.001

# specify number of imputation iterations:
ImputeIter = 25
'''
==============================================================================
'''
# LOAD DATA
# Load data from a csv for analysis:
Xdata_df = pd.read_csv(DataPath)
Xdata = Xdata_df.values
del Xdata_df

# Load data with missing values from a csv for analysis:
Xdata_df = pd.read_csv(CorruptDataPath)
Xdata_Missing = Xdata_df.values
del Xdata_df

# Properties of data:
Xdata_length = Xdata_Missing.shape[0] # number of data points to use
n_x = Xdata_Missing.shape[1] # dimensionality of data space
ObsRowInd = np.where(np.isfinite(np.sum(Xdata_Missing,axis=1)))
NanRowInd = np.where(np.isnan(np.sum(Xdata_Missing,axis=1)))
NanIndex = np.where(np.isnan(Xdata_Missing))
Xdata_Missing_Rows = NanRowInd[0] # number of rows with missing values

# Number of missing values
NanCount = len(NanIndex[0])

# Zscore for reconstruction error checking:
scRecon = StandardScaler()
scRecon.fit(Xdata)

# Zscore of data produces much better results
sc = StandardScaler()
Xdata_Missing_complete = np.copy(Xdata_Missing[ObsRowInd[0],:])
# standardise using complete records:
sc.fit(Xdata_Missing_complete)
Xdata_Missing[NanIndex] = 0
Xdata_Missing = sc.transform(Xdata_Missing)
Xdata_Missing[NanIndex] = np.nan
del Xdata_Missing_complete
Xdata = sc.transform(Xdata)

def next_batch(Xdata,batch_size, MissingVals = False):
    """ Randomly sample batch_size elements from the matrix of data, Xdata.
        Xdata is an [NxM] matrix, N observations of M variables.
        batch_size must be smaller than N.
        
        Returns Xdata_sample, a [batch_size x M] matrix.
    """
    if MissingVals:
        # This returns records with any missing values replaced by 0:
        Xdata_length = Xdata.shape[0]
        X_indices = random.sample(range(Xdata_length),batch_size)
        Xdata_sample = np.copy(Xdata[X_indices,:])
        NanIndex = np.where(np.isnan(Xdata_sample))
        Xdata_sample[NanIndex] = 0
    else:
        # This returns complete records only:
        ObsRowIndex = np.where(np.isfinite(np.sum(Xdata,axis=1)))
        X_indices = random.sample(list(ObsRowIndex[0]),batch_size)
        Xdata_sample = np.copy(Xdata[X_indices,:])
    
    return Xdata_sample
'''
==============================================================================
'''
# INITIALISE AND TRAIN VAE
# define dict for network structure:
network_architecture = \
    dict(n_hidden_recog_1=Encoder_hidden1, # 1st layer encoder neurons
         n_hidden_recog_2=Encoder_hidden2, # 2nd layer encoder neurons
         n_hidden_gener_1=Decoder_hidden1, # 1st layer decoder neurons
         n_hidden_gener_2=Decoder_hidden2, # 2nd layer decoder neurons
         n_input=n_x, # data input size
         n_z=latent_size)  # dimensionality of latent space

# initialise VAE:
vae = TFVariationalAutoencoder(network_architecture, 
                             learning_rate=learning_rate, 
                             batch_size=batch_size)

# train VAE on corrupted data:
vae = vae.train(XData=Xdata_Missing,
                training_epochs=training_epochs)

# plot training history:
fig = plt.figure(dpi = 150)
plt.plot(vae.losshistory_epoch,vae.losshistory)
plt.xlabel('Epoch')
plt.ylabel('Evidence Lower Bound (ELBO)')
plt.show()
'''
==============================================================================
'''
# IMPUTE MISSING VALUES
# impute missing values:
X_impute = vae.impute(X_corrupt = Xdata_Missing, max_iter = ImputeIter)

# plot imputation results for sample values:
fig = plt.figure(dpi = 150)
subplotmax = min(NanCount,4)
for plotnum in range(subplotmax):
    TrueVal = Xdata[NanIndex[0][plotnum]][NanIndex[1][plotnum]]
    plt.subplot(subplotmax,1,plotnum+1)
    plt.plot(range(ImputeIter),vae.MissVal[:,plotnum],'-.')
    plt.plot([0, ImputeIter-1],[TrueVal, TrueVal])
    plt.xlabel('Iteration')
    plt.ylabel('Missing value ' + str(plotnum+1), fontsize=6)
    plt.tick_params(labelsize='small')
plt.show()

# plot imputation results for one variable:
var_i = 0
min_i = np.min(Xdata[:,var_i])
max_i = np.max(Xdata[:,var_i])

fig = plt.figure(dpi = 150)
plt.plot(Xdata[NanIndex[0][np.where(NanIndex[1]==var_i)],var_i],X_impute[NanIndex[0][np.where(NanIndex[1]==var_i)],var_i],'.')
plt.plot([min_i, max_i], [min_i, max_i])
plt.xlabel('True value')
plt.ylabel('Imputed value')
plt.show()

# Standardise Xdata_Missing and Xdata wrt Xdata:
Xdata = sc.inverse_transform(Xdata)
X_impute = sc.inverse_transform(X_impute)

Xdata = scRecon.transform(Xdata)
X_impute = scRecon.transform(X_impute)

ReconstructionError = sum(((X_impute[NanIndex] - Xdata[NanIndex])**2)**0.5)/NanCount
print('Reconstruction error (VAE):')
print(ReconstructionError)
ReconstructionError_baseline = sum(((Xdata[NanIndex])**2)**0.5)/NanCount
print('Reconstruction error (replace with mean):')
print(ReconstructionError_baseline)
'''
==============================================================================
'''
# GENERATE VALUES AND PLOT HISTOGRAMS
np_x = next_batch(Xdata_Missing, 2000)
# reconstruct data by sampling from distribution of reconstructed variables:
x_hat = vae.reconstruct(np_x, sample = 'sample')

x_hat_prior = vae.generate(n_samples = 1000)
x_hat_prior = x_hat_prior.eval()

subplotmax = min(n_x,5)
f, axarr = plt.subplots(subplotmax, subplotmax, sharex='col', dpi = 150)
f.suptitle('Posterior sample')
f.subplots_adjust(wspace = 0.3)
for k in range(subplotmax):
    for j in range(subplotmax):
        if k == j:
            axarr[k, j].hist(np_x[:,k],bins = 30, density=True)
            axarr[k, j].hist(x_hat[:,k],bins = 30, alpha = 0.7, density=True)
            axarr[k, j].tick_params(labelsize='xx-small', pad = 0)
            if j == 0:
                axarr[k, j].set_ylabel('Variable 1', fontsize=6)
            elif j == subplotmax-1:
                axarr[k, j].set_xlabel('Variable ' + str(subplotmax), fontsize=6)
        else:
            axarr[k, j].plot(np_x[:,k], np_x[:,j], '+',label = 'Data')
            axarr[k, j].plot(x_hat[:,k], x_hat[:,j], '.', alpha = 0.2, label='Posterior')
            axarr[k, j].tick_params(labelsize='xx-small', pad = 0)
            if j == 0:
                axarr[k, j].set_ylabel('Variable ' + str(k+1), fontsize=6)
            if k == subplotmax-1:
                axarr[k, j].set_xlabel('Variable ' + str(j+1), fontsize=6)

f, axarr = plt.subplots(subplotmax, subplotmax, sharex='col', dpi = 150)
f.suptitle('Prior sample')
f.subplots_adjust(wspace = 0.3)
for k in range(subplotmax):
    for j in range(subplotmax):
        if k == j:
            axarr[k, j].hist(np_x[:,k],bins = 30, density=True)
            axarr[k, j].hist(x_hat_prior[:,k],bins = 30, alpha = 0.7, density=True)
            axarr[k, j].tick_params(labelsize='xx-small', pad = 0)
            if j == 0:
                axarr[k, j].set_ylabel('Variable 1', fontsize=6)
            elif j == subplotmax-1:
                axarr[k, j].set_xlabel('Variable ' + str(subplotmax), fontsize=6)
        else:
            axarr[k, j].plot(np_x[:,k], np_x[:,j], '+',label = 'Data')
            axarr[k, j].plot(x_hat_prior[:,k], x_hat_prior[:,j], '.', alpha = 0.2, label='Prior')
            axarr[k, j].tick_params(labelsize='xx-small', pad = 0)
            if j == 0:
                axarr[k, j].set_ylabel('Variable ' + str(k+1), fontsize=6)
            if k == subplotmax-1:
                axarr[k, j].set_xlabel('Variable ' + str(j+1), fontsize=6)

vae.sess.close()
