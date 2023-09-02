# MNIST_Autoencoder
Testing of a Dense Autoencoder with MNIST

## Description
On this project I used the MNIST dataset to test the limits of a dense and convolutional autoencoder with different bottleneck sizes. We also plot the specific case of a latent 2D space, in an attempt to recreate a 2D clustering of digits.

## Build Status
Complete.

## Files
- _AutoEncoder.py_: implementation of an autoencoder in pytorch
- _AutoEncoderConv.py_, _AutoEncoderConv2.py_, _AutoEncoderConv3.py_: implementation of a particular convolution autoencoder in pytorch
- _main.ipynb_: Main notebook where the tests are performed.
- params, params_conv: files containing the parameters of pre trained models

## Packages
- torch
- time
- torchaudio
- os
- pandas
- matplotlib
- numpy
