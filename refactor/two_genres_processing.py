#! /usr/bin/env python3
#
# This script need to be run inside the fma repository, where fma_metadata
# is also extracted:
#
# git clone https://github.com/mdeff/fma.git
# cd fma
# wget https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
# unzip fma_metadata.zip
#
# Now copy script to fma directory and run. Make sure all python requirements
# are installed.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import utils

import scipy.linalg as linalg
import argparse

import FmaData
import Plotter
import algorithms.clustering.ExpectationMaximizationClassifier

script_name = "main"


def PCA(X, dim):
    u = np.mean(X, axis=0)
    Z = X - u
    C = np.cov(Z, rowvar=False)
    [v, V] = linalg.eigh(C)
    v = np.flipud(v)
    V = np.flipud(V.T)[0:dim,:]
    P = np.dot(Z,V.T)[:,0:dim]
    return (P)


def main():

    # Process commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--feature", default="mfcc", help="feature type, i.e. mfcc")
    parser.add_argument("-s", "--size", default="small", help="small, medium, large")
    parser.add_argument("--genre1", default="Rock")
    parser.add_argument("--genre2", default="Electronic")
    parser.add_argument("-o", "--offset", default=0, type=int)
    parser.add_argument("-p", "--plot", action="store_true")
    parser.add_argument("-d", "--dim", default=2, type=int)
    parser.add_argument("-a", "--algorithm", default="EM")
    args = parser.parse_args()


    ############################################################################################
    # Load all metadata and get desired data set
    ############################################################################################

    # Read in metadata csv
    print("[%s] Loading Metadata..." % script_name)
    fma = FmaData.FmaData()
    fma.LoadMetadata()

    # Get X (feature vector) and y (label vector) of two genres
    (X, y) = fma.GetTwoGenre(args.feature, args.size, args.genre1, args.genre2)

    # Reduce dimensions with PCA
    X = PCA(X, args.dim)

    ############################################################################################
    # Optionally, adjust all genre1 data to manually separate clusters
    ############################################################################################

    if args.offset:
        col = y==args.genre1
        index = col
        for i in range(args.dim-1):
            index = np.vstack((index, col))
        X[index.transpose()] -= args.offset

    ############################################################################################
    # Plot all points
    ############################################################################################
    if args.plot:
        Plotter.Plotter.PlotData(X, y)

    ############################################################################################
    # Run algorithm on X and y
    #   X is Nxd feature array
    #   y is Nx1 genre label array
    ############################################################################################
    if args.algorithm == "EM":
        em = algorithms.clustering.ExpectationMaximizationClassifier.ExpectationMaximizationClassifier(X, 2)
        em.Run(iterations=10000, tolerance=10**-3, print_progress=True, plot=args.plot)

    ### ADD Additional Algorithms here ###


    # Keep plot open
    if args.plot:
        while True:
            plt.pause(0.1)

if __name__ == "__main__":
    main()

