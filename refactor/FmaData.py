#! /usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import fma.utils as utils


class FmaData:

    def LoadMetadata(self):
        """Loads metadata from csv files"""

        plt.rcParams['figure.figsize'] = (17, 5)

        # Load metadata and features.
        self.tracks   = utils.load("fma_metadata/tracks.csv")
        self.genres   = utils.load("fma_metadata/genres.csv")
        self.features = utils.load("fma_metadata/features.csv")
        self.echonest = utils.load("fma_metadata/echonest.csv")

        np.testing.assert_array_equal(self.features.index, self.tracks.index)
        assert self.echonest.index.isin(self.tracks.index).all()

    def LoadQuery(self, csv, feature="mfcc"):
        self.query = utils.load("featureFiles/" + csv)

        f = self.query[0] == feature
        self.query = self.query.loc[f]
        return self.query[3].values
        
    def GetFeatures(self):
        return self.features

    def GetGenres(self):
        return self.genres

    def GetTwoGenre(self, feature, size, genre1, genre2):
        """
        Inputs:
           feature - feature type such as mfcc
           size - small, medium, or large data set
           genre1/genre2 - top level genre names

        Outputs:
           X - feature vector
           y - labels
        """
        
        small  = self.tracks['set', 'subset'] <= size
        genre1 = self.tracks['track', 'genre_top'] == genre1
        genre2 = self.tracks['track', 'genre_top'] == genre2

        X = self.features.loc[small & (genre1 | genre2), feature]
        y = self.tracks.loc[small & (genre1 | genre2), ('track', 'genre_top')]

        return (X.values, y.values)

