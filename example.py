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


# Imports from example
import IPython.display as ipd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as skl
import sklearn.utils, sklearn.preprocessing, sklearn.decomposition, sklearn.svm
import librosa
import librosa.display

import utils



script_name = "Example"


class FMA:
    """Methods dealing with FMA data."""

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
        return self.tracks, self.genres, self.features, self.echonest

    def RunExampleGraph(self):
        """Example code that uses PCA on mfcc features of two top level genres and graphs it."""

        small  = self.tracks['set', 'subset'] <= 'small'
        genre1 = self.tracks['track', 'genre_top'] == 'Instrumental'
        genre2 = self.tracks['track', 'genre_top'] == 'Hip-Hop'

        X = self.features.loc[small & (genre1 | genre2), 'mfcc']
        X = skl.decomposition.PCA(n_components=2).fit_transform(X)

        y = self.tracks.loc[small & (genre1 | genre2), ('track', 'genre_top')]
        y = skl.preprocessing.LabelEncoder().fit_transform(y)

        plt.scatter(X[:,0], X[:,1], c=y, cmap='RdBu', alpha=0.5)
        print("Close graph to continue")
        plt.show()

    def RunExampleGenreClassification(self):
        """Example Genre classification from features."""

        small   = self.tracks['set', 'subset'] <= 'small'

        train   = self.tracks['set', 'split'] == 'training'
        val     = self.tracks['set', 'split'] == 'validation'
        test    = self.tracks['set', 'split'] == 'test'

        y_train = self.tracks.loc[small & train, ('track', 'genre_top')]
        y_test  = self.tracks.loc[small & test, ('track', 'genre_top')]
        X_train = self.features.loc[small & train, 'mfcc']
        X_test  = self.features.loc[small & test, 'mfcc']

        print('{} training examples, {} testing examples'.format(y_train.size, y_test.size))
        print('{} features, {} classes'.format(X_train.shape[1], np.unique(y_train).size))

        # Be sure training samples are shuffled.
        X_train, y_train = skl.utils.shuffle(X_train, y_train, random_state=42)

        # Standardize features by removing the mean and scaling to unit variance.
        scaler = skl.preprocessing.StandardScaler(copy=False)
        scaler.fit_transform(X_train)
        scaler.transform(X_test)

        # Support vector classification.
        clf = skl.svm.SVC()
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        print('Accuracy: {:.2%}'.format(score))


def main():

    fma = FMA()

    print("[%s] Loading Metadata..." % script_name)
    fma.LoadMetadata()

    print("[%s] RunExampleGraph" % script_name)
    fma.RunExampleGraph()

    print("[%s] RunExampleGenreClassification" % script_name)
    fma.RunExampleGenreClassification()


if __name__ == "__main__":
    main()

