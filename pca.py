import example
from example import FMA
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
import argparse
import code
import scipy.linalg as linalg

FEATURES = ['chroma_stft', 'chroma_cqt', 'chroma_cens', 'tonnetz', 'mfcc', 'rmse', 'zcr', 'spectral_centroid', 'spectral_bandwidth', 'spectral_contrast', 'spectral_rolloff']
LEVEL1 = FEATURES
LEVEL2 = ["kurtosis", "mean", "min", "max", "skew", "median", "std"]
GENRES = ["Hip-Hop", "Pop", "Folk", "Experimental", "Rock", "International", "Electronic", "Instrumental"]
#GENRES = ['Rock', 'Electronic', 'Experimental', 'Hip-Hop', "Folk", "Instrumental", "Pop", "International", "Classical", "Jazz", "Country", "Soul-RnB", "Spoken", "Blues", "Easy Listening"]

class PCA:
    def __init__(self):
        """
        pass
        """
        
    def LoadMetaData(self):
        fma = FMA()
        script_name = r"Example"
        print("[%s] Loading Metadata..." % script_name)
        self.tracks, self.genres, self.features, self.echonest = fma.LoadMetadata()

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

        #if feature not in FEATURES or feature != 'all':
        #    sys.exit("wrong features")

        if feature == 'all':
            X = self.features.loc[small & (genre1 | genre2)]
        else:
            X = self.features.loc[small & (genre1 | genre2), feature]

        y = self.tracks.loc[small & (genre1 | genre2), ('track', 'genre_top')]

        return (X, y)

    def GetTwoGenreFeatures(self, feature, size, genre1, genre2, level1, level2, dump):
        """
        Inputs:
           feature - all features
           size - small, medium, or large data set
           genre1/genre2 - top level genre names
           level1/2 - level 1/2 features

        Outputs:
           X - feature vector, mean feature vector
           y - labels
        """
        small  = self.tracks['set', 'subset'] <= size
        genre1 = self.tracks['track', 'genre_top'] == genre1
        genre2 = self.tracks['track', 'genre_top'] == genre2

        #if feature != 'all':
        #    sys.exit("--features needs to be all")


        if feature == 'all':
            F = self.features.loc[small & (genre1 | genre2)]
        else:
            F = self.features.loc[small & (genre1 | genre2), feature]

        if level1 == 'all':
            if level2 == 'all':
                X = F
            elif level2 in LEVEL2:
                X = F.loc(axis=1)[:, level2]
            else:
                sys.exit("please specity a valid level 2")
        else:
            if level2 == 'all':
                X = F[level1]
            elif level2 in LEVEL2:
                X = F[level1, level2]
        #code.interact(local=locals())
        
        # dump principal csv
        if dump:
            df = pd.DataFrame(X)
            df.to_csv(dump)

        y = self.tracks.loc[small & (genre1 | genre2), ('track', 'genre_top')]
        return (X, y)


    def getPCA(self, X, y):
        X = skl.decomposition.PCA(n_components=2).fit_transform(X)
        y = skl.preprocessing.LabelEncoder().fit_transform(y)
        return X, y

    
    def PCA(self, X, y):
        u = np.mean(X, axis=0)
        Z = X - u
        C = np.cov(Z, rowvar=False)
        [v, V] = linalg.eigh(C)
        v = np.flipud(v)
        #V = np.flipud(V.T)[0:dim,:]
        #P = np.dot(Z,V.T)[:,0:dim]
        V = np.flipud(V.T)
        P = np.dot(Z,V.T)
        return (P, y)

    def eigenLength(self, P):
        """
        n-dimentional length of an eigen vector
        """
        return np.sqrt(np.sum(np.square(P), axis=1))

    def findOptimalPrincipals(self, P, y):
        eigen_len = self.eigenLength(P)
        dim = 0;
        accuracy = 0.0;
        while accuracy < 0.9:
            dim += 1
            curr_eigen_len = self.eigenLength(P[:,0:dim])
            accuracy = np.mean(curr_eigen_len/eigen_len)
            print("# of PCA:", dim, "accuracy", accuracy)

        return dim, accuracy
    
    def findGenrePrincipals(self, feature, size, genre1, genre2, level1, level2, plot, dump):

        if genre1 == "all" and genre2 == "all":
            for i in range(0, len(GENRES)-1):
                genre1 = GENRES[i]
                for j in range(i+1, len(GENRES)):
                    genre2 = GENRES[j]
                    print("\n[PCA:] \nlevel 1 feature:", level1, "\nlevel 2 feature:", level2)
                    print("[# Principal] Genre1:", genre1, "vs", "Genre2:", genre2)
                    X, y = self.GetTwoGenreFeatures(feature, size, genre1, genre2, level1, level2, dump)
                    #code.interact(local=locals())
                    try:
                        P, y = self.PCA(X, y)
                        dim, accuracy = self.findOptimalPrincipals(P, y)
                    except:
                        pass
        else:
            print("\n[PCA:] \nlevel 1 feature:", level1, "\nlevel 2 feature:", level2)
            print("[# Principal] Genre1:", genre1, "vs", "Genre2:", genre2)
            X, y = self.GetTwoGenreFeatures(feature, size, genre1, genre2, level1, level2, dump)
            P, y = self.PCA(X, y)
            dim, accuracy = self.findOptimalPrincipals(P, y)
            if plot:
                X, y = self.getPCA(X, y)
                self.plot(X, y)



    def plot(self, X, y):
        plt.scatter(X[:,0], X[:,1], c=y, cmap='RdBu', alpha=0.5)
        plt.show()
        
def main():
    # commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--load", action="store_true")
    parser.add_argument("-f", "--feature", default="all", help="feature type has to be all for PCA estimation")
    parser.add_argument("-s", "--size", default="small", help="small, medium, large")
    parser.add_argument("--genre1", default="all")
    parser.add_argument("--genre2", default="all")
    parser.add_argument("-lv1", "--level1", default="all")
    parser.add_argument("-lv2", "--level2", default="all")
    parser.add_argument("-p", "--plot", action="store_true")
    parser.add_argument("-d", "--dumpfile", default="principal_feature.csv")
    args = parser.parse_args()

    # MFCC PCA
    # small = tracks['set', 'subset'] <= 'small'
    # genre1 = tracks['track', 'genre_top'] == 'Instrumental'
    # genre2 = tracks['track', 'genre_top'] == 'Hip-Hop'

    # Load metadata
    if args.load:
        pca = PCA()
        pca.LoadMetaData()
    # find PCAs
    pca.findGenrePrincipals(args.feature, args.size, args.genre1, args.genre2, args.level1, args.level2, args.plot, args.dumpfile)






if __name__ == "__main__":
    main()
    


