#!/usr/bin/env python3
import glob
import os
import sys
import numpy as np
import youtube_dl
import algorithms.classification.MeanSquareErrorMinimizerLinearClassifier as linear
import algorithms.classification.EnsembleLinearClassifier as ensemble
import algorithms.clustering.KMeansClusteringClassifier as kMeans
import algorithms.clustering.ExpectationMaximizationClassifier as EM
import refactor.features as features
import refactor.FmaData as FmaData
import scipy.linalg as linalg
import pandas as pd

# FFmpeg must be installed in order for Downloader to work


class Downloader:
    def __init__(self):
        pass

    @staticmethod
    def download(url):
        """" Extracts audio from a youtube video
             Returns the name of the mp3 file if successfully downloaded """

        download_options = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
        }

        if not os.path.exists('mp3Files'):
            os.mkdir('mp3Files')
        os.chdir('mp3Files')

        # downloads mp3 file to current directory
        with youtube_dl.YoutubeDL(download_options) as ydl:
            ydl.download([url])

        list_of_files = glob.glob('*.mp3')
        name_of_mp3 = max(list_of_files, key=os.path.getctime)
        os.chdir('..')

        return name_of_mp3


class Extracter:
    """ Wrapper class for feature extraction """

    def __init__(self):
        pass

    @staticmethod
    def extract(mp3Name, extractedName):
        """ Creates .csv file of features from .mp3 file  """
        if not os.path.exists('featureFiles'):
            os.mkdir('featureFiles')

        os.chdir('featureFiles')
        if os.path.exists(extractedName):
            os.remove(extractedName)
        features.extract('../mp3Files/' + mp3Name, extractedName)
        os.chdir('..')


# Removes all mp3 files from a given directory
# Created for some testing/debugging
def clearAllMp3(dirNmae = None):
    if(dirNmae != None):
        os.chdir(dirNmae)
    for file in glob.glob('*.mp3'):
        os.remove(file)


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
    url = sys.argv[1]   # youtube video url
    audio_file = Downloader.download(url)
    features_file_name = os.path.splitext(audio_file)[0] + '.csv'
    
    Extracter.extract(audio_file, features_file_name)

    fma = FmaData.FmaData()
    script_name = 'demo.py'
    print("[%s] Loading Metadata..." % script_name)
    fma.LoadMetadata()
    query = fma.LoadQuery(features_file_name)
    genre1 = "Rock"
    genre2 = "Electronic"
    feature = "mfcc"
    size = "small"
    nd_data, target = fma.GetTwoGenre(feature, size, genre1, genre2)
    # Reduce dimensions with PCA
#    f = pd.concat(nd_data,query)

    f= np.zeros(shape=(1,query.shape[0]))
    j = 0
    for i in query:
        f[0,j] = i
        j += 1

    x = np.append(nd_data,f,axis=0)
    x = PCA(x, 4)
    nd_data = x[0:x.shape[0]-1,:]
    query = x[x.shape[0]-1,:]
#    print (nd_data)
#    print (query)
#    print (nd_data.shape)
#    print (query.shape)

    # Train all Classifiers
    ensembleClassifier = ensemble.EnsembleLinearClassifier()
    ensembleClassifier.train( nd_data, target, 10)
    linearClassifier   =  linear.MeanSquareErrorMinimizerLinearClassifier()
    linearClassifier.train( nd_data, target )
    emClassifier       = EM.ExpectationMaximizationClassifier(nd_data, 2)
#    emClassifier.Run(iterations=10000, tolerance=10**-3)
    KMeansClassifier   = kMeans.KMeansClusteringClassifier(nd_data)
    KMeansClassifier.train( 2 )

    # Classify query using all classifiers
    print ('__________________________________________________')
    print('\n\n\n\nEnsemble Classifier classified as "%s"\n' % ensembleClassifier.classify(query))
    print('Linear Classifier classified as "%s"\n' % linearClassifier.classify(query))
    kmeansResult = "Rock"
    if KMeansClassifier.classify(query) == 0:
        kmeansResult = "Electronic"
    print('KMeans Classifier classified as "%s"\n' % kmeansResult)
    
    sys.exit()

    # TRAIN ALL CLASSIFIERS
    print('Training Ensemble Classifier... ', end='')
    # *ensembleClassifier.train()
    print(' Done')

    print('Training Linear Classifier... ', end='')
    # *linearClassifier.train()
    print(' Done')

    print('Training EM Classifier... ', end='')
    # *emClassifier.train()                      <--------- doesn't have train function
    print(' Done')

    print('Training KMeans Classifier... ', end='')
    # *KMeansClassifier.train()
    print(' Done')
    

    # CLASSIFY QUERY USING ALL CLASSIFIERS
    print('\n\n')

    print('Classifying using Ensemble Classifier... ')
    # *ensembleClassifier.classify()

    print('Classifying using Linear Classifier... ')
    # *linearClassifier.classify()

    print('Classifying using EM Classifier... ')
    # *emClassifier.classify()                <------------ doesn't have classify function

    print('Classifying using KMeans Classifier... ')
    # *KMeansClassifier.classify()


if __name__ == '__main__':

    if len(sys.argv) != 2:
        print('Usage: python3 demo.py youtube_video_url')

    else:
        main()
