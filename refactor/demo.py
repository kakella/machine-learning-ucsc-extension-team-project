import glob
import os
import sys

import youtube_dl

import algorithms.classification.EnsembleLinearClassifier as Ensemble
import algorithms.classification.MeanSquareErrorMinimizerLinearClassifier as Linear
import algorithms.clustering.ExpectationMaximizationClassifier as EM
import algorithms.clustering.KMeansClusteringClassifier as KMeans
from demo import features


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
        features.extract('../mp3Files/' + mp3Name, extractedName)
        os.chdir('..')


# Removes all mp3 files from a given directory
# Created for some testing/debugging
def clearAllMp3(dirNmae = None):
    if(dirNmae != None):
        os.chdir(dirNmae)
    for file in glob.glob('*.mp3'):
        os.remove(file)



# EVERYTHING WITH "*" NEEDS TO BE FILLED/FINISHED
def main():
    url = sys.argv[1]   # youtube video url
    audio_file = Downloader.download(url)
    features_file_name = os.path.splitext(audio_file)[0] + '.csv'
    Extracter.extract(audio_file, feature_file_name)

    
    # *LOAD TRAINING DATA


    # CREATE CLASSIFIER OBJECTS
    # *ensembleClassifier = Ensemble.EnsembleLinearClassifier()      
    # *linearClassifier   = Linear.MeanSquareErrorMinimizerLinearClassifier()
    # *emClassifier       = EM.ExpectationMaximizationClassifier()    
    # *KMeansClassifier   = KMeans.KMeansClusteringClassifier()      


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