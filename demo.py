import glob
import os
import sys

import youtube_dl

import algorithms.classification.EnsembleLinearClassifier as Ensemble
import algorithms.classification.MeanSquareErrorMinimizerLinearClassifier as Linear
import algorithms.clustering.EM as EM
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


def main():
    url = sys.argv[1]   # youtube video url
    audio_file = Downloader.download(url)
    feature_file_name = os.path.splitext(audio_file)[0] + '.csv'
    Extracter.extract(audio_file, feature_file_name)

    ensembleClassifier = Ensemble.EnsembleLinearClassifier
    linearClassifier   = Linear.MeanSquareErrorMinimizerLinearClassifier
    emClassifier       = EM.EM
    KMeansClassifier   = KMeans.KMeansClusteringClassifier


    # Train all Classifiers
    # Classify query using all classifiers


if __name__ == '__main__':

    if len(sys.argv) != 2:
        print('Usage: python3 demo.py youtube_video_url')

    else:
        main()