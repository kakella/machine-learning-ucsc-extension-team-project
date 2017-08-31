import fma.utils as utils


class FmaDataLoader:

    SUBSETS = ['small', 'medium', 'large']

    FEATURES = ['chroma_stft', 'chroma_cqt', 'chroma_cens', 'tonnetz', 'mfcc', 'rmse', 'zcr', 'spectral_centroid',
                'spectral_bandwidth', 'spectral_contrast', 'spectral_rolloff']

    ATTRIBUTES = ["kurtosis", "mean", "min", "max", "skew", "median", "std"]

    GENRES = ['Rock', 'Electronic', 'Experimental', 'Hip-Hop', "Folk", "Instrumental", "Pop", "International",
              "Classical", "Jazz", "Country", "Soul-RnB", "Spoken", "Blues", "Easy Listening", "Old-Time / Historic"]

    def __init__(self, path_prefix='.'):
        self.__load_data(path_prefix)

    def __load_data(self, path_prefix):
        self.tracks = utils.load(path_prefix + "/tracks.csv")
        self.genres = utils.load(path_prefix + "/genres.csv")
        self.features = utils.load(path_prefix + "/features.csv")

    def load_specific_data(self, f_size, f_feature='all', f_genre='all', f_attributes='all'):
        small = self.tracks['set', 'subset'] <= f_size

        if f_genre != 'all' and f_feature != 'all' and f_attributes != 'all':
            genre = self.tracks['track', 'genre_top'] == f_genre
            X = self.features.loc[small & genre, f_feature][f_attributes]
            y = self.tracks.loc[small & genre, ('track', 'genre_top')]

        elif f_feature != 'all' and f_attributes != 'all':
            X = self.features.loc[small, f_feature][f_attributes]
            y = self.tracks.loc[small, ('track', 'genre_top')]

        elif f_genre != 'all':
            genre = self.tracks['track', 'genre_top'] == f_genre
            X = self.features.loc[small & genre]
            y = self.tracks.loc[small & genre, ('track', 'genre_top')]

        else:
            X = self.features.loc[small]
            y = self.tracks.loc[small, ('track', 'genre_top')]

        return X, y.values

    def load_split_data(self):
        small = self.tracks['set', 'subset'] <= 'small'

        train = self.tracks['set', 'split'] == 'training'
        val = self.tracks['set', 'split'] == 'validation'
        test = self.tracks['set', 'split'] == 'test'

        y_train = self.tracks.loc[small & train, ('track', 'genre_top')]
        y_test = self.tracks.loc[small & test, ('track', 'genre_top')]
        X_train = self.features.loc[small & train, 'mfcc']
        X_test = self.features.loc[small & test, 'mfcc']

        return X_train, y_train, X_test, y_test


def main():
    fma = FmaDataLoader('./data')
    X, y = fma.load_specific_data(f_size=fma.SUBSETS[0], f_feature=fma.FEATURES[0], f_attributes=fma.ATTRIBUTES[0])
    print(X)
    print(y)


if __name__ == "__main__":
    main()
