# machine-learning-ucsc-extension-team-project

Team members:
* Yao Lu
* Feiyang Zhao
* Marvin Corro
* Bryce Edwards
* Albert Azali
* Sahil Faruque
* Ksheerasagar Akella


- PCA cmd:
python3.5 pca.py --load --feature all --size small --genre1 Instrumental --genre2 Rock --level1 all --level2 mean --plot --dumpfile test.csv

--genre1/genre2 = all or in [Hip-Hop, Pop, Folk, Experimental, Rock, International, Electronic, Instrumental]

--level1 = all or in ['chroma_stft', 'chroma_cqt', 'chroma_cens', 'tonnetz', 'mfcc', 'rmse', 'zcr', 'spectral_centroid', 'spectral_bandwidth', 'spectral_contrast', 'spectral_rolloff']

--level2 = all or in ["kurtosis", "mean", "min", "max", "skew", "median", "std"]

--plot does 2D PCA plot

--dumpfile dumps the selected level1/2 feature vectors into a csv


