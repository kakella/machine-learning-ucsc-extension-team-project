import numpy as np
import random as rd
import pandas as pd
from sklearn.cross_validation import train_test_split

import algorithms.classification.MeanSquareErrorMinimizerLinearClassifier as linear
import algorithms.classification.EnsembleLinearClassifier as ensemble

import algorithms.clustering.KMeansClusteringClassifier as kMeans
import algorithms.clustering.ExpectationMaximizationClassifier as expMax

import algorithms.util.RunAsBinaryClassifier as run

# import fma.FmaDataLoader as fma
import fma.excel_operations as eo


def run_KMeansClusteringClassifier():
    print('****************************************************************')
    print('K-Means Clustering Classifier')
    print('****************************************************************')
    for k in range(1, 10):
        print('num of classes:', k)
        classifier = kMeans.KMeansClusteringClassifier()
        successive_means, n_iterations = classifier.train(nd_data, k)
        print('classifier created in [%s] iterations' % n_iterations)
        # print('classifier:\n', classifier.classifier)
        # print('successive means:\n', successive_means)

        classification_output = classifier.classify(v_query)
        print('query %s classified as "%s"\n' % (v_query, classification_output))


def run_LinearClassifier():
    print('****************************************************************')
    print('Mean Square Error Minimizer - Linear Classifier')
    print('****************************************************************')
    classifier = linear.MeanSquareErrorMinimizerLinearClassifier()
    classifier.train(nd_data, v_target)
    classification_output = classifier.classify(v_query)
    print('query %s classified as "%s"\n' % (v_query, classification_output))


def create_EnsembleClassifier(k):
    print('****************************************************************')
    print('Ensemble - Linear Classifier')
    print('****************************************************************')
    print('num of linear classifiers:', k)
    runner = run.RunAsBinaryClassifier()
    runner.createClassifiers(ensemble.EnsembleLinearClassifier, nd_data, v_target, [], k)

    classification_output = []
    for v_query in test_data:
        classification_output.append(runner.runClassifiers(v_query))

    print(classification_output)
    df = pd.DataFrame(classification_output)
    df.to_csv('../fma/data/final_data/ensembleClassifier_output.csv')


def run_ExpectationMaximizationClassifier():
    print('****************************************************************')
    print('Expectation Maximization Clustering Classifier')
    print('****************************************************************')
    for k in range(1, 10):
        classifier = expMax.ExpectationMaximizationClassifier(nd_data, k)
        classifier.Run()
        # classification_output = classifier.classify(v_query)
        # print('query %s classified as "%s"\n' % (v_query, classification_output))


def get_random_data():
    X = (np.random.random_sample(10000) + np.random.random_sample(10000)).reshape(500, 20)
    T = [rd.randint(0, 2) for _ in range(500)]
    Q = np.random.random_sample(20)
    return X, T, Q


def get_audio_data(feature):
    if feature == 'mfcc':
        inputExcelFile = r"../fma/data/mfcc_pca_dataset.xlsx"
        data = eo.readExcel(inputExcelFile)
    elif feature == 'tonnetz':
        inputExcelFile = r"../fma/data/tonnetz_pca_dataset.xlsx"
        data = eo.readExcel(inputExcelFile)
    else:
        return None, None

    inputExcelFile = r"../fma/data/small_target_dataset.xlsx"
    target = eo.readExcel(inputExcelFile)

    ###########################################################################
    # Below code only needs to be used once to create small_target_dataset.xlsx
    ###########################################################################
    # fmaObj = fma.FmaDataLoader('../fma/data')
    # _, target = fmaObj.load_specific_data(fmaObj.SUBSETS[0], feature)
    # outputExcelFile = r"../fma/data/small_target_dataset.xlsx"
    # eo.writeExcelData(data=[target],
    #                   excelFile=outputExcelFile,
    #                   sheetName='Sheet1',
    #                   startRow=2,
    #                   startCol=1)

    return data, target.T[0]


def get_final_data():
    train_data = pd.read_csv('../fma/data/final_data/training_pca.csv', index_col=0).as_matrix()[1:]
    test_data = pd.read_csv('../fma/data/final_data/test_pca.csv', index_col=0).as_matrix()[1:]
    train_targets = pd.read_csv('../fma/data/final_data/training_label.csv', index_col=0).as_matrix().T[0]
    test_targets = pd.read_csv('../fma/data/final_data/test_label.csv', index_col=0).as_matrix().T[0]

    print(train_data.shape, train_targets.shape, test_data.shape, test_targets.shape)

    train_data_cleaned = []
    train_targets_cleaned = []

    for i in range(train_data.shape[0]):
        if isinstance(train_targets[i], str):
            train_data_cleaned.append(train_data[i])
            train_targets_cleaned.append(train_targets[i])

    test_data_cleaned = []
    test_targets_cleaned = []

    for i in range(test_data.shape[0]):
        if isinstance(test_targets[i], str):
            test_data_cleaned.append(test_data[i])
            test_targets_cleaned.append(test_targets[i])

    return np.array(train_data_cleaned), np.array(train_targets_cleaned), np.array(test_data_cleaned), np.array(test_targets_cleaned)


def get_original_audio_data(feature):
    if feature == 'mfcc':
        inputExcelFile = r"../fma/data/mfcc_pca_dataset.xlsx"
        data = eo.readExcel(inputExcelFile)
    elif feature == 'tonnetz':
        inputExcelFile = r"../fma/data/tonnetz_pca_dataset.xlsx"
        data = eo.readExcel(inputExcelFile)
    else:
        return None, None


def get_mfcc_specific_features_data():
    raw_data = pd.read_csv('../fma/data/mfcc_specific_features.csv', index_col=0).as_matrix()
    pca_data = pd.read_csv('../fma/data/mfcc_specific_features_pca.csv', index_col=0).as_matrix()
    T = raw_data[:, -1]
    return pca_data, T


if __name__ == "__main__":
    # nd_data, v_target, v_query = get_random_data()
    # test_data=[v_query]
    # X, T = get_audio_data('mfcc')
    X, T = get_mfcc_specific_features_data()
    print(X.shape, T.shape)

    X = np.column_stack((X, T))
    train, test = train_test_split(X, test_size=0.3)

    nd_data = train[:, :train.shape[1]-1]
    v_target = train[:, train.shape[1]-1]
    test_data = test[:, :test.shape[1]-1]
    v_test_target = test[:, test.shape[1]-1]

    # nd_data, v_target, test_data, v_test_target = get_final_data()

    print(nd_data.shape, v_target.shape, test_data.shape, v_test_target.shape)

    print('data to classify:\n', nd_data.shape)
    print(nd_data)
    print('----------------------------------------------------------------\n\n')
    print('targets:\n', v_target.shape)
    print(v_target)
    print('----------------------------------------------------------------\n\n')

    create_EnsembleClassifier(5000)
