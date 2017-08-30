import numpy as np
import algorithms.util.PrincipalComponentAnalysis as pca
import fma.FmaDataLoader as fma
import fma.excel_operations as eo


def create_pca_dataset_for_feature(feature, output_excel_file):
    data = None
    column_headers = []

    for attribute in fmaObj.ATTRIBUTES:
        nd_data, v_target = fmaObj.load_specific_data(f_size=fmaObj.SUBSETS[0],
                                                      f_feature=feature,
                                                      f_attributes=attribute)
        if nd_data.shape[1] > 1:
            P, num_of_pcs = pcaObj.pca(nd_data.as_matrix())
        else:
            P, num_of_pcs = nd_data.as_matrix(), 1

        if data is None:
            data = P
        else:
            data = np.column_stack((data, P))

        column_headers.extend([attribute] * num_of_pcs)

    eo.writeExcelData(data=[column_headers],
                      excelFile=output_excel_file,
                      sheetName='Sheet1',
                      startRow=1,
                      startCol=1)

    eo.writeExcelData(data=data,
                      excelFile=output_excel_file,
                      sheetName='Sheet1',
                      startRow=2,
                      startCol=1)


if __name__ == "__main__":
    pcaObj = pca.PrincipalComponentAnalysis()
    fmaObj = fma.FmaDataLoader('./data')

    outputExcelFile = r"./data/mfcc_pca_dataset.xlsx"
    create_pca_dataset_for_feature('mfcc', outputExcelFile)
    outputExcelFile = r"./data/tonnetz_pca_dataset.xlsx"
    create_pca_dataset_for_feature('tonnetz', outputExcelFile)
