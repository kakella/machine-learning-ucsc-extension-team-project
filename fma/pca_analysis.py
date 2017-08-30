import algorithms.util.PrincipalComponentAnalysis as pca
import fma.FmaDataLoader as fma
import fma.excel_operations as eo


outputExcelFile = r"pca_analysis_output.xlsx"


def analyze():
    pcaObj = pca.PrincipalComponentAnalysis()
    fmaObj = fma.FmaDataLoader('./data')

    analysis_output = []

    for feature in fmaObj.FEATURES:
        for attribute in fmaObj.ATTRIBUTES:
            nd_data, v_target = fmaObj.load_specific_data(f_size=fmaObj.SUBSETS[0],
                                                          f_feature=feature,
                                                          f_attributes=attribute)
            if nd_data.shape[1] > 1:
                P, num_of_pcs = pcaObj.pca(nd_data.as_matrix())
                analysis_output.append([feature, attribute, nd_data.shape[1], num_of_pcs,
                                        (nd_data.shape[1] - num_of_pcs) / nd_data.shape[1]])
            else:
                analysis_output.append([feature, attribute, 1, 1, 0])

    eo.writeExcelData(data=analysis_output,
                      excelFile=outputExcelFile,
                      sheetName='analysis',
                      startRow=2,
                      startCol=1)


if __name__ == "__main__":
    analyze()
