def readExcelSheet1(excelfile):
    from pandas import read_excel
    return read_excel(excelfile).values


def readExcelRange(excelFile, sheetName="Sheet1", startRow=1, endRow=1, startCol=1, endCol=1):
    from pandas import read_excel
    values = read_excel(excelFile, sheetName, header=None).values
    return values[startRow - 1:endRow, startCol - 1:endCol]


def readExcel(excelFile, **args):
    if args:
        data = readExcelRange(excelFile, **args)
    else:
        data = readExcelSheet1(excelFile)
    if data.shape == (1, 1):
        return data[0, 0]
    elif data.shape[0] == 1:
        return data[0]
    else:
        return data


def getSheetNames(excelFile):
    from pandas import ExcelFile
    return (ExcelFile(excelFile)).sheet_names


def writeExcelData(data, excelFile, sheetName, startRow, startCol):
    from pandas import DataFrame, ExcelWriter
    from openpyxl import load_workbook
    df = DataFrame(data=data)
    book = load_workbook(excelFile)
    writer = ExcelWriter(excelFile, engine='openpyxl')
    writer.book = book
    writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
    df.to_excel(writer, sheet_name=sheetName, startrow=startRow - 1, startcol=startCol - 1, header=False, index=False)
    writer.save()
    writer.close()
