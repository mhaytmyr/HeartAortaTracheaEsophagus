import pandas as pd
import xlrd

def open_file_test(path):
    """
    Open and read an Excel file
    """
    book = xlrd.open_workbook(path)
 
    # print number of sheets
    print(book.nsheets)
 
    # print sheet names
    print(book.sheet_names())
 
    # get the first worksheet
    first_sheet = book.sheet_by_index(0)
 
    # read a row
    print(first_sheet.row_values(0))
 
    # read a cell
    cell = first_sheet.cell(10,3)
    print(cell)
    print(cell.value)
 
    # read a row slice
    print(first_sheet.row_slice(rowx=0,
                                start_colx=0,
                                end_colx=2))

def open_file(path):
    '''
    Method to read xls file and return pointer
    '''
    return xlrd.open_workbook(path)

def get_train_stats(book):
    '''
    Method to read trains, col_index starts at 0
    '''
    # get the first worksheet
    first_sheet = book.sheet_by_index(0)

    #iterate over all entries and store them in dictionary
    dice = {'Esophagus':[],'Heart':[],'Trachea':[],'Aorta':[]}
    hausdorff = {'Esophagus':[],'Heart':[],'Trachea':[],'Aorta':[]}
    # print("Esophagus \t Heart \t Trachea \t Aorta")
    # print("--------------------------------------")
    for row in range(3,241):
        #get organ name
        organ = first_sheet.cell(row,0).value.title()

        #now append dice to corresponding place
        cell = first_sheet.cell(row,1).value
        if organ in dice:
            dice[organ].append(cell)
        
        #now append hausdorff to corresponding place
        cell = first_sheet.cell(row,2).value
        if organ in hausdorff:
            hausdorff[organ].append(cell)

    
    return dice, hausdorff


def get_test_stats(book):
    '''
    Method to read trains, col_index starts at 0
    '''
    # get the first worksheet
    first_sheet = book.sheet_by_index(0)

    #iterate over all entries and store them in dictionary
    dice = {'Esophagus':[],'Heart':[],'Trachea':[],'Aorta':[]}
    hausdorff = {'Esophagus':[],'Heart':[],'Trachea':[],'Aorta':[]}
    # print("Esophagus \t Heart \t Trachea \t Aorta")
    # print("--------------------------------------")
    for row in range(3,121):
        #get organ name
        organ = first_sheet.cell(row,0).value.title()

        #now append dice to corresponding place
        cell = first_sheet.cell(row,5).value
        if organ in dice:
            dice[organ].append(cell)
        
        #now append hausdorff to corresponding place
        cell = first_sheet.cell(row,6).value
        if organ in hausdorff:
            hausdorff[organ].append(cell)

   
    return dice, hausdorff

import numpy as np
def box_plotter(dict,metric):
    #create list from dictionary
    data = [dict[key] for key in sorted(dict.keys())]

    green_diamond = {'markerfacecolor':'g', 'marker':'D'}
    fig, ax = plt.subplots()
    #ax.set_title('Training Phase')
    ax.set_title('Testing Phase')
    ax.set_xlabel('Organs')
    ax.set_ylabel(metric)
    ax.boxplot(data,flierprops=green_diamond)
    plt.xticks(np.arange(1,5), sorted(dict.keys()), rotation='horizontal')
    # Pad margins so that markers don't get clipped by the axes
    plt.margins(0.25)
    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(bottom=0.2)
    # plt.subplots_adjust(top=0.1)
    plt.show()

import matplotlib.pyplot as plt
if __name__ == "__main__":
    path = "Train_Results.xlsx"
    book = open_file(path)
    train_dice,train_hausdorff = get_train_stats(book)
    test_dice,test_hausdorff = get_test_stats(book)

    print("Dice coeff.")
    for key in test_dice:
        print(key," ",np.mean(test_dice[key])," +/- ",np.std(test_dice[key]))

    print("Hausdorff dist.")
    for key in test_dice:
        print(key," ",np.mean(test_hausdorff[key])," +/- ",np.std(test_hausdorff[key]))


    # box_plotter(test_hausdorff,metric='Hausdorff Distance')
    # box_plotter(test_dice,metric='Dice Coefficient')

    # box_plotter(train_hausdorff,metric='Hausdorff Distance')
    # box_plotter(train_dice,metric='Dice Coefficient')