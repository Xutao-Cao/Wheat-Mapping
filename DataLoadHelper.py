import numpy as np
#/content/drive/MyDrive/SRTP2021/20191/2015/S1_20191_20150901.tif
def Data_Load_Helper(location, years, path_template ="/content/drive/MyDrive/SRTP_Datasets/DatasetsForML"):
    """
    location: a county code
    years is a list of int
        e.g [2019,2018]
    path_template is the template path from which the program loads the dataset.
        Default = "/content/drive/MyDrive/SRTP_Datasets/DatasetsForML" if you are using drive for colab.
    """
    x_list = []
    y_list = []
    for year in years:
        curpath_x = path_template+"/"+str(location)+"/"+str(year)+"_x.npy"
        curpath_y = path_template+"/"+str(location)+"/"+str(year)+"_y.npy" 
        x = np.load(curpath_x)
        x_list.append(x)
        y = np.load(curpath_y)
        y_list.append(y)
    X = x_list[0]
    Y = y_list[0]
    for i in range(1, len(x_list)):
        X = np.concatenate((X ,x_list[i]), axis = 0)
        Y = np.concatenate((Y, y_list[i]), axis = 0)
    # print("finished loading")
    # print(X.shape)
    # print(Y.shape)
    return X, Y