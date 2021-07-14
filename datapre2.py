import numpy as np
import datetime
from datetime import date
from osgeo import gdal
dates = ['0901','0925','1019','1112','1206','1230','0123','0216','0311','0404','0428','0522']
# load_path_template = "/content/drive/MyDrive/SRTP2021"
# save_path_template = "/content/drive/MyDrive/SRTP_Datasets/DatasetsForWMM"
load_path_template = "C:/Users/11027/Downloads/drive-download-20210515T111831Z-002"
save_path_template = "C:/Users/11027/Downloads/dataset"
#/content/drive/MyDrive/SRTP2021/20191/2015/S1_20191_20150901.tif
#/content/drive/MyDrive/SRTP2021/20191/2015/CDL_20191_2015.tif
# year = 2019
# timelist=[]
# start = datetime.date(year, 9,1)
# delta = datetime.timedelta(days=24)
# for i in range(12):
#     curtime = delta*i + start
#     timelist.append(curtime.strftime("20%y%m%d"))
# print(timelist)

def FilterForWMM(height, width, locations, years, num_of_pic):
    """
    input:
        height and width determines the size of the figure. int required
        locations: a list of int, each stands for the county Flips code of the location
        years: a list of int, each stands for the years of the raw data
        num_of_pic: the number of figures generated from one specific year of one specific location. int required
    output 
        shape: (timestep, channels, height, width)
        save path example : /content/drive/MyDrive/SRTP_Datasets/DatasetsForWMM/20191/2015/
    """
    for location in locations:
        print("processing "+str(location))
        for year in years:
            print("1")
            cdlpath = load_path_template+"/"+str(location)+"/"+str(year)+"/CDL_"+ str(location)+"_"+str(year)+".tif"
            cdl=gdal.Open(cdlpath)
            Xsize = cdl.RasterYSize
            Ysize = cdl.RasterXSize
            cnt = 0
            figs = []
            print(2)
            datecnt=1
            datelist=[]
            start = datetime.date(year, 9,1)
            delta = datetime.timedelta(days=24)
            for i in range(12):
                curtime = delta*i + start
                datelist.append(curtime.strftime("20%y%m%d"))
            for date in datelist:
                print(date)
                curpath = load_path_template+"/"+str(location)+"/"+str(year)+"/S1_"+str(location)+"_"+date+".tif"
                curpic = gdal.Open(curpath)
                Band1 = curpic.GetRasterBand(1).ReadAsArray()
                Band2 = curpic.GetRasterBand(2).ReadAsArray()
                fig = np.stack((Band1,Band2), axis=0)
                figs.append(fig)
                datecnt += 1
            pic_dict = dict()
            print(3)
            while cnt < num_of_pic:
                x_index = np.random.randint(0,Xsize-1)
                y_index = np.random.randint(0,Ysize-1)
                flag = 0
                for figure in figs:
                    if np.isnan(figure[:,x_index:x_index+width,y_index:y_index+height]).any()==True:
                        flag = 1
                        break
                if(flag == 0):
                    print(cnt)
                    new_data = np.zeros((12, 2, height, width))
                    cnt+=1
                    for i in range(12):
                        new_data[i,:,:,:] = figs[i][:,x_index:x_index+width,y_index:y_index+height]
                    save_path = save_path_template+"/"+str(location)+"/"+str(year)+"/"+"x"+str(cnt)+".npy"
                    assert np.isnan(new_data).any()==False
                    np.save(save_path, new_data)
                    cdldata1 = cdl.GetRasterBand(1).ReadAsArray()
                    cdldata = cdldata1[x_index:x_index+width,y_index:y_index+height]
                    cdl_save_path = save_path_template+"/"+str(location)+"/"+str(year)+"/"+"y"+str(cnt)+".npy"
                    assert np.isnan(cdldata).any()==False
                    np.save(cdl_save_path, cdldata)
                    pic_dict[str(cnt)]="("+str(x_index)+","+str(y_index)+")"
            np.save(save_path_template+"/"+str(location)+"/"+str(year)+"/"+"dict.npy", pic_dict)
        print("year "+str(year)+" finished!")


def Filter(graphpath, cdlpath):
    cdl = gdal.Open(cdlpath, gdal.GA_ReadOnly)
    dataset = gdal.Open(graphpath[0], gdal.GA_ReadOnly)
    nanfilter = np.zeros((dataset.RasterYSize, dataset.RasterXSize),dtype='int64')

    for path in graphpath:
        dataset = gdal.Open(path, gdal.GA_ReadOnly)
        # print(1)
        image1 = np.zeros((dataset.RasterYSize, dataset.RasterXSize,2),dtype='float32')
        image1[:,:,0] = dataset.GetRasterBand(1).ReadAsArray()
        image1[:,:,1] = dataset.GetRasterBand(2).ReadAsArray()
        for x in range(dataset.RasterYSize):
            for y in range(dataset.RasterXSize):
                if(np.isnan(image1[x, y, 0]) or np.isnan(image1[x, y, 1])):
                    nanfilter[x,y] = 1
                    continue 
    cnt = 0
    for x in range(dataset.RasterYSize):
        for y in range(dataset.RasterXSize):
            if(nanfilter[x,y] == 1):
                continue
            cnt = cnt + 1 
    
    i = 0            
    X = np.zeros((cnt, 24),dtype='float32')
    Y = np.zeros((cnt),dtype='int64')

    for path in graphpath:
       dataset = gdal.Open(path, gdal.GA_ReadOnly)
    #    print(dataset.RasterYSize,dataset.RasterXSize)
    #    print(cdl.RasterYSize,cdl.RasterXSize)
       assert dataset.RasterYSize == cdl.RasterYSize
       assert dataset.RasterXSize == cdl.RasterXSize
       image = np.zeros((dataset.RasterYSize, dataset.RasterXSize,2),dtype='float32')
       CDL = np.zeros((dataset.RasterYSize, dataset.RasterXSize), dtype='int64')
       CDL[:,:] = cdl.GetRasterBand(1).ReadAsArray()
       image[:,:,0] = dataset.GetRasterBand(1).ReadAsArray()
       image[:,:,1] = dataset.GetRasterBand(2).ReadAsArray() 
       cnt2 = 0 
       for x in range(dataset.RasterYSize):
           for y in range(dataset.RasterXSize):
               if(nanfilter[x, y] == 1):
                   continue
               X[cnt2, i] = image[x, y, 0]
               X[cnt2, i + 1] = image[x, y, 1]
               Y[cnt2] = CDL[x,y]
               cnt2 = cnt2+1
       i = i + 2
    #    print(cnt2)
    #    print(cnt)
    #    print(i)
    #    print(np.isnan(X).any())
       assert cnt2 == cnt
    assert np.isnan(X).any() == False
    assert np.isnan(Y).any() == False
    return X, Y, nanfilter

def FilterForML(locations, years):
    for location in locations:
        print("processing "+str(location))
        for year in years:
            paths = []
            datelist=[]
            start = datetime.date(year, 9,1)
            delta = datetime.timedelta(days=24)
            for i in range(12):
                curtime = delta*i + start
                datelist.append(curtime.strftime("20%y%m%d"))
            for date in datelist:
                path = load_path_template+"/"+str(location)+"/"+str(year)+"/S1_"+str(location)+"_"+date+".tif"
                paths.append(path)
                cdlpath = load_path_template+"/"+str(location)+"/"+str(year)+"/CDL_"+ str(location)+"_"+str(year)+".tif"
            X, Y, nanfilter = Filter(paths,cdlpath)
            np.save(save_path_template+"/"+str(location)+"/"+str(year)+"_x.npy",X)
            np.save(save_path_template+"/"+str(location)+"/"+str(year)+"_y.npy",Y)
            np.save(save_path_template+"/"+str(location)+"/"+str(year)+"_nanfilter.npy",nanfilter)
            print("year "+str(year)+" finished!")

# FilterForWMM(128,128,[20191],[2019,2018,2017,2016],100)
FilterForML([20191],[2019,2018,2017,2016])
                    




                


