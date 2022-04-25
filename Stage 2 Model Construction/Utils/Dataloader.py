import imp
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
BASE_PATH = "./Data"

def get_data(sites: list, years: list, base_path: str, type: str, verbose: bool):
    """
    Get data imgs from base path

    Parameters
    ----------
    sites: list
        contains the list of sites of images
    
    years: list
        contains the list of years of images

    base_path: str
        the base path of the images

    feature_type: str
        "S1" , "L8" or "Both"
    
    verbose: bool
        if true, print out the loading process
    
    Returns:
    -------
    X, Y ->list
        X is a list of training images and Y is a list of corresponding labels.
        image dim: [H, W, C, T] Where C is the channel for features and T is timestep
    """
    curr_path = None
    CDL_path = os.path.join(base_path, "CDL_label")

    if type == "S1":
        curr_path = os.path.join(base_path, "SAR_features")
    elif type == "L8":
        curr_path = os.path.join(base_path, "Sp_features")
    elif type == "Both":
        curr_path = os.path.join(base_path, "All_features")
    else:
        raise NotImplementedError
    X = []
    Y = []
    total_imgs = 0
    for site in sites:
        for year in years:
            cnt = 0
            for num in range(100):
                X_filename = "{site}_{year}_{num}.npy".format(site = site, year = year, num = num)
                Y_filename = "{site}_{year}_{num}.npy".format(site = site, year = year, num = num)
                path = os.path.join(curr_path, X_filename)
                CDL_file = os.path.join(CDL_path, Y_filename)
                if os.path.isfile(path):
                    X.append(np.load(path))
                    Y.append(np.load(CDL_file))
                    cnt += 1
                    total_imgs += 1
                else:
                    break
            if verbose:
                print("Loaded {cnt} images in site {site}, year {year}".format(cnt = cnt, site = site, year = year))
    if verbose:
        print("Completed! Loaded {total} images in total.".format(total = total_imgs))
    return X, Y

def ML_dataloader(sites: list, years: list, base_path = BASE_PATH, type = "Both", verbose = False):
    """
    Generate input for a machine learning classifer

    Parameters
    ----------
    sites: list
        contains the list of sites of images
    
    years: list
        contains the list of years of images

    base_path: str
        the base path of the images

    feature_type: str
        "S1" , "L8" or "Both"
    
    verbose: bool
        if true, print out the loading process
    
    Returns:
    -------
    X, Y ->ndarray
        X is (N, C) shaped array and Y is a list of corresponding labels.
    """
    X_list , Y_list = get_data(sites, years, base_path, type, verbose)
    X_stacked = np.stack(X_list, axis=0)
    Y_stacked = np.stack(Y_list, axis=0)
    X_stacked = X_stacked.reshape((-1, X_stacked.shape[3], X_stacked.shape[4]))
    X_stacked = X_stacked.reshape((X_stacked.shape[0], -1))
    Y_stacked = Y_stacked.reshape((-1,1))

    return X_stacked, Y_stacked

def Seq_model_dataloader(sites: list, years: list, base_path = BASE_PATH, type = "Both", verbose = False):
    """
    Generate input for sequence models

    Parameters
    ----------
    sites: list
        contains the list of sites of images
    
    years: list
        contains the list of years of images

    base_path: str
        the base path of the images

    feature_type: str
        "S1" , "L8" or "Both"
    
    verbose: bool
        if true, print out the loading process
    
    Returns:
    -------
    X, Y ->ndarray
        X is (N, L, C) shaped array where L is the length of the sequence , Y is a list of corresponding labels.
    """
    X_list , Y_list = get_data(sites, years, base_path, type, verbose)
    X_stacked = np.stack(X_list, axis=0)
    Y_stacked = np.stack(Y_list, axis=0)
    X_stacked = X_stacked.reshape((-1, X_stacked.shape[3], X_stacked.shape[4]))
    X_stacked = np.moveaxis(X_stacked, 3, -1)
    Y_stacked = Y_stacked.reshape((-1,1))

    return X_stacked, Y_stacked

def Graph_model_dataloader(sites: list, years: list, base_path = BASE_PATH, type = "Both", verbose = False):
    """
    Generate input for graph models

    Parameters
    ----------
    sites: list
        contains the list of sites of images
    
    years: list
        contains the list of years of images

    base_path: str
        the base path of the images

    type: str
        "S1" , "L8" or "Both"
    
    verbose: bool
        if true, print out the loading process
    
    Returns:
    -------
    X, Y ->ndarray
        X is (H,W)
    """
    X_list , Y_list = get_data(sites, years, base_path, type, verbose)
    X_stacked = np.stack(X_list, axis=0)
    Y_stacked = np.stack(Y_list, axis=0)
    X_stacked = X_stacked.reshape((X_stacked.shape[0], X_stacked.shape[1], -1))
    Y_stacked = Y_stacked.reshape((-1,1))

class Satellite_image_dataset(Dataset):
    """
    Construct a satellite image dataset with normalization

    Parameters
    ----------
    sites: list
        contains the list of sites of images
    
    years: list
        contains the list of years of images

    base_path: str
        the base path of the images

    type: str
        "S1" , "L8" or "Both"

    """
    def __init__(self, sites: list, years: list, type: str, base_path = BASE_PATH,) :
        
        self.sites = sites
        self.years = years
        self.bath_path = base_path
        self.type = type
        self.image, self.label = get_data(sites, years, base_path, type)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize()
        ])

    def __len__(self):
        return len(self.image)
    
    def __getitem__(self, index) :
        images = self.transform(self.image[index])
        labels = torch.IntTensor(self.label[index])
        return images, labels




