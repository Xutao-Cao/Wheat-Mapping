import imp
import os
import numpy as np
from regex import R
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
BASE_PATH = "./Data"

def get_data(sites: list, years: list, base_path: str , type: str, verbose: bool):
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
    
    model: "Unet" "ConvLSTM" "dcm"

    """
    def __init__(self, sites: list, years: list, type: str, base_path = BASE_PATH, model = "Unet") :
        self.model = model
        self.sites = sites
        self.years = years
        self.bath_path = base_path
        self.type = type
        #image (H, W, T, C)
        self.image, self.label = get_data(sites, years, base_path, type, False)
        self.channel_mean = None
        self.channel_std = None
        self.transform = None
        if self.model == "Unet":
            x_stacked = np.moveaxis(np.stack(self.image, -1), [2, 3], [0, 1])
            x_stacked = x_stacked.reshape((x_stacked.shape[0], x_stacked.shape[1], -1))
            x_reshaped = x_stacked.reshape(-1, x_stacked.shape[-1])
            self.channel_mean = np.mean(x_reshaped, axis=1)
            self.channel_std = np.std(x_reshaped, axis=1)
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.channel_mean, self.channel_std)
            ])
        elif self.model == "ConvLSTM":
            x_stacked = np.stack(self.image, 0)
            self.channel_mean = np.mean(x_stacked, [0,1,2,3])
            self.channel_std = np.std(x_stacked, [0,1,2,3])
            

    def __len__(self):
        return len(self.image)
    
    def __getitem__(self, index) :
        # reshaped_image = np.moveaxis(self.image[index], [2, 3], [0, 1])
        images = None
        if self.model == "Unet":
            reshaped_image = self.image[index].reshape((self.image[index].shape[0], self.image[index].shape[1], -1)).astype('float32')
            images = self.transform(reshaped_image)
        elif self.model == "ConvLSTM":
            normalized_image = (self.image[index] - np.broadcast_to(self.channel_mean, self.image[index].shape))/(np.broadcast_to(self.channel_std, self.image[index].shape) + 1e-7)
            images = torch.FloatTensor(np.moveaxis(normalized_image, [0, 1], [2, 3])) 
            
        images = self.transform(reshaped_image)
        labels = torch.FloatTensor(self.label[index])
        return images, labels




