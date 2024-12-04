from torch.utils.data import Dataset
import os
import numpy as np
import torch

class IncompleteDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        # Return the size of the dataset
        return len(self.X)

    def __getitem__(self, idx):
        # Return a single data point and its label
        data_point = self.X[idx]
        label = self.y[idx]
        return data_point, label
    

def create_dataset(it,dataset_size,number_of_incomplete_samples,incomplete_dir = 'incomplete_depth_maps'):
    projection_dir = 'projected_depth_maps'
    
    
    X = []
    y = []


    #for pcd_file in tqdm(pcd_files):
    for i in range(it,dataset_size):
        GT_depth_images = [
        np.loadtxt(projection_dir + f'/{i}/depth_image_0.txt',dtype=np.float32),
        np.loadtxt(projection_dir + f'/{i}/depth_image_1.txt',dtype=np.float32),
        np.loadtxt(projection_dir + f'/{i}/depth_image_2.txt',dtype=np.float32),
        np.loadtxt(projection_dir + f'/{i}/depth_image_3.txt',dtype=np.float32)
        ]
        #print(f'This is size of GT: { np.asarray(depth_images).shape}')
        
        for j in range(0,number_of_incomplete_samples):
            y.append(np.asarray(GT_depth_images)) # we start each iteration with the same y

            depth_images = [
            np.loadtxt(incomplete_dir + f'/{i}_{j}/depth_image_0.txt',dtype=np.float32),
            np.loadtxt(incomplete_dir + f'/{i}_{j}/depth_image_1.txt',dtype=np.float32),
            np.loadtxt(incomplete_dir + f'/{i}_{j}/depth_image_2.txt',dtype=np.float32),
            np.loadtxt(incomplete_dir + f'/{i}_{j}/depth_image_3.txt',dtype=np.float32)
            ]

            X.append(np.asarray(depth_images)) # we start each iteration with the same y
    
    #X = torch.from_numpy(X).float()
    #y = torch.tensor(np.asarray(y,dtype=np.float32),dtype=torch.float32)
    X = torch.from_numpy(np.asarray(X, dtype=np.float32))
    y = torch.from_numpy(np.asarray(y, dtype=np.float32))


    dataset = IncompleteDataset(X, y)
    
    return dataset


def create_dataset2(it,dataset_size,incomplete_dir = 'XYZ_projections_sparse\\XYZ_'):
    projection_dir = 'XYZ_projections\\XYZ_'
        
    X = []
    y = []

    for i in range(it,dataset_size):
        GT_XYZ = np.load(projection_dir+str(i)+".npy")
        y.append(GT_XYZ) 

        input_XYZ = np.load(incomplete_dir+str(i)+".npy")
        X.append(input_XYZ) 
    
    #X = torch.from_numpy(X).float()
    #y = torch.tensor(np.asarray(y,dtype=np.float32),dtype=torch.float32)
    X = torch.from_numpy(np.asarray(X, dtype=np.float32))
    y = torch.from_numpy(np.asarray(y, dtype=np.float32))


    dataset = IncompleteDataset(X, y)
    
    return dataset
