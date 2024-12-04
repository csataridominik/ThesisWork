from model import MultiViewCompletionNet,MultiViewCompletionNet2
from MVPCC_Networks import InpaintGenerator
from reproject import reproject
from reproject2 import reproject2
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from raycasting_projections import plot_ as plot_2
def plot_(matrix,name):
    plt.imshow(matrix)

    plt.colorbar()  # Optional: shows a color scale bar
    plt.title(name)
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.show()

def predict(model,device,input,target=0,IsThereTarget=False):
    loss = nn.MSELoss()
    model.eval()

    single_input = torch.tensor([input], dtype=torch.float32)
    if IsThereTarget:
        single_target = torch.tensor([target], dtype=torch.float32)
    
    if torch.cuda.is_available():
        single_input = single_input.to(device)

    with torch.no_grad():
        
        output = model(single_input)
    
    if IsThereTarget:
        loss = loss(output[0], single_target[0])
        return output, loss
    
    return output

def predict_main():

    path = 'pretrained_models/trained_model_object_loss.pth'
    model = MultiViewCompletionNet()
    model.load_state_dict(torch.load(path,map_location=torch.device('cpu')),strict=False)


    if model:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            model = model.to(device)
        else:
            model = model

    path =  'sparse_depth_maps/66_65/'
    input = [
        np.loadtxt(path + 'depth_image_0.txt'),
        np.loadtxt(path + 'depth_image_1.txt'),
        np.loadtxt(path + 'depth_image_2.txt'),
        np.loadtxt(path + 'depth_image_3.txt')
        ]

    path = 'projected_depth_maps/66/'
    target = [
        np.loadtxt(path + 'depth_image_0.txt'),
        np.loadtxt(path + 'depth_image_1.txt'),
        np.loadtxt(path + 'depth_image_2.txt'),
        np.loadtxt(path + 'depth_image_3.txt')
        ]

    output,loss = predict(model,device,input,target)


    loss = 0
    print(f'This is MSE loss for output: {loss}')
    #reproject('projected_depth_maps/65/',depth_images=target)
    output = np.asarray(output).squeeze()


    reproject('incomplete_depth_maps/66_0/',depth_images=output)


    path = 'projected_depth_maps/66/'
    target = [
        np.loadtxt(path + 'depth_image_0.txt'),
        np.loadtxt(path + 'depth_image_1.txt'),
        np.loadtxt(path + 'depth_image_2.txt'),
        np.loadtxt(path + 'depth_image_3.txt')
        ]



    for i in range(0,4):
        #plot_(input[i],"Input")
        #plot_(target[i],"Target")
        plot_(output[i],"Output")

    #plot_(target[0],"asd")

def predict_main2():
    target_path = "XYZ_projections\XYZ_66.npy"
    sparse_input_path = "XYZ_projections_sparse\XYZ_66.npy"

    target= np.load(target_path)
    target = target.transpose(0, 3, 1, 2).copy()

    sparse_input = np.load(sparse_input_path)
    sparse_input = sparse_input.transpose(0, 3, 1, 2).copy()


    path = 'trained_model/full_target5.pth'
    model = InpaintGenerator(residual_blocks=8)
    model.load_state_dict(torch.load(path,map_location=torch.device('cpu')),strict=False)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        model = model.to(device)
    else:
        model = model

    
    output = predict(model,device,sparse_input,target,False)
    output = output.cpu().detach().numpy().squeeze()  # Convert to NumPy array
    #print(f'This is size of output: {output.shape}')
    
    output = output.reshape(4, 3, 256, 256)


    #print(f"This is MSE loss: {loss}")
    #reproject2(target)
    #reproject2(sparse_input)
    
    reproject2(output)

    for i in range(4):
        for j in range(1):
            #plot_(sparse_input[i][j],"Input")
            #plot_(target[i][j],"Target")
            #plot_(output[i][j],"Output")
            break
    
    return output,sparse_input




'''
The model cannot capture depth via xyz coordiantes, where the original coordinates are negative numbers.
'''

from scipy.spatial.distance import cdist
def closest_point_coordinates(target_coordinates, pcd_coordinates):
    # Compute the distance matrix (Euclidean distances between target and pcd points)
    D = cdist(target_coordinates,pcd_coordinates, metric='euclidean')

    # Find the closest point in the pcd for each target point
    #closest_indices = np.argmin(D, axis=1)  # Axis=1 because we're looking for the closest pcd point for each target point

    '''
    Here:
    loop thru each closest indices, and have a D_nan matrix set
    to all false, but when we select an index from pcd_coords
    put the whole row of it in d_nan to True.
    If the value in d_nan is true for index pcd_coord than we
    append selected_coords([-1,-1,-1])
    '''
    
    output = np.ones(target_coordinates.shape)*(-1)
    selected_coords = np.ones(pcd_coordinates.shape[0])*(0)

    for idx in range(target_coordinates.shape[0]):
        
        closest_idx = np.argmin(D[idx])
        if selected_coords[closest_idx] == 0:
            output[idx] = pcd_coordinates[closest_idx]
            selected_coords[closest_idx] = 0
        else:
            output[idx] = [-1,-1,-1]


    return output

def predict_main3():


    epsilon = 0.01
    output,inputs = predict_main2()


    transformed_output = []
    for image,in_image in zip(output,inputs):
        coordinates = []
        in_coordinates = []
        epsilon = 0.01
        
        for i in range(image.shape[1]):
            for j in range(image.shape[2]):
                x, y, z = image[:,i, j]
                x2,y2,z2 = in_image[:,i, j]
                
                # Only keep the point if not all coordinates are zero
                #if abs(x) > epsilon and abs(y) > epsilon and abs(z) > epsilon:
                coordinates.append([x, y, z])
                if abs(x2) > epsilon and abs(y2) > epsilon and abs(z2) > epsilon:
                    in_coordinates.append([x, y, z])

        coordinates = np.array(coordinates)
        in_coordinates = np.array(in_coordinates)
        changed_input = closest_point_coordinates(coordinates,in_coordinates)
    
        x_res = np.reshape(changed_input[:,0], (image.shape[1],image.shape[2]))
        y_res = np.reshape(changed_input[:,1], (image.shape[1],image.shape[2]))
        z_res = np.reshape(changed_input[:,2], (image.shape[1],image.shape[2]))
        XYZ = np.stack((x_res, y_res, z_res), axis=-1)
        transformed_output.append(XYZ)
        #plot_(x_res,"i")
        #plot_(y_res,"i")
        plot_(z_res,"i")

    #reproject2(np.asanyarray(transformed_output).transpose(0, 3, 1, 2).copy())

#predict_main3()
#predict_main2()