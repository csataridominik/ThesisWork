import open3d as o3d
import numpy as np
from tqdm import tqdm
import glob
import os

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

def load_obj():

    idx = 0
    for folder in tqdm(glob.glob(os.path.join('Original_data', '*'))):
        
        name = 'model_normalized.obj'
        model_path = os.path.join(folder, 'models', name)
            
        mesh = o3d.io.read_triangle_mesh(model_path)

        #num_points = 100_000
        #points = mesh.sample_points_uniformly(number_of_points=num_points)
        
        o3d.io.write_triangle_mesh("100k_points\\" + str(idx) + ".obj" , mesh)
        idx += 1


load_obj()