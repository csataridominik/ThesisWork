# author: Vinit Sarode (vinitsarode5@gmail.com) 03/23/2020

import open3d as o3d
import argparse
import os
import sys
import logging
import numpy
import numpy as np
import torch
import torch.utils.data
import torchvision
from torch.utils.data import DataLoader, random_split
from tensorboardX import SummaryWriter
from tqdm import tqdm
#from learning3d.data_utils import ClassificationData, UserData

import os
import numpy as np
import torch
from plyfile import PlyData, PlyElement
import fpsample # ---------------------------------------------------- Chooses Farthest points
import random
from torch.utils.data import Dataset
import glob
import trimesh


def load_data_shapenet(train, use_normals,save=False, random = True):

	o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Warning)

	num_points = 16384  # Adjust the number of points as needed

	BASE_DIR = os.path.dirname(os.path.abspath(__file__))

	all_data = []
	all_label = []
	curr_folder = '01'
	counter = 0
	for folder in tqdm(glob.glob(os.path.join('..','data',curr_folder, '*'))):
		'''counter += 1
		if counter < 1720:
			continue
		if counter == 1722:
			print(f'I was looking f u: {folder}')'''
		# Do not load these:
		# b0e8
		# 703
		# b998ce1c2a335d8
		# b9bf493040
		# e364eac9ec
		# e36cda06eed31d11
		# e392
		# e3a
		# e3, e4 e5

		name = 'model_normalized_16k_2024_september_30.obj'
		model_path = os.path.join(folder, 'models', name)
		if os.path.exists(model_path):

			if not save:
				scene = trimesh.load(model_path)
				data = scene.vertices

			else:
				mesh = o3d.io.read_triangle_mesh(model_path)
				
				data = mesh.sample_points_uniformly(number_of_points=num_points)

				data = np.asarray(data.points)
			
			label = 0

			all_data.append([data])

			if save:
				print('should save.')

			all_label.append([label])

		  
	all_data = np.concatenate(all_data, axis=0)
	#all_label = np.concatenate(all_label, axis=0)
	 
	return all_data, np.array(all_label)
# ------------------------------------ ShapeNet Dataset -----------------------------------------------------
def load_data_shapenet2(train, use_normals,save=True, random = True):
	x = 12000
	counter = 0
	BASE_DIR = os.path.dirname(os.path.abspath(__file__))

	all_data = []
	all_label = []
	curr_folder = 'Uniformly_sampled_farthest_point_sampling'

	for folder in tqdm(glob.glob(os.path.join('..','data',curr_folder, '*'))):
		name = 'model_normalized.obj'
		model_path = os.path.join(folder, 'models', name)
		if os.path.exists(model_path):
		
			scene = trimesh.load(model_path)
			
			if isinstance(scene, trimesh.Scene):
				# Concatenate vertices from all geometries in the scene
				data = np.concatenate([mesh.vertices for mesh in scene.geometry.values()])
			else:
				# Otherwise, it's a single mesh
				
				data = scene.vertices

			if len(data[:]) < x:
				continue


			label = 0

			if random:

				if len(data[:]) < x:
					continue

				kdline_fps_samples_idx  = fpsample.bucket_fps_kdline_sampling(data, x, h=9)

				data = data[kdline_fps_samples_idx]

				all_data.append([data])

			else:
				all_data.append([data[:x]])
				
			all_label.append([label])
			
			counter += 1
			if counter % 100 == 0:
				print("100 done.")
				print(np.size(data,0))
				print(data)


	all_data = np.concatenate(all_data, axis=0)
	#all_label = np.concatenate(all_label, axis=0)
	 
	return all_data, np.array(all_label)



def load_data_shapenet2(train, use_normals,save=False, random = False):
	x = 12000
	BASE_DIR = os.path.dirname(os.path.abspath(__file__))

	all_data = []
	all_label = []
	curr_folder = 'uniformly_sampled_12000'
	counter = 0
	for folder in tqdm(glob.glob(os.path.join('..','data',curr_folder, '*'))):
		name = 'model_normalized_down_sampled_12k.obj'
		model_path = os.path.join(folder, 'models', name)
		if os.path.exists(model_path):
		
			scene = trimesh.load(model_path)
			
			if isinstance(scene, trimesh.Scene):
				# Concatenate vertices from all geometries in the scene
				data = np.concatenate([mesh.vertices for mesh in scene.geometry.values()])
			else:
				# Otherwise, it's a single mesh
				
				data = scene.vertices

			if len(data[:]) < x:
				continue


			label = 0
			if random:

				if isinstance(scene, trimesh.Scene):
					# Simplify each mesh in the scene individually
					factor = 10000 / len(data[:])

					simplified_data = []
					for mesh in scene.geometry.values():

						target_vertices = int(len(mesh.vertices) * factor)
						
						# Estimate the number of faces proportional to the target vertices
						target_faces = int(len(mesh.faces) * (target_vertices / len(mesh.vertices)))


						simplified_mesh = mesh.simplify_quadric_decimation(face_count = target_faces)
						simplified_data.append(simplified_mesh.vertices)


					data = np.concatenate(simplified_data)
				else:
					# Simplify the single mesh
					simplified_mesh = scene.simplify_quadratic_decimation(face_count = x)
					data = simplified_mesh.vertices

				if len(data[:]) < x:
					continue

				all_data.append([data[:x]])
			else:
				all_data.append([data[:x]])
				
			all_label.append([label])
			
			counter+=1
			if counter % 100 == 0:
				print("100 done.")
				print(np.size(data))
		  
	all_data = np.concatenate(all_data, axis=0)
	#all_label = np.concatenate(all_label, axis=0)
	 
	return all_data, np.array(all_label)


class ShapeNet(Dataset):
	def __init__(
		self,
		train=True,
		num_points=16384,
		randomize_data=False,
		use_normals=False
	):
		super(ShapeNet, self).__init__()
		self.data, self.labels = load_data_shapenet(train, use_normals)

		if not train: self.shapes = self.read_classes_ShapeNet()
		self.num_points = num_points
		self.randomize_data = randomize_data

	def __getitem__(self, idx):
		if self.randomize_data: current_points = self.randomize(idx)
		else: current_points = self.data[idx].copy()
		current_points = torch.from_numpy(current_points[:self.num_points, :]).float()
		label = torch.from_numpy(self.labels[idx]).type(torch.LongTensor)

		return current_points, label

	def __len__(self):
		return self.data.shape[0]

	def randomize(self, idx):
		pt_idxs = np.arange(0, self.num_points)
		np.random.shuffle(pt_idxs)
		return self.data[idx, pt_idxs].copy()

	def get_shape(self, label):
		return self.shapes[label]

	def read_classes_ShapeNet(self):
		shape_names = 'ship'
		return shape_names

# ------------------------------------------------------------------------------







# ----------------------------------------------------------------------------- This is the UserData part ------------------------------------------------------------



class ClassificationData:
	def __init__(self, data_dict):
		self.data_dict = data_dict
		print("asdsad")
		self.pcs = self.find_attribute('pcs')
		self.labels = self.find_attribute('labels')
		self.check_data()

	def find_attribute(self, attribute):
		try:
			attribute_data = self.data_dict[attribute]
		except:
			print("Givenn data directory has no key attribute \"{}\"".format(attribute))
		return attribute_data

	def check_data(self):
		assert 1 < len(self.pcs.shape) < 4, "Error in dimension of point clouds! Given data dimension: {}".format(self.pcs.shape)
		assert 0 < len(self.labels.shape) < 3, "Error in dimension of labels! Given data dimension: {}".format(self.labels.shape)
		
		if len(self.pcs.shape)==2: self.pcs = self.pcs.reshape(1, -1, 3)
		if len(self.labels.shape) == 1: self.labels = self.labels.reshape(1, -1)

		assert self.pcs.shape[0] == self.labels.shape[0], "Inconsistency in the number of point clouds and number of ground truth labels!"


	def __len__(self):
		return self.pcs.shape[0]

	def __getitem__(self, index):
		#return torch.tensor(self.pcs[index]).float(), torch.from_numpy(self.labels[idx]).type(torch.LongTensor) ----------------------------------------------------
		return torch.tensor(self.pcs[index]).float(), torch.from_numpy(self.labels[index]).type(torch.LongTensor)


class RegistrationData:
	def __init__(self, data_dict):
		self.data_dict = data_dict
		self.template = self.find_attribute('template')
		self.source = self.find_attribute('source')
		self.transformation = self.find_attribute('transformation')
		self.check_data()

	def find_attribute(self, attribute):
		try:
			attribute_data = self.data[attribute]
		except:
			print("Given data directory has no key attribute \"{}\"".format(attribute))
		return attribute_data

	def check_data(self):
		assert 1 < len(self.template.shape) < 4, "Error in dimension of point clouds! Given data dimension: {}".format(self.template.shape)
		assert 1 < len(self.source.shape) < 4, "Error in dimension of point clouds! Given data dimension: {}".format(self.source.shape)
		assert 1 < len(self.transformation.shape) < 4, "Error in dimension of transformations! Given data dimension: {}".format(self.transformation.shape)

		if len(self.template.shape)==2: self.template = self.template.reshape(1, -1, 3)
		if len(self.source.shape)==2: self.source = self.source.reshape(1, -1, 3)
		if len(self.transformation.shape) == 2: self.transformation = self.transformation.reshape(1, 4, 4)

		assert self.template.shape[0] == self.source.shape[0], "Inconsistency in the number of template and source point clouds!"
		assert self.source.shape[0] == self.transformation.shape[0], "Inconsistency in the number of transformation and source point clouds!"

	def __len__(self):
		return self.template.shape[0]

	def __getitem__(self, index):
		return torch.tensor(self.template[index]).float(), torch.tensor(self.source[index]).float(), torch.tensor(self.transformation[index]).float()


class FlowData:
	def __init__(self, data_dict):
		self.data_dict = data_dict
		self.frame1 = self.find_attribute('frame1')
		self.frame2 = self.find_attribute('frame2')
		self.flow = self.find_attribute('flow')
		self.check_data()

	def find_attribute(self, attribute):
		try:
			attribute_data = self.data[attribute]
		except:
			print("Given data directory has no key attribute \"{}\"".format(attribute))
		return attribute_data

	def check_data(self):
		assert 1 < len(self.frame1.shape) < 4, "Error in dimension of point clouds! Given data dimension: {}".format(self.frame1.shape)
		assert 1 < len(self.frame2.shape) < 4, "Error in dimension of point clouds! Given data dimension: {}".format(self.frame2.shape)
		assert 1 < len(self.flow.shape) < 4, "Error in dimension of flow! Given data dimension: {}".format(self.flow.shape)

		if len(self.frame1.shape)==2: self.frame1 = self.frame1.reshape(1, -1, 3)
		if len(self.frame2.shape)==2: self.frame2 = self.frame2.reshape(1, -1, 3)
		if len(self.flow.shape) == 2: self.flow = self.flow.reshape(1, -1, 3)

		assert self.frame1.shape[0] == self.frame2.shape[0], "Inconsistency in the number of frame1 and frame2 point clouds!"
		assert self.frame2.shape[0] == self.flow.shape[0], "Inconsistency in the number of flow and frame2 point clouds!"

	def __len__(self):
		return self.frame1.shape[0]

	def __getitem__(self, index):
		return torch.tensor(self.frame1[index]).float(), torch.tensor(self.frame2[index]).float(), torch.tensor(self.flow[index]).float()


class UserData:
	def __init__(self, application, data_dict):
		self.application = application

		if self.application == 'classification':
			self.data_class = ClassificationData(data_dict)
		elif self.application == 'registration':
			self.data_class = RegistrationData(data_dict)
		elif self.application == 'flow_estimation':
			self.data_class = FlowData(data_dict)

	def __len__(self):
		return len(self.data_class)

	def __getitem__(self, index):
		return self.data_class[index]


# -------------------------------------------------------------------- Here starts the test code: ---------------------------------------------------

# Only if the files are in example folder.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR[-8:] == 'examples':
	sys.path.append(os.path.join(BASE_DIR, os.pardir))
	os.chdir(os.path.join(BASE_DIR, os.pardir))
	
from learning3d.models import PCN
from learning3d.data_utils import ModelNet40Data, ClassificationData
from learning3d.losses import ChamferDistanceLoss

def display_open3d(input_pc, output):
	input_pc_ = o3d.geometry.PointCloud()
	output_ = o3d.geometry.PointCloud()

	input_pc_.points = o3d.utility.Vector3dVector(input_pc)
	output_.points = o3d.utility.Vector3dVector(output + np.array([1,0,0]))
	input_pc_.paint_uniform_color([1, 0, 0]) # This is red probably for input
	output_.paint_uniform_color([0, 1, 0]) # This is the green for output
	o3d.visualization.draw_geometries([input_pc_, output_])

def display_original(input_pc):
	input_pc_ = o3d.geometry.PointCloud()

	input_pc_.points = o3d.utility.Vector3dVector(input_pc)
	input_pc_.paint_uniform_color([0, 0, 1])
	o3d.visualization.draw_geometries([input_pc_])



# ----------------------------------------------------------------------------- cutting test set with planes -----------------------------------------------------------
def cut(points, plane_normal, plane_point, cut_direction):
    # Subtract plane_point from all points
    vec_to_points = points - plane_point.unsqueeze(0).expand_as(points)

    # Calculate distances from the plane
    distances = torch.sum(vec_to_points * plane_normal.unsqueeze(0).expand_as(points), dim=-1)

    # Determine which side of the plane to keep based on cut_direction
    if cut_direction == 'l':  # 'l' for left side
        mask = distances >= 0
    elif cut_direction == 'r':  # 'r' for right side
        mask = distances <= 0
    else:
        raise ValueError("Invalid cut direction. Use 'l' for left or 'r' for right.")

    # Initialize an array to store the filtered points, keeping the same shape
    filtered_points = torch.zeros_like(points)

    # Copy points that meet the criteria
    filtered_points[mask] = points[mask]

    return filtered_points



def test_one_epoch2(device, model, test_loader,original_loader,own_data,own_data_vis):
	model.eval()
	test_loss = 0.0
	pred  = 0.0
	count = 0

	for i, data in enumerate(tqdm(test_loader)):

		for j, original_data in enumerate(tqdm(original_loader)):
			if j==i:
				original_vis,_ = original_data
				original_vis = original_vis.to(device)
				break

		
		# -------------- original part -----------------
		points, _ = data
		points = points[0,:,:]

		points = points.to(device)
		#sad = own_data.to(device)

		orig_points = points
		
		print(list(points.size()))
		
		# ---------------------------------------------------------- this part is for cutting -------------------------------------------------------
		plane_normal = torch.tensor([1.0, 1.0, 1.0], device='cuda:0')
		plane_point = torch.tensor([0.0, 0.0, 0.0], device='cuda:0') 
		
		#plane_normal = np.array([1.0, 1.0, 1.0])
		#plane_point = np.array([0.0, 0.0, 0.0]) 
		
		cut_direction = 'l'
		
		filtered_points = cut(points, plane_normal,plane_point,cut_direction)
#		filtered_points = torch.reshape(filtered_points, (list(points.size())[0], list(filtered_points.size())[1],3))

		filtered_points = torch.reshape(filtered_points, (1, 2048,3))

		non_zero_elements = []
		for line in filtered_points:
			non_zero_elements = []
			for element in line:
				if not torch.all(torch.eq(element, torch.tensor([0.], device=element.device))):
					x = element.cpu().numpy()[0]
					y = element.cpu().numpy()[1]
					z = element.cpu().numpy()[2]
					non_zero_elements.append(x)
					non_zero_elements.append(y)
					non_zero_elements.append(z)
					#print(element.cpu().numpy())
		print('§')
		keep = (len(non_zero_elements)/3)%32
		non_zero_elements = non_zero_elements[0:-int((keep*3))]

		vis = np.reshape(non_zero_elements, [int(len(non_zero_elements)/3),3])
		non_zero_elements = np.reshape(non_zero_elements, [32,int(len(non_zero_elements)/3/32),3])
		
	
		filtered_points = torch.as_tensor(non_zero_elements,device='cuda:0') 

		points = filtered_points
		output = model(points)
		#output = model(own_data)

		loss_val = ChamferDistanceLoss()(points, output['coarse_output'])
		print("Loss Val: ", loss_val)
		
		display_open3d(vis, output['coarse_output'][0].detach().cpu().numpy())
		#display_open3d(own_data_vis, output['coarse_output'][0].detach().cpu().numpy())


		#display_original(original_vis[0].detach().cpu().numpy())
		#display_open3d(points[0].detach().cpu().numpy(), output['coarse_output'][0].detach().cpu().numpy())
		display_original(orig_points.detach().cpu().numpy()) # ez a jo!!!!!!!!!!!!!!!!! ha partial
		#display_original(vis)



		test_loss += loss_val.item()
		count += 1

	test_loss = float(test_loss)/count
	return test_loss


#............................................................................................................................................................................................above this is not orignal below yes.....................................

def test_one_epoch33333(device, model, test_loader):
	model.eval()
	test_loss = 0.0
	pred  = 0.0
	count = 0

	for i, data in enumerate(tqdm(test_loader)):
		points, _ = data

		points = points.to(device)

		output = model(points)
		loss_val = ChamferDistanceLoss()(points, output['coarse_output'])
		print("Loss Val: ", loss_val)
		print('Input size:  ' + str(len(points[0].detach().cpu().numpy())))
		print('Output size:  ' + str(len(output['coarse_output'][0].detach().cpu().numpy())))
		display_open3d(points[0].detach().cpu().numpy(), output['coarse_output'][0].detach().cpu().numpy())

		test_loss += loss_val.item()
		count += 1

	test_loss = float(test_loss)/count
	return test_loss


def test_one_epoch(device, model, test_loader):
	model.eval()
	test_loss = 0.0
	pred  = 0.0
	count = 0
	for i, data in enumerate(tqdm(test_loader)):
		points, _ = data

		points = points.to(device)

		output = model(points)
		loss_val = ChamferDistanceLoss()(points, output['coarse_output'])
		print("Loss Val: ", loss_val)
		display_open3d(points[0].detach().cpu().numpy(), output['coarse_output'][0].detach().cpu().numpy())

		test_loss += loss_val.item()
		count += 1

	test_loss = float(test_loss)/count
	return test_loss

def test(args, model, test_loader):
	test_loss = test_one_epoch(args.device, model, test_loader)

def test2(args, model, test_loader,original_loader,own_data,own_data_vis):
	test_loss = test_one_epoch2(args.device, model, test_loader,original_loader,own_data,own_data_vis)

def options():
	parser = argparse.ArgumentParser(description='Point Completion Network')
	parser.add_argument('--exp_name', type=str, default='exp_pcn', metavar='N',
						help='Name of the experiment')
	parser.add_argument('--dataset_path', type=str, default='ModelNet40',
						metavar='PATH', help='path to the input dataset') # like '/path/to/ModelNet40'
	parser.add_argument('--eval', type=bool, default=False, help='Train or Evaluate the network.')

	# settings for input data
	parser.add_argument('--dataset_type', default='modelnet', choices=['modelnet', 'shapenet2'],
						metavar='DATASET', help='dataset type (default: modelnet)')
	parser.add_argument('--num_points', default=16384, type=int,
						metavar='N', help='points in point-cloud (default: 1024)')

	# settings for PCN
	parser.add_argument('--emb_dims', default=16384, type=int,
						metavar='K', help='dim. of the feature vector (default: 1024)')
	parser.add_argument('--detailed_output', default=False, type=bool,
						help='Coarse + Fine Output')

	# settings for on training
	parser.add_argument('--seed', type=int, default=1234)
	parser.add_argument('-j', '--workers', default=4, type=int,
						metavar='N', help='number of data loading workers (default: 4)')
	parser.add_argument('-b', '--batch_size', default=32, type=int,
						metavar='N', help='mini-batch size (default: 32)')
	parser.add_argument('--pretrained', default='checkpoints/exp_pcn/models/best_model_16k_2024_september_30.t7', type=str, 
						metavar='PATH', help='path to pretrained model file (default: null (no-use))')  # This part i set by me there was learning3d/ instead of ../
	parser.add_argument('--device', default='cuda:0', type=str,
						metavar='DEVICE', help='use CUDA if available')

	args = parser.parse_args()
	return args


import open3d as o3d

def read_ply_file(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)
    return points

def main():
	
	args = options()

	args.dataset_path = os.path.join(os.getcwd(), os.pardir, os.pardir, 'ModelNet40', 'ModelNet40')
	'''

	trainset = ClassificationData(ModelNet40Data(train=True,root_dir='../'))
	testset = ClassificationData(ModelNet40Data(train=False,root_dir='..',num_points=10000))

	original = ClassificationData(ModelNet40Data(train=False,root_dir='..',num_points=2048))
    
	# ------------------------------------------------------------ setting up own data ---------------------------------------------------
	data = np.loadtxt("./Agoston_repcsi/NormalizedSparsePointCloud_boat_1.txt")
	data = data.reshape(1, -1)
	data = data[0]


	keep = (len(data)/3)%32
	
	if keep > 0:
		data = data[0:-int((keep*3))]
	
	

	own_data_vis = np.reshape(data, [int(len(data)/3),3])
	data = np.reshape(data, [32,int(len(data)/3/32),3])
	
	
	own_data = torch.as_tensor(data,device='cuda:0') 

	this_dict = {
	"pcs": np.array([[data],[data]]),
	"labels": np.array([[1.0],[1.0]])
	}
	'''

	#testset = ClassificationData(UserData('classification',data_dict=this_dict))
	
	CommonSet = ClassificationData(ShapeNet(num_points=16384, train=True))

	train_size = int(0.8 * len(CommonSet)) 
	test_size = len(CommonSet) - train_size 

	train_dataset, test_dataset = random_split(CommonSet, [train_size, test_size])

#	train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.workers)
#	test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.workers)
#	original_loader = DataLoader(original, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.workers)

	if not torch.cuda.is_available():
		print('No CUDA')
		args.device = 'cpu'
	args.device = torch.device(args.device)

	# Create PointNet Model.
	numofpoints = 16384
	model = PCN(emb_dims=512, num_coarse=numofpoints, detailed_output=False)

	if args.pretrained:
		assert os.path.isfile(args.pretrained)

		model.load_state_dict(torch.load(args.pretrained, map_location='cpu'))

	model.to(args.device)

	#test(args, model, test_loader,original_loader,own_data.float(),own_data_vis)
	#testset = ClassificationData(ShapeNet(train=False,num_points=10000))
	test_loader = DataLoader(CommonSet, batch_size=1)

	test(args, model, test_loader)
if __name__ == '__main__':
	main()