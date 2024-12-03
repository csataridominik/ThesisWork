# author: Vinit Sarode (vinitsarode5@gmail.com) 03/23/2020

import argparse
import os
import sys
import logging
import numpy as np
import torch
import torch.utils.data
import torchvision
from torch.utils.data import DataLoader, random_split
from tensorboardX import SummaryWriter
from tqdm import tqdm
from learning3d.data_utils import UserData

import random

import open3d as o3d

import fpsample # ---------------------------------------------------- Chooses Farthest points

from torch.utils.data import Dataset
import glob
import trimesh



def save_obj_file(filename, vertices):
    with open(filename, 'w') as file:
        # Write vertices
        for vertex in vertices:
            file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")


#------------------------------------ ShapeNet Dataset -----------------------------------------------------
def load_data_shapenet(train, use_normals,save=False, random = False):
	x = 12000
	counter = 0
	BASE_DIR = os.path.dirname(os.path.abspath(__file__))

	all_data = []
	all_label = []
	curr_folder = 'Uniformly_sampled_farthest_point_sampling'

	for folder in tqdm(glob.glob(os.path.join('..','data',curr_folder, '*'))):
		name = 'model_normalized_farthest_point_sampling.obj'
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
				if save:
					filename = os.path.join(folder,'models', 'model_normalized_farthest_point_sampling.obj')
					save_obj_file(filename,data)
			else:
				all_data.append([data[:]])
				
			all_label.append([label])
			
			counter += 1
			if counter % 100 == 0:
				print("100 done.")
				print(np.size(data,0))
				print(data)


	all_data = np.concatenate(all_data, axis=0)
	#all_label = np.concatenate(all_label, axis=0)
	 
	return all_data, np.array(all_label)

class ShapeNet(Dataset):
	def __init__(
		self,
		train=True,
		num_points=12000,
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


# Only if the files are in example folder.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR[-8:] == 'examples':
	sys.path.append(os.path.join(BASE_DIR, os.pardir))
	os.chdir(os.path.join(BASE_DIR, os.pardir))
	
from learning3d.models import PCN
from learning3d.losses import ChamferDistanceLoss
from learning3d.data_utils import ClassificationData, ModelNet40Data

def _init_(args):
	if not os.path.exists('checkpoints'):
		os.makedirs('checkpoints')
	if not os.path.exists('checkpoints/' + args.exp_name):
		os.makedirs('checkpoints/' + args.exp_name)
	if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
		os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')
	os.system('cp train_pcn.py checkpoints' + '/' + args.exp_name + '/' + 'main.py.backup')


class IOStream:
	def __init__(self, path):
		self.f = open(path, 'a')

	def cprint(self, text):
		print(text)
		self.f.write(text + '\n')
		self.f.flush()

	def close(self):
		self.f.close()

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

		test_loss += loss_val.item()
		count += 1

	test_loss = float(test_loss)/count
	return test_loss

def test(args, model, test_loader, textio):
	test_loss = test_one_epoch(args.device, model, test_loader)
	textio.cprint('Validation Loss: %f'%(test_loss))

def train_one_epoch(device, model, train_loader, optimizer):
	model.train()
	train_loss = 0.0
	pred  = 0.0
	count = 0
	for i, data in enumerate(tqdm(train_loader)):
		
		points, _ = data
		points = points.to(device)
		

		output = model(points)
		loss_val = ChamferDistanceLoss()(points, output['coarse_output'])

		# backward + optimize
		optimizer.zero_grad()
		loss_val.backward()
		optimizer.step()

		train_loss += loss_val.item()
		count += 1

	train_loss = float(train_loss)/count
	return train_loss

def train(args, model, train_loader, test_loader, boardio, textio, checkpoint):
	learnable_params = filter(lambda p: p.requires_grad, model.parameters())
	if args.optimizer == 'Adam':
		optimizer = torch.optim.Adam(learnable_params)
	else:
		optimizer = torch.optim.SGD(learnable_params, lr=0.1)

	if checkpoint is not None:
		min_loss = checkpoint['min_loss']
		optimizer.load_state_dict(checkpoint['optimizer'])

	best_test_loss = np.inf

	for epoch in range(args.start_epoch, args.epochs):
		train_loss = train_one_epoch(args.device, model, train_loader, optimizer)
		test_loss = test_one_epoch(args.device, model, test_loader)

		if test_loss<best_test_loss:
			best_test_loss = test_loss
			snap = {'epoch': epoch + 1,
					'model': model.state_dict(),
					'min_loss': best_test_loss,
					'optimizer' : optimizer.state_dict(),}
			torch.save(snap, 'checkpoints/%s/models/best_model_snap_12k_2.t7' % (args.exp_name))
			torch.save(model.state_dict(), 'checkpoints/%s/models/best_model_12k_2.t7' % (args.exp_name))

		torch.save(snap, 'checkpoints/%s/models/model_snap2.t7' % (args.exp_name))
		torch.save(model.state_dict(), 'checkpoints/%s/models/model_12k_2.t7' % (args.exp_name))
		
		boardio.add_scalar('Train Loss', train_loss, epoch+1)
		boardio.add_scalar('Test Loss', test_loss, epoch+1)
		boardio.add_scalar('Best Test Loss', best_test_loss, epoch+1)

		textio.cprint('EPOCH:: %d, Traininig Loss: %f, Testing Loss: %f, Best Loss: %f'%(epoch+1, train_loss, test_loss, best_test_loss))

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
	parser.add_argument('-j', '--workers', default=16, type=int,
						metavar='N', help='number of data loading workers (default: 4)')
	parser.add_argument('-b', '--batch_size', default=4, type=int,
						metavar='N', help='mini-batch size (default: 32)')
	parser.add_argument('--epochs', default=150, type=int,
						metavar='N', help='number of total epochs to run')
	parser.add_argument('--start_epoch', default=0, type=int,
						metavar='N', help='manual epoch number (useful on restarts)')
	parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'],
						metavar='METHOD', help='name of an optimizer (default: Adam)')
	parser.add_argument('--resume', default='', type=str,
						metavar='PATH', help='path to latest checkpoint (default: null (no-use))')
	parser.add_argument('--pretrained', default='', type=str,
						metavar='PATH', help='path to pretrained model file (default: null (no-use))')
	parser.add_argument('--device', default='cuda:0', type=str,
						metavar='DEVICE', help='use CUDA if available')

	args = parser.parse_args()
	return args

import numpy as np
import trimesh
import glob
import os
import shutil



def delete(x=16384):
    curr_folder = 'ships_greater_16384'
    for folder in tqdm(glob.glob(os.path.join('..','data',curr_folder, '*'))):
        model_path = os.path.join( folder, 'models', 'model_normalized.obj')
        if os.path.exists(model_path):
        
            scene = trimesh.load(model_path)
            if isinstance(scene, trimesh.Scene):
                # Concatenate vertices from all geometries in the scene
                data = np.concatenate([mesh.vertices for mesh in scene.geometry.values()])
            else:
                # Otherwise, it's a single mesh
                
                data = scene.vertices

            if len(data[:]) < x:
                shutil.rmtree(folder)




def main():
	# This is to delete .obj files having les than x points
	#delete(x)

	
	torch.cuda.empty_cache() # empty cache ---------------------------------------------------------------------------------

	numofpoints = 12000

	args = options()

	torch.backends.cudnn.deterministic = True
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)
	np.random.seed(args.seed)

	boardio = SummaryWriter(log_dir='checkpoints/' + args.exp_name)
	_init_(args)

	textio = IOStream('checkpoints/' + args.exp_name + '/run.log')
	textio.cprint(str(args))

	CommonSet = ClassificationData(ShapeNet(num_points=numofpoints, train=True))

	train_size = int(0.8 * len(CommonSet)) 
	test_size = len(CommonSet) - train_size 

	train_dataset, test_dataset = random_split(CommonSet, [train_size, test_size])

	train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.workers)
	test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.workers)

	if not torch.cuda.is_available():
		args.device = 'cpu'
	args.device = torch.device(args.device)

	# Create PointNet Model.
	model = PCN(emb_dims=512, num_coarse=numofpoints, detailed_output=False)

	checkpoint = None
	if args.resume:
		assert os.path.isfile(args.resume)
		checkpoint = torch.load(args.resume)
		args.start_epoch = checkpoint['epoch']
		model.load_state_dict(checkpoint['model'])

	if args.pretrained:
		assert os.path.isfile(args.pretrained)
		model.load_state_dict(torch.load(args.pretrained, map_location='cpu'))
	model.to(args.device)
	
	
	if args.eval:
		test(args, model, test_loader, textio)
	else:
		print('Training starts.')
		train(args, model, train_loader, test_loader, boardio, textio, checkpoint)
	
if __name__ == '__main__':
	print('Load Data.')
	main()