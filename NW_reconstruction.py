import pickle
import torch

import gradoptics as optics
from gradoptics.integrator import HierarchicalSamplingIntegrator
from ml.siren import Siren

import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# import scene and targets from setup notebook
scene_objects = pickle.load(open("NW_mot_scene_components.pkl", "rb"))
targets = pickle.load(open("NW_mot_images.pkl", "rb"))

# Adding a mask to focus on relevant bit of images
sel_mask = torch.ones(targets.shape[1:], dtype=torch.bool)
sel_mask[:250] = 0
sel_mask[1750:] = 0
sel_mask[:, 1500:] = 0
targets = targets.flatten(start_dim=1).cuda()

# set up model -- here we use SIREN
device = 'cuda'
in_features = 3
hidden_features = 256
hidden_layers = 3
out_features = 1

model = Siren(in_features, hidden_features, hidden_layers, out_features,
              outermost_linear=True, 
              outermost_linear_activation=nn.ReLU()).double().to(device)

# model = torch.nn.DataParallel(model)

# set up scene used for rendering in training
# samples from a neural net for a light source
# Region we want to integrate in + position
rad = 0.03
obj_pos = (0, 0, 0)

light_source = optics.LightSourceFromNeuralNet(model, 
                                               optics.BoundingSphere(radii=rad, 
                                                                     xc=obj_pos[0], yc=obj_pos[1], zc=obj_pos[2]),
                                                                     rad=rad, 
                                                                     x_pos=obj_pos[0], y_pos=obj_pos[1], z_pos=obj_pos[2])
scene_train = optics.Scene(light_source)

for obj in scene_objects:
    scene_train.add_object(obj)

# train
sensor_list = [obj for obj in scene_train.objects if type(obj) == optics.Sensor]
lens_list = [obj for obj in scene_train.objects if type(obj) == optics.PerfectLens]

batch_size = 512
loss_fn = torch.nn.MSELoss()
integrator = HierarchicalSamplingIntegrator(64, 64)
optimizer = torch.optim.Adam(scene_train.light_source.network.parameters(), lr=1e-4)

losses = []

camera_list = torch.randperm(len(targets))
for i_iter in tqdm(range(100000)): 
    # Grab camera image
    rand_data_id = camera_list[i_iter % len(targets)]
    sensor_here = sensor_list[rand_data_id]
    lens_here = lens_list[rand_data_id]
    
    h_here, w_here = sensor_here.resolution
    
    # Grab masked pixel indices + sample randomly
    idxs_all = torch.cartesian_prod(torch.arange(h_here//2, -h_here//2, -1), 
                                    torch.arange(w_here//2, -w_here//2, -1))
    
    idxs_all = idxs_all[sel_mask.flatten()]
    
    rand_pixels = torch.randint(0, len(idxs_all), (batch_size,))
    target_vals = targets[rand_data_id][sel_mask.flatten()][rand_pixels]

    
    batch_pix_x = idxs_all[rand_pixels, 0]
    batch_pix_y = idxs_all[rand_pixels, 1]

    # Render image from neural network light source
    intensities = optics.ray_tracing.ray_tracing.render_pixels(sensor_here, 
                                                  lens_here, 
                                                 scene_train, scene_train.light_source, 1, 5, 
                                                 batch_pix_x, batch_pix_y,
                                                 integrator, device='cuda',max_iterations=6)
    

    # Scaling to help control loss values
    im_scale = targets[rand_data_id][sel_mask.flatten()].mean().item()
    
    # Calculate loss and update neural network parameters
    loss = loss_fn(intensities/im_scale*1e5, target_vals.double().cuda()/im_scale)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Record and print out
    losses.append(loss.item())

    if i_iter % 500 == 0:
        with torch.no_grad():
            torch.save(scene_train.light_source.network.state_dict(), 
                       f'model_{i_iter}_NW_MOT_all_cameras_long.pt')
from numpy import savetxt            
savetxt('loss.csv',losses,delimiter=',')
            
