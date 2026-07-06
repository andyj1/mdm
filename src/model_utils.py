import random
import torch
import math 
import numpy as np
import copy 
import glob
import os
from rich import print
try:
	from src.networks import CLIPModel_full
except:
	from networks import CLIPModel_full
	
import copy
import torch

import collections
from math import sqrt
EPS = 1e-8

def compute_ratio(angle_dict, k=2):
	ratio_dict = {}
	for key in angle_dict.keys():
		angle = np.deg2rad(angle_dict[key])
		ratio_dict[key] = k*np.cos(angle) / ((k-1)*np.cos(angle)+1+EPS)

	return ratio_dict 

def compute_angle(state_dict_1, state_dict_2, ref_state_dict, add_ignore_keys=[], return_cos=False, device='cuda'):
	ignore_keys = []
	return_dict = collections.OrderedDict()

	with torch.no_grad():
		for key in ref_state_dict:
			if key in ignore_keys:
				continue

			state_dict_1_val = state_dict_1[key]            
			state_dict_2_val = state_dict_2[key]                        
			ref_val = ref_state_dict[key]

			if not (state_dict_1_val.shape == state_dict_2_val.shape == ref_val.shape):
				continue 

			vector1 = (state_dict_1_val.to(device) - ref_val.to(device)).clone().detach()
			vector2 = (state_dict_2_val.to(device) - ref_val.to(device)).clone().detach()

			vector1 = vector1.float()
			vector2 = vector2.float()

			cosine_val = torch.sum(vector1 * vector2) / (sqrt(torch.sum(vector1 ** 2) * torch.sum(vector2 ** 2))+EPS)
			cosine_val = torch.clamp(cosine_val, min=-1., max=1.) # Too prevent nan from acos
   
			cosine_val = cosine_val.to('cpu')
			if return_cos:
				return_dict[key] = cosine_val 
			else:
				return_dict[key] = np.rad2deg(torch.acos(cosine_val).detach().cpu())

	return return_dict

@torch.inference_mode()
def merge(w1, w2, w0, ratio, device='cuda', dtype=torch.float32, non_blocking=True):
    w_merge = {}

    for k in w0.keys():
        if (k in w1) and (k in w2) \
           and torch.is_tensor(w1[k]) and torch.is_tensor(w2[k]) and torch.is_tensor(w0[k]) \
           and w1[k].is_floating_point() and w2[k].is_floating_point() and w0[k].is_floating_point() \
           and (w1[k].shape == w2[k].shape == w0[k].shape):

            t1 = w1[k].to(device=device, dtype=dtype, non_blocking=non_blocking)
            t2 = w2[k].to(device=device, dtype=dtype, non_blocking=non_blocking)
            t0 = w0[k].to(device=device, dtype=dtype, non_blocking=non_blocking)

            w12 = 0.5 * (t1 + t2)  # 평균

            outk = w12

            if k in ratio:
                r = torch.as_tensor(ratio[k], dtype=dtype, device=device)
                outk = t0 + r * (w12 - t0)

            w_merge[k] = outk

        else:
            w_merge[k] = w0[k]

    return w_merge

def merge_expert_buffers(expert1_state_dict, expert2_state_dict, initial_state_dict):
	angle = compute_angle(expert1_state_dict, expert2_state_dict, initial_state_dict)
	ratio = compute_ratio(angle)
	merged_state_dict = merge(expert1_state_dict, expert2_state_dict, initial_state_dict, ratio)
	return merged_state_dict

def load_model_state_dict(state_dict, map_location='cpu'):
	state_dict = torch.load(state_dict, map_location=map_location)
	return state_dict

def _load_expert_buffers(img_expert_files, txt_expert_files):
	import random
	NUM_RANDOM_EXPERTS = 2
	NUM_RANDOM_EPOCHS = 2
	random_experts = random.sample(range(len(img_expert_files)), NUM_RANDOM_EXPERTS)
	random_epochs = random.sample(range(len(img_expert_files[0])), NUM_RANDOM_EPOCHS)
	
	random_experts, random_epochs = [random.randint(0, 19), random.randint(0, 19)], [random.randint(1, 10), random.randint(1, 10)]
	
	img_expert_1 = load_model_state_dict(img_expert_files[random_experts[0]]) # per expert iteration
	txt_expert_1 = load_model_state_dict(txt_expert_files[random_experts[0]]) # per expert iteration
	img_expert_1 = img_expert_1[random_epochs[0]] # per epoch
	txt_expert_1 = txt_expert_1[random_epochs[0]] # per epoch
	
	img_expert_2 = load_model_state_dict(img_expert_files[random_experts[1]]) # per expert iteration
	txt_expert_2 = load_model_state_dict(txt_expert_files[random_experts[1]]) # per expert iteration
	img_expert_2 = img_expert_2[random_epochs[1]] # per epoch
	txt_expert_2 = txt_expert_2[random_epochs[1]] # per epoch

	return img_expert_1, txt_expert_1, img_expert_2, txt_expert_2
	
def make_distillation_model(args, student_net, base_dir='./buffer_lors/flickr8k/nfnet_bert/InfoNCE', file_format='replay_buffer', merge_image=True, merge_text=True, verbose=False):

	BASE_DIR = base_dir
	FILE_FORMAT = file_format
	
	total_img_buffers = 19 #len(img_expert_files)-1

	FIX_EXPERT_VARY_EPOCH, VARY_EXPERT_VARY_EPOCH = False, True
	# FIX_EXPERT_VARY_EPOCH, VARY_EPOCH_VARY_EPOCH = True, False
 
	if FIX_EXPERT_VARY_EPOCH:
		# fix expert number, vary epochs
		EXPERT_NUM1 = random.randint(0, total_img_buffers)
	
		MAX_EPOCH = args.max_start_epoch
  
		EPOCH_NUM1 = random.choice(range(1, MAX_EPOCH+1)) # args.max_start_epoch)
		img_file = os.path.join(BASE_DIR, F'img_{FILE_FORMAT}_{EXPERT_NUM1}.pt')
		txt_file = os.path.join(BASE_DIR, F'txt_{FILE_FORMAT}_{EXPERT_NUM1}.pt')
		img_expert_1 = load_model_state_dict(img_file, map_location='cuda')[0][EPOCH_NUM1]
		txt_expert_1 = load_model_state_dict(txt_file, map_location='cuda')[0][EPOCH_NUM1]
		
		# Sample a random number between 1 and 5, excluding the previously selected EPOCH_NUM
		epoch_pool = [i for i in range(1, MAX_EPOCH+1) if i != EPOCH_NUM1]
		EPOCH_NUM2 = random.choice(epoch_pool)
		# EPOCH_NUM = random.randint(1, 5) # args.max_start_epoch)
		img_file = os.path.join(BASE_DIR, F'img_{FILE_FORMAT}_{EXPERT_NUM1}.pt')
		txt_file = os.path.join(BASE_DIR, F'txt_{FILE_FORMAT}_{EXPERT_NUM1}.pt')
		img_expert_2 = load_model_state_dict(img_file, map_location='cuda')[0][EPOCH_NUM2]
		txt_expert_2 = load_model_state_dict(txt_file, map_location='cuda')[0][EPOCH_NUM2]
  	
	elif VARY_EXPERT_VARY_EPOCH:
		EXPERT_NUM1 = random.randint(0, total_img_buffers)
		# EPOCH_NUM = random.randint(1, args.max_start_epoch)
		# EPOCH_NUM1 = random.choice(range(1, 6)) 
		MAX_EPOCH = args.max_start_epoch
  
		EPOCH_NUM1 = random.choice(range(1, MAX_EPOCH+1)) 
		img_file = os.path.join(BASE_DIR, F'img_{FILE_FORMAT}_{EXPERT_NUM1}_{EPOCH_NUM1}.pth')
		txt_file = os.path.join(BASE_DIR, F'txt_{FILE_FORMAT}_{EXPERT_NUM1}_{EPOCH_NUM1}.pth')
		img_expert_1 = load_model_state_dict(img_file, map_location='cuda')
		txt_expert_1 = load_model_state_dict(txt_file, map_location='cuda')
		
		EXPERT_NUM2 = random.randint(0, total_img_buffers)
		# EPOCH_NUM = random.randint(1, args.max_start_epoch)
		# epoch_pool = [i for i in range(1, 6) if i != EPOCH_NUM1]
		epoch_pool = [i for i in range(1, MAX_EPOCH+1) if i != EPOCH_NUM1]
		EPOCH_NUM2 = random.choice(epoch_pool)
		img_file = os.path.join(BASE_DIR, F'img_{FILE_FORMAT}_{EXPERT_NUM1}_{EXPERT_NUM2}.pth')
		txt_file = os.path.join(BASE_DIR, F'txt_{FILE_FORMAT}_{EXPERT_NUM1}_{EXPERT_NUM2}.pth')
		img_expert_2 = load_model_state_dict(img_file, map_location='cuda')
		txt_expert_2 = load_model_state_dict(txt_file, map_location='cuda')

			
	method = 'FIX_EXPERT_VARY_EPOCH' if FIX_EXPERT_VARY_EPOCH else 'VARY_EXPERT_VARY_EPOCH' if VARY_EXPERT_VARY_EPOCH else 'RANDOM_EXPERT_RANDOM_EPOCH'
	print(f'{method} || EXPERT_NUM1: {EXPERT_NUM1}, EPOCH_NUM1: {EPOCH_NUM1} || EXPERT_NUM2: {EXPERT_NUM2}, EPOCH_NUM2: {EPOCH_NUM2}')
 
	
	# check if the expert state dicts are valid
	assert isinstance(img_expert_1, dict) and isinstance(txt_expert_1, dict) and isinstance(img_expert_2, dict) and isinstance(txt_expert_2, dict)
	
	# initial model
	img_initial = student_net.image_encoder.state_dict()
	txt_initial = student_net.text_projection.state_dict()
	
	img_angle = compute_angle(img_expert_1, img_expert_2, img_initial)
	img_ratio = compute_ratio(img_angle)
	# print(img_angle.keys())
	# visualize(img_angle, img_ratio, save_dir='fig/img_angle.png')
	# merge image expert buffers
	if merge_image:
		merged_img_model = merge(img_expert_1, img_expert_2, img_initial, img_ratio, device='cuda')
	
	txt_angle = compute_angle(txt_expert_1, txt_expert_2, txt_initial)
	txt_ratio = compute_ratio(txt_angle)
	# visualize(img_angle, img_ratio, save_dir='fig/img_angle.png')
	
	# merge text expert buffers
	if merge_text:
		merged_txt_model = merge(txt_expert_1, txt_expert_2, txt_initial, txt_ratio, device='cuda')
	
	# Save a copy of the original image encoder state dict
	orig_img_encoder_weights = copy.deepcopy(student_net.image_encoder.state_dict())
	orig_txt_encoder_weights = copy.deepcopy(student_net.text_projection.state_dict())

	# Load merged model as done below (after this block)
	# student_net.image_encoder.load_state_dict(merged_img_model)
	# student_net.text_projection.load_state_dict(merged_txt_model)

	# After loading, compare the parameters
	weights_changed = False
	for key in orig_img_encoder_weights:
		if not torch.equal(orig_img_encoder_weights[key].to('cpu'), merged_img_model[key].data.to('cpu')):
			weights_changed = True
			break
	assert weights_changed, "student_net.image_encoder weights have NOT changed after loading merged_img_model."
 
	# if weights_changed:
	# 	print("student_net.image_encoder weights have been updated by merged_img_model.")
	# else:
	# 	print("WARNING: student_net.image_encoder weights have NOT changed after loading merged_img_model.")

	weights_changed = False
	for key in orig_txt_encoder_weights:
		if not torch.equal(orig_txt_encoder_weights[key].to('cpu'), merged_txt_model[key].data.to('cpu')):
			# print(f"Key: {key}")		
			# print(f"Original weight: {orig_txt_encoder_weights[key].to('cpu')}")
			# print(f"Merged weight: {merged_txt_model[key].data.to('cpu')}")
			weights_changed = True
			break
	assert weights_changed, "student_net.text_projection weights have NOT changed after loading merged_txt_model."
 
	# if weights_changed:
	# 	print("student_net.text_projection weights have been updated by merged_txt_model.")
	# else:
	# 	print("WARNING: student_net.text_projection weights have NOT changed after loading merged_txt_model.")
	
	# Ensure we load the state_dict inplace, allowing keys that are missing or unexpected
	image_encoder_result = student_net.image_encoder.load_state_dict(merged_img_model, strict=False)
	text_proj_result = student_net.text_projection.load_state_dict(merged_txt_model, strict=False)

	# Optionally report the result (could be logged if needed)
	if len(image_encoder_result.missing_keys) > 0 or len(image_encoder_result.unexpected_keys) > 0:
		print(f"Warning when loading image_encoder weights: {image_encoder_result}")
	if len(text_proj_result.missing_keys) > 0 or len(text_proj_result.unexpected_keys) > 0:
		print(f"Warning when loading text_projection weights: {text_proj_result}")
	
	if verbose:
		# return student_net
		return merged_img_model, merged_txt_model, (EXPERT_NUM1, EPOCH_NUM1), (EXPERT_NUM2, EPOCH_NUM2)
	else:
		return merged_img_model, merged_txt_model

if __name__ == "__main__":
	# sanity check
	import glob
	import os
	buffer_path = 'buffers_ours_final/flickr8k/nfnet_bert/InfoNCE'
	img_expert_files = glob.glob(os.path.join(buffer_path, 'img_replay_buffer_*.pt')) # list of filenames
	txt_expert_files = glob.glob(os.path.join(buffer_path, 'txt_replay_buffer_*.pt')) # list of filenames
 
	EXPERT_NUM = random.randint(0, len(img_expert_files)-1) # specified by the count in the filename
	EPOCH_NUM = random.randint(1, len(load_model_state_dict(img_expert_files[EXPERT_NUM])[0])-1)  # ignore first (initial model parameters)
	img_expert = load_model_state_dict(img_expert_files[EXPERT_NUM])[0][EPOCH_NUM]
	txt_expert = load_model_state_dict(txt_expert_files[EXPERT_NUM])[0][EPOCH_NUM]
		
	EXPERT_NUM_2 = random.randint(0, len(img_expert_files)-1) # specified by the count in the filename
	EPOCH_NUM_2 = random.randint(1, len(load_model_state_dict(img_expert_files[EXPERT_NUM_2])[0])-1)  # ignore first (initial model parameters)
	img_expert_2 = load_model_state_dict(img_expert_files[EXPERT_NUM_2])[0][EPOCH_NUM_2]
	txt_expert_2 = load_model_state_dict(txt_expert_files[EXPERT_NUM_2])[0][EPOCH_NUM_2]
	
	assert isinstance(img_expert, dict) and isinstance(txt_expert, dict)
 
	# verify this is the correct state dict to load in CLIPModel_full
	import argparse
	args = argparse.Namespace()
	args.buffer_path = './buffers_ours_final/flickr8k/nfnet_bert/InfoNCE'
	args.image_encoder = 'nfnet'
	args.image_pretrained = True
	args.only_has_image_projection = False
	args.image_trainable = True
	args.text_encoder = 'bert'
	args.text_pretrained = True
	args.text_trainable = False
	args.temperature = 0.07
	args.device = 'cuda'
	args.wandb = False
	args.log_freq = 100
	args.log_dir = 'logs'
	args.model_dir = 'models'
	args.distill = False
	args.data_dir = 'data'
	args.num_queries = 100
	args.mini_batch_size = 128
	args.loss_type = 'WBCE'
	student_net = CLIPModel_full(args, temperature=0.07)
	
	try:
		student_net.image_encoder.load_state_dict(img_expert)
		student_net.text_projection.load_state_dict(txt_expert)
	except:
		raise FileNotFoundError("Incorrect state dict for image encoder or text encoder")    
	
	
	
	
	make_distillation_model(args, student_net, merge_image=True, merge_text=True)
