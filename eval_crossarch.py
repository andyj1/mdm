import glob
import os
import copy
import math
import argparse
import random
import datetime
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard.summary import logger
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
import json 
from collections import defaultdict
import collections

import shutil
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from math import sqrt
from torch.utils.tensorboard import SummaryWriter

# ===== Project/LoRS modules (expected in repo) =====
from data import get_dataloaders
from src.clustering_utils import KMeansCluster, OnlineCentroidBank
from src.epoch import epoch_test, itm_eval, evaluate_synset
from src.networks import CLIPModel_full
from src.utils import DiffAugment, ParamDiffAug
from src.vl_distill_utils import (
	shuffle_files, nearest_neighbor, get_images_texts, load_or_process_file, textprocess_test, textprocess_train
)

import prettytable


'''
python distill_DM_Final_cluster_nce_uMMD_cgMMD_modelmix.py --lr_txt 100 --lr_img 100 --num_queries 100 --teacher_resample 1 --Iteration 3000 --eval_it 100 --num_eval 1 --syn_init random --epoch_eval_train 100 --eval_eval_freq 10 --w_nce 1.0 --w_sph_mmd 5 --w_cgap_x 0.4 --w_cgap_y 0.4 --dataset flickr30k --init_model_method mixed --name flickr30k_MM_nce

'''
	
EPS = 1e-8

@torch.no_grad()
def print_results(multi_eval_aggr_result, title='image-text retrieval results'):
	mean_result, std_result = defaultdict(), defaultdict()
	for k, v in multi_eval_aggr_result.items(): mean_result[k], std_result[k] = round(np.mean(v), 2), round(np.std(v), 2)

	# print(f"[blue]{title}[/blue]")
	results_table = prettytable.PrettyTable()
	results_table.float_format = ".2f"
	results_table.align = 'c'
	results_table.border = True
	results_table.field_names = ['Img R@1', 'Img R@5', 'Img R@10', 'Img R_Mean', 
								 'Txt R@1', 'Txt R@5', 'Txt R@10', 'Txt R_Mean', 
								 'R_Mean']
	results_table.title = title

	if prettytable.__version__ >= '3.16.0':
		results_table.add_divider()
	else:
		results_table.add_row(['-'*10]*9)         

	# mean
	results_table.add_row(
		[f"{mean_result['img_r1']:.2f}", f"{mean_result['img_r5']:.2f}", f"{mean_result['img_r10']:.2f}", f"{mean_result['img_r_mean']:.2f}", 
		 f"{mean_result['txt_r1']:.2f}", f"{mean_result['txt_r5']:.2f}", f"{mean_result['txt_r10']:.2f}", f"{mean_result['txt_r_mean']:.2f}", 
		 f"{mean_result['r_mean']:.2f}"]
		)
	# std
	results_table.add_row(
		# \u00B1
		[f"± {std_result['img_r1']:.2f}", f"± {std_result['img_r5']:.2f}", f"± {std_result['img_r10']:.2f}", f"± {std_result['img_r_mean']:.2f}", 
		f"± {std_result['txt_r1']:.2f}", f"± {std_result['txt_r5']:.2f}", f"± {std_result['txt_r10']:.2f}", f"± {std_result['txt_r_mean']:.2f}", 
		f"± {std_result['r_mean']:.2f}"]
		)
	
	return results_table

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

def merge(w1, w2, w0, ratio, device='cpu'):
	w12 = {} # w12 = (w1 + w2) / 2
	for key in w1.keys():                
		w12[key] = (w1[key].clone().to(device) + w2[key].clone().to(device)) / 2.

	w_merge = copy.deepcopy(w12)
	for key, r in ratio.items():        
		w_merge[key] = w12[key].clone().to(device) * r + w0[key].clone().to(device) * (1. - r)
	return w_merge

def _state_to_device(d, device='cuda', non_blocking=True, dtype=torch.float32):
	"""Move a (state_dict-like) mapping to device once, skipping non-floats/buffers."""
	out = {}
	for k, v in d.items():
		if torch.is_tensor(v) and v.is_floating_point():
			# cast once to a consistent dtype for stable dot products
			out[k] = v.to(device=device, dtype=dtype, non_blocking=non_blocking)
		else:
			# keep as-is; shapes will fail-fast later if mismatched
			out[k] = v
	return out

@torch.inference_mode()
def compute_cosine_dict(state1, state2, ref, device='cuda'):
	"""
	Return per-key cosine similarities between (state1-ref) and (state2-ref),
	computed entirely on GPU without acos/deg.
	"""
	cos_dict = collections.OrderedDict()
	# s1 = _state_to_device(state1, device)
	# s2 = _state_to_device(state2, device)
	# r  = _state_to_device(ref,    device)
	s1 = state1
	s2 = state2
	r = ref

	for k in r.keys():
		if k not in s1 or k not in s2:
			continue
		v1, v2, vr = s1[k], s2[k], r[k]

		# must be float tensors with identical shape
		if (not torch.is_tensor(v1)) or (not torch.is_tensor(v2)) or (not torch.is_tensor(vr)):
			continue
		if (not v1.is_floating_point()) or (not v2.is_floating_point()) or (not vr.is_floating_point()):
			continue
		if v1.shape != v2.shape or v1.shape != vr.shape:
			continue

		a = (v1 - vr).clone().detach()
		b = (v2 - vr).clone().detach()

		# fused dot/norms – avoid intermediate allocations
		dot = torch.dot(a.reshape(-1), b.reshape(-1))
		na  = torch.linalg.vector_norm(a)
		nb  = torch.linalg.vector_norm(b)
		cos = (dot / (na * nb + EPS)).clamp(-1.0, 1.0)
		cos_dict[k] = cos.detach().to('cpu')  # small scalars back to CPU for ratio calc
	return cos_dict

def compute_ratio_from_cos(cos_dict, k=2.0):
	"""
	Same formula as your compute_ratio(angle_dict, k), but uses cos(theta) directly.
	ratio = k cosθ / ((k-1) cosθ + 1)
	"""
	out = {}
	for k_ in cos_dict.keys():
		c = float(cos_dict[k_])
		out[k_] = (k * c) / (((k - 1) * c) + 1.0 + EPS)
	return out

@torch.inference_mode()
def fast_merge(w1, w2, w0, ratio, device='cuda', dtype=torch.float32, non_blocking=True):
	"""
	빠른 버전 (당신의 merge와 동일한 수식):
	  1) w12 = (w1 + w2) / 2
	  2) w_merge[key] = r * w12 + (1 - r) * w0  (ratio에 key가 있으면), 없으면 w12 유지

	최적화:
	  - float 파라미터만 이동/연산
	  - 각 key당 .to() 1회
	  - r*w12 + (1-r)*w0 = w0 + r*(w12 - w0) 로 한 번에 연산
	"""
	w_merge = {}

	for k in w0.keys():
		# 세 dict 모두에 있어야 하고, float tensor이면서 shape 동일해야 안전
		if (k in w1) and (k in w2) \
		   and torch.is_tensor(w1[k]) and torch.is_tensor(w2[k]) and torch.is_tensor(w0[k]) \
		   and w1[k].is_floating_point() and w2[k].is_floating_point() and w0[k].is_floating_point() \
		   and (w1[k].shape == w2[k].shape == w0[k].shape):

			t1 = w1[k].to(device=device, dtype=dtype, non_blocking=non_blocking)
			t2 = w2[k].to(device=device, dtype=dtype, non_blocking=non_blocking)
			t0 = w0[k].to(device=device, dtype=dtype, non_blocking=non_blocking)

			w12 = 0.5 * (t1 + t2)  # 평균

			# 기본은 당신 코드처럼 w12로 초기화 (ratio에 key가 없으면 평균 유지)
			outk = w12

			# ratio에 key가 있으면 w0와 보간
			if k in ratio:
				r = torch.as_tensor(ratio[k], dtype=dtype, device=device)
				# r*w12 + (1-r)*w0  ==  w0 + r*(w12 - w0)  (연산 1번으로 축약)
				outk = t0 + r * (w12 - t0)

			w_merge[k] = outk

		else:
			# float 가 아니거나 shape가 다르면 원본 w0를 그대로 둡니다 (안전한 기본값)
			w_merge[k] = w0[k]

	return w_merge

def load_model_state_dict(state_dict, map_location='cpu'):
	state_dict = torch.load(state_dict, map_location=map_location)
	return state_dict

def make_distillation_model(args, img_expert_files, txt_expert_files, student_net, base_dir='./buffer/flickr8k/nfnet_bert/InfoNCE', file_format='replay_buffer', merge_image=True, merge_text=True, verbose=False):
	BASE_DIR = base_dir
	FILE_FORMAT = file_format	
	total_img_buffers = len(img_expert_files)-1

	FIX_EXPERT_VARY_EPOCH, VARY_EXPERT_VARY_EPOCH = False, False
	# FIX_EXPERT_VARY_EPOCH, VARY_EPOCH_VARY_EPOCH = True, False
 
	if FIX_EXPERT_VARY_EPOCH:
		EXPERT_NUM1 = random.randint(0, total_img_buffers)
		MAX_EPOCH = args.max_start_epoch
	
  
		EPOCH_NUM1 = random.choice(range(1, MAX_EPOCH+1)) # args.max_start_epoch)
		img_file = os.path.join(BASE_DIR, F'img_{FILE_FORMAT}_{EXPERT_NUM1}.pt')
		txt_file = os.path.join(BASE_DIR, F'txt_{FILE_FORMAT}_{EXPERT_NUM1}.pt')
		img_expert_1 = load_model_state_dict(img_file, map_location='cuda')[0][EPOCH_NUM1]
		txt_expert_1 = load_model_state_dict(txt_file, map_location='cuda')[0][EPOCH_NUM1]
		
		epoch_pool = [i for i in range(1, MAX_EPOCH+1) if i != EPOCH_NUM1]
		EPOCH_NUM2 = random.choice(epoch_pool)
		img_file = os.path.join(BASE_DIR, F'img_{FILE_FORMAT}_{EXPERT_NUM1}.pt')
		txt_file = os.path.join(BASE_DIR, F'txt_{FILE_FORMAT}_{EXPERT_NUM1}.pt')
		img_expert_2 = load_model_state_dict(img_file, map_location='cuda')[0][EPOCH_NUM2]
		txt_expert_2 = load_model_state_dict(txt_file, map_location='cuda')[0][EPOCH_NUM2]
  	
	elif VARY_EXPERT_VARY_EPOCH:
		EXPERT_NUM1 = random.randint(0, total_img_buffers)
		MAX_EPOCH = args.max_start_epoch

		EPOCH_NUM1 = random.choice(range(1, MAX_EPOCH+1)) 
		img_file = os.path.join(BASE_DIR, F'img_{FILE_FORMAT}_{EXPERT_NUM1}.pt')
		txt_file = os.path.join(BASE_DIR, F'txt_{FILE_FORMAT}_{EXPERT_NUM1}.pt')
		img_expert_1 = load_model_state_dict(img_file, map_location='cuda')[0][EPOCH_NUM1]
		txt_expert_1 = load_model_state_dict(txt_file, map_location='cuda')[0][EPOCH_NUM1]
		
		EXPERT_NUM2 = random.randint(0, total_img_buffers)
		epoch_pool = [i for i in range(1, MAX_EPOCH+1) if i != EPOCH_NUM1]
		EPOCH_NUM2 = random.choice(epoch_pool)
		img_file = os.path.join(BASE_DIR, F'img_{FILE_FORMAT}_{EXPERT_NUM2}.pt')
		txt_file = os.path.join(BASE_DIR, F'txt_{FILE_FORMAT}_{EXPERT_NUM2}.pt')
		img_expert_2 = load_model_state_dict(img_file, map_location='cuda')[0][EPOCH_NUM2]
		txt_expert_2 = load_model_state_dict(txt_file, map_location='cuda')[0][EPOCH_NUM2]
		
	else:
		MIN_EPOCH = args.min_start_epoch
		MAX_EPOCH = args.max_start_epoch
  
		EXPERT_NUM1 = random.randint(0, 19)
		EPOCH_NUM1 = random.randint(MIN_EPOCH, MAX_EPOCH+1)
		img_file = os.path.join(BASE_DIR, f'img_replay_buffer_{EXPERT_NUM1}_{EPOCH_NUM1}.pth')
		txt_file = os.path.join(BASE_DIR, f'txt_replay_buffer_{EXPERT_NUM1}_{EPOCH_NUM1}.pth')
		img_expert_1 = load_model_state_dict(img_file, map_location='cuda')
		txt_expert_1 = load_model_state_dict(txt_file, map_location='cuda')
	
		EXPERT_NUM2 = random.randint(0, 19)
		if EPOCH_NUM1 == 0:
			EPOCH_NUM2 = random.randint(1, MAX_EPOCH+1)
		else:
			EPOCH_NUM2 = random.randint(MIN_EPOCH, MAX_EPOCH+1)
		img_file = os.path.join(BASE_DIR, f'img_replay_buffer_{EXPERT_NUM2}_{EPOCH_NUM2}.pth')
		txt_file = os.path.join(BASE_DIR, f'txt_replay_buffer_{EXPERT_NUM2}_{EPOCH_NUM2}.pth')
		img_expert_2 = load_model_state_dict(img_file, map_location='cuda')
		txt_expert_2 = load_model_state_dict(txt_file, map_location='cuda')
	
	method = 'FIX_EXPERT_VARY_EPOCH' if FIX_EXPERT_VARY_EPOCH else 'VARY_EXPERT_VARY_EPOCH' if VARY_EXPERT_VARY_EPOCH else 'RANDOM_EXPERT_RANDOM_EPOCH'
	# print(f'{method} || EXPERT_NUM1: {EXPERT_NUM1}, EPOCH_NUM1: {EPOCH_NUM1} || EXPERT_NUM2: {EXPERT_NUM2}, EPOCH_NUM2: {EPOCH_NUM2}')
 
	# check if the expert state dicts are valid
	assert isinstance(img_expert_1, dict) and isinstance(txt_expert_1, dict) and isinstance(img_expert_2, dict) and isinstance(txt_expert_2, dict)
	
	# initial model
	student_net.image_encoder.to('cuda')
	student_net.text_projection.to('cuda')
	img_initial = student_net.image_encoder.state_dict()
	txt_initial = student_net.text_projection.state_dict()
	
	if merge_image:
		# img_angle = compute_angle(img_expert_1, img_expert_2, img_initial)
		# img_ratio = compute_ratio(img_angle)
		# merged_img_model = merge(img_expert_1, img_expert_2, img_initial, img_ratio, device='cuda')
  
		cos_img = compute_cosine_dict(img_expert_1, img_expert_2, img_initial, device="cuda")
		ratio_img = compute_ratio_from_cos(cos_img, k=2.0)
		merged_img_model = fast_merge(img_expert_1, img_expert_2, img_initial, ratio_img, device="cuda")
	else:
		merged_img_model = None

	if merge_text:
		# txt_angle = compute_angle(txt_expert_1, txt_expert_2, txt_initial)
		# txt_ratio = compute_ratio(txt_angle)
		# merged_txt_model = merge(txt_expert_1, txt_expert_2, txt_initial, txt_ratio, device='cuda')
  
		cos_txt = compute_cosine_dict(txt_expert_1, txt_expert_2, txt_initial, device="cuda")
		ratio_txt = compute_ratio_from_cos(cos_txt, k=2.0)
		merged_txt_model = fast_merge(txt_expert_1, txt_expert_2, txt_initial, ratio_txt, device="cuda")
	else:
		merged_txt_model = None

	# orig_img_encoder_weights = copy.deepcopy(student_net.image_encoder.state_dict())
	# orig_txt_encoder_weights = copy.deepcopy(student_net.text_projection.state_dict())

	# weights_changed = False
	# for key in orig_img_encoder_weights:
	# 	if not torch.equal(orig_img_encoder_weights[key].to('cpu'), merged_img_model[key].data.to('cpu')):
	# 		weights_changed = True
	# 		break
	# assert weights_changed, "student_net.image_encoder weights have NOT changed after loading merged_img_model."

	# weights_changed = False
	# for key in orig_txt_encoder_weights:
	# 	if not torch.equal(orig_txt_encoder_weights[key].to('cpu'), merged_txt_model[key].data.to('cpu')):
	# 		weights_changed = True
	# 		break
	# assert weights_changed, "student_net.text_projection weights have NOT changed after loading merged_txt_model."
	# image_encoder_result = student_net.image_encoder.load_state_dict(merged_img_model, strict=False)
	# text_proj_result = student_net.text_projection.load_state_dict(merged_txt_model, strict=False)
	
	if verbose:
		return merged_img_model, merged_txt_model, (EXPERT_NUM1, EPOCH_NUM1), (EXPERT_NUM2, EPOCH_NUM2)
	else:
		return merged_img_model, merged_txt_model


def make_dir(p: str):
	if not os.path.exists(p):
		os.makedirs(p)

def to_jsonable(v):
	try:
		json.dumps(v)
		return v
	except TypeError:
		return {k: to_jsonable(x) for k, x in vars(v).items()} if hasattr(v, "__dict__") else str(v)

def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
	return x / (x.norm(dim=dim, keepdim=True) + eps)


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
	a = l2_normalize(a, dim=-1)
	b = l2_normalize(b, dim=-1)
	return a @ b.t()

# =============================
# Image utils
# =============================

def denormalize_clip(x):
	# x: float Tensor in CLIP-normalized space, shape [B,3,H,W]
	mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=x.device).view(1,3,1,1)
	std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=x.device).view(1,3,1,1)
	return x * std + mean

def to_grid_for_tb(x, nrow=8):
	# x: [B,3,H,W] in CLIP-normalized space (synthetic images)
	x_denorm = denormalize_clip(x)                # 역정규화
	x_denorm = torch.clamp(x_denorm, 0.0, 1.0)    # TB expects [0,1]
	return make_grid(x_denorm.cpu(), nrow=nrow)

def _norm_t(it, T):
	if T is None or T <= 0:
		return 1.0
	t = it / float(T)
	return max(0.0, min(1.0, t))
# =====================================================
# Teachers (buffer mix)
# =====================================================

# def load_model_state_dict(state_dict):
# 	state_dict = torch.load(state_dict, map_location='cpu')
# 	return state_dict

def _load_clip_from_buffers(args):
	k = random.randint(0, args.num_buffers - 1)
	img_path = os.path.join(args.buffer_path, f'img_replay_buffer_{k}_10.pth')
	txt_path = os.path.join(args.buffer_path, f'txt_replay_buffer_{k}_10.pth')
	img_sd = torch.load(img_path, map_location='cuda')
	txt_sd = torch.load(txt_path, map_location='cuda')
 
	
	return img_sd, txt_sd, k


def _param_interpolate_(tgt_model, sd_a, sd_b, alpha):
	with torch.no_grad():
		msd = tgt_model.state_dict()
		for k in msd.keys():
			if k in sd_a and k in sd_b and torch.is_tensor(sd_a[k]) and torch.is_tensor(sd_b[k]):
				if msd[k].shape == sd_a[k].shape == sd_b[k].shape and msd[k].dtype == sd_a[k].dtype == sd_b[k].dtype:
					msd[k].copy_(alpha * sd_a[k] + (1.0 - alpha) * sd_b[k])
		tgt_model.load_state_dict(msd)


def _make_mixed_teacher(args, device, alpha_from=None, alpha_to=None):
	if alpha_from is None:
		alpha_from = 0.0
	if alpha_to is None:
		alpha_to = 1.0
		
	if alpha_from == alpha_to:
		alpha = alpha_from
	else:
		alpha = random.uniform(alpha_from, alpha_to)
	
 
	rand_model = CLIPModel_full(args, temperature=args.temperature).to(device)
	rand_img_sd = copy.deepcopy(rand_model.image_encoder.state_dict())
	rand_txt_sd = copy.deepcopy(rand_model.text_projection.state_dict())
	
	# load from buffer files
	# import glob, os
	# BASE_DIR = args.buffer_path
	# FILE_FORMAT = 'replay_buffer'		
	# img_expert_files = glob.glob(os.path.join(BASE_DIR, f'img_{FILE_FORMAT}_*.pt')) # list of filenames
	# txt_expert_files = glob.glob(os.path.join(BASE_DIR, f'txt_{FILE_FORMAT}_*.pt')) # list of filenames
	# total_img_buffers = len(img_expert_files)-1
 
	# EXPERT_NUM1 = random.randint(0, total_img_buffers)
	# EPOCH_NUM1 = random.choice(range(1, args.max_start_epoch+1))
	# pretrain_img_sd = load_model_state_dict(os.path.join(BASE_DIR, F'img_{FILE_FORMAT}_{EXPERT_NUM1}.pt'))[0][EPOCH_NUM1]
	# pretrain_txt_sd = load_model_state_dict(os.path.join(BASE_DIR, F'txt_{FILE_FORMAT}_{EXPERT_NUM1}.pt'))[0][EPOCH_NUM1]
 	
	pretrain_img_sd, pretrain_txt_sd, chosen_k = _load_clip_from_buffers(args)
  
	mixed = CLIPModel_full(args, temperature=args.temperature).to(device)
	_param_interpolate_(mixed.image_encoder, pretrain_img_sd, rand_img_sd, alpha)
	_param_interpolate_(mixed.text_projection, pretrain_txt_sd, rand_txt_sd, alpha)

	mixed.eval()
	for p in mixed.parameters():
		p.requires_grad_(False)
	# return mixed, alpha, EXPERT_NUM1
	return mixed, alpha, chosen_k


def _make_teachers(args, device, n_teachers: int, alpha_from=None, alpha_to=None) -> List[CLIPModel_full]:
	teachers = []
	alphas = []
	for _ in range(n_teachers):
		t, alpha, _ = _make_mixed_teacher(args, device, alpha_from=alpha_from, alpha_to=alpha_to)
		teachers.append(t)
		alphas.append(alpha)
	return teachers, alphas


# =====================================================
# Losses & Assignments
# =====================================================

def clip_symmetric_nce_loss(img_feats: torch.Tensor, txt_feats: torch.Tensor, temperature: float = 0.07):
	logits = (img_feats @ txt_feats.t()) / temperature
	targets = torch.arange(img_feats.size(0), device=img_feats.device)
	loss_i2t = F.cross_entropy(logits, targets)
	loss_t2i = F.cross_entropy(logits.t(), targets)
	return 0.5 * (loss_i2t + loss_t2i)


# =====================================================
# Feature extraction
# =====================================================

def get_clip_feats(model, images, txt_embeds, args):
	model = model.to(images.device)
	image_features = model.image_encoder(images)
	if hasattr(model, 'image_projection'):
		im_embed = model.image_projection(image_features.float())
	else:
		im_embed = image_features.float()
	if args.distill:
		text_features = txt_embeds
	else:
		text_features = model.text_encoder(txt_embeds)
	txt_embed = model.text_projection(text_features.float())
	im_embed = l2_normalize(im_embed, dim=1)
	txt_embed = l2_normalize(txt_embed, dim=1)
	return im_embed, txt_embed

# =====================================================
# Geodesic MMD
# ====================================================

# === add near Losses & Assignments ===
def geodesic_distance(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
	# a,b: [N,D] L2-normalized 가정(안전하게 내부 normalize)
	a = F.normalize(a, dim=1)
	b = F.normalize(b, dim=1)
	cosv = (a @ b.t()) if a.dim()==2 and b.dim()==2 else (a*b).sum(dim=-1, keepdim=False)
	cosv = cosv.clamp(-1.0 + eps, 1.0 - eps)
	return torch.acos(cosv)  # [N,N] or [N] or scalar-like

def geodesic_loss_pair(a, b, squared: bool = True) -> torch.Tensor:
	theta = geodesic_distance(a, b)  # broadcast 가능
	return (theta * theta).mean() if squared else theta.mean()

def spherical_rbf_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float = 0.5) -> torch.Tensor:
	# k(x,y) = exp( - geodesic(x,y)^2 / (2 sigma^2) )
	theta = geodesic_distance(x, y)   # [Nx, Ny]
	return torch.exp( - (theta * theta) / (2.0 * sigma * sigma) )

# =====================================================
# Conditional Spherical MMD
# =====================================================
def spherical_rbf_weights(query: torch.Tensor, keys: torch.Tensor, sigma: float = 0.5, eps: float = 1e-8):
	"""
	query: [Nq, D] (L2-normalized 권장)
	keys : [Nk, D]
	반환:  [Nq, Nk] softmax-like normalized weights (row-normalized)
	"""
	K = spherical_rbf_kernel(query, keys, sigma=sigma)  # [Nq, Nk]
	W = K / (K.sum(dim=1, keepdim=True) + eps)
	return W

def conditional_spherical_mean(values: torch.Tensor, keys: torch.Tensor, query: torch.Tensor, sigma: float = 0.5, eps: float = 1e-8):
	"""
	values: [Nk, D]  (ex. real X, real Y, real G)
	keys  : [Nk, D]  (조건변수: ex. real Y for E[X|Y])
	query : [Nq, D]  (ex. syn y to evaluate E_real[X|Y=y_syn])
	반환:   [Nq, D]   (row-wise L2-normalized)
	"""
	W = spherical_rbf_weights(query, keys, sigma=sigma, eps=eps)         # [Nq, Nk]
	W = torch.ones_like(W)
	W = W / W.sum(dim=1, keepdim=True)
	m = W @ values                                                        # [Nq, D]
	m = F.normalize(m, dim=1, eps=eps)
	return m

# === Conditional kernel-MMD: L_{cond-MMD}(g | key)  ==========================
# g_real:  [Br, D] (unit-norm 권장)
# g_syn:   [B,  D] (unit-norm 권장)
# key_real: [Br, D]  (e.g., feat_txt_real or feat_img_real)
# key_syn:  [B,  D]  (e.g., yi_s or xi_s)
# sigma_g: kernel bandwidth on g-space (geodesic)
# sigma_key: kernel bandwidth on key-space (geodesic)
# weight_mode: 'normalize' (kernel 정규화) 또는 'softmax' (온도=1/sigma_key^2 근사)
# stopgrad_W: True면 W를 detach하여 키-가중치 경로의 gradient를 끊음(안정성↑)
def conditional_kernel_mmd(
	g_real: torch.Tensor,
	g_syn: torch.Tensor,
	key_real: torch.Tensor,
	key_syn: torch.Tensor,
	*,
	sigma_g: float = 0.5,
	sigma_key: float = 0.5,
	weight_mode: str = 'normalize',
	stopgrad_W: bool = False,
	eps: float = 1e-12,
) -> torch.Tensor:
	device = g_syn.device
	Br = g_real.size(0)
	B  = g_syn.size(0)
	# 1) key-space kernel between real keys and syn keys: [Br, B]
	#    (real -> columns sum to 1 for each syn sample)
	Ky = spherical_rbf_kernel(key_real, key_syn, sigma=sigma_key)  # [Br, B]

	if weight_mode == 'softmax':
		# softmax over real(j) per syn(i): numerically stable
		W = torch.softmax(Ky, dim=0)  # [Br, B]
	else:
		# kernel normalize per syn(i)
		W = Ky / (Ky.sum(dim=0, keepdim=True) + eps)  # [Br, B]

	# 3) 엔트로피/커버리지 보정
	L_reg = entropy_reg_kl_to_uniform(W)
	
	# 4) (옵션) syn key repulsion
	keys_syn = F.normalize(key_syn, dim=1)
	L_rep = repulsion_loss_spherical(keys_syn, sigma_key=sigma_key)
	
	if stopgrad_W:
		W = W.detach() # real-syn key gap

	# 2) g-space kernels
	Kg_sr = spherical_rbf_kernel(g_syn,  g_real, sigma=sigma_g)  # [B, Br]
	Kg_rr = spherical_rbf_kernel(g_real, g_real, sigma=sigma_g)  # [Br, Br]

	# 3) Per-sample conditional MMD^2 for each syn i:
	#    term1 = k(g_i, g_i) = 1
	#    term2 = 2 * <k(g_i,·), mu_{p(g|key_i)}> = 2 * sum_j w_{ji} k(g_i, g_j)
	#    term3 = ||mu||^2 = w_i^T K_rr w_i
	term1 = torch.ones(B, device=device)

	# (Kg_sr * W^T).sum(dim=1) == sum_j w_{ji} k(g_i,g_j)
	term2 = 2.0 * (Kg_sr * W.t()).sum(dim=1)  # [B]

	# W: [Br, B]  ->  W^T K_rr W  : [B, B]
	# diag gives each sample's w_i^T K_rr w_i
	# (mm 순서: (W^T K_rr) 먼저 해서 메모리 절약)
	M = (W.t() @ Kg_rr) @ W         # [B, B]
	term3 = torch.diagonal(M, dim1=-2, dim2=-1)  # [B]

	L_cgap_mmd = (term1 - term2 + term3).mean()
	
	return L_cgap_mmd, L_reg, L_rep

def entropy_and_coverage_regularizer(W: torch.Tensor, *, eps: float = 1e-12):
	"""
	W: [Br, B] (real-by-syn), columns sum to 1
	반환:
	  L_ent:  - mean entropy (값이 작아지면 엔트로피 커짐)  -> loss에 +λ_ent * L_ent
	  L_cov:  row-sum 균형 손실 (coverage 균형)              -> loss에 +λ_cov * L_cov
	"""
	# --- (A) Column entropy: encourage high entropy per column ---
	# H_i = - sum_j W_{j,i} log W_{j,i}
	Wc = (W + eps)
	H_col = -(Wc * Wc.log()).sum(dim=0)               # [B]
	L_ent = - H_col.mean()                             # minimize(-mean H) => maximize mean H

	# --- (B) Row coverage: encourage balanced usage of reals ---
	row_sum = W.sum(dim=1)                             # [Br]
	target = W.size(1) / W.size(0)                     # B/Br
	L_cov = torch.mean((row_sum - target) ** 2)
	return L_ent, L_cov

def entropy_reg_kl_to_uniform(W: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
	# W: [Br, B], columns sum to 1
	Br = W.size(0)
	Wc = (W + eps)
	# mean over columns of sum_j w log(w*Br)
	kl = (Wc * (Wc * Br).log()).sum(dim=0).mean()
	return kl  # >= 0, 0 at perfectly uniform

def repulsion_loss_spherical(keys_syn: torch.Tensor, sigma_key: float = 0.5):
	# keys_syn: [B,D], unit-norm
	Kss = spherical_rbf_kernel(keys_syn, keys_syn, sigma=sigma_key)  # [B,B]
	offdiag = (Kss.sum() - Kss.diagonal().sum()) / max(1, (Kss.numel() - Kss.size(0)))
	return offdiag  # minimize => push apart

# ==== Product-kernel MMD^2 for joint variables (A,B) ====
def mmd2_product_kernel(
	A_syn: torch.Tensor, B_syn: torch.Tensor,
	A_real: torch.Tensor, B_real: torch.Tensor,
	sigma_A: float, sigma_B: float,
) -> torch.Tensor:
	"""
	A_*, B_*: [N, D] vectors (구면 비교 권장 → 내부에서 normalize는 spherical_rbf_kernel이 처리)
	커널: k((a,b),(a',b')) = kA(a,a') * kB(b,b')
	반환: unbiased MMD^2 (scalar)
	"""
	# 개별 커널들 (지오데식 RBF)
	KA_ss = spherical_rbf_kernel(A_syn,  A_syn,  sigma=sigma_A)  # [Ns,Ns]
	KA_rr = spherical_rbf_kernel(A_real, A_real, sigma=sigma_A)  # [Nr,Nr]
	KA_sr = spherical_rbf_kernel(A_syn,  A_real, sigma=sigma_A)  # [Ns,Nr]

	KB_ss = spherical_rbf_kernel(B_syn,  B_syn,  sigma=sigma_B)  # [Ns,Ns]
	KB_rr = spherical_rbf_kernel(B_real, B_real, sigma=sigma_B)  # [Nr,Nr]
	KB_sr = spherical_rbf_kernel(B_syn,  B_real, sigma=sigma_B)  # [Ns,Nr]

	# 제품 커널(아다마르 곱)
	K_ss = KA_ss * KB_ss
	K_rr = KA_rr * KB_rr
	K_sr = KA_sr * KB_sr

	# unbiased estimator (네 코드의 offdiag_mean 규칙과 동일)
	def offdiag_mean(K):
		n = K.size(0)
		if K.size(0) == K.size(1) and n > 1:
			return (K.sum() - K.diag().sum()) / (n * (n - 1))
		else:
			return K.mean()

	mmd2 = offdiag_mean(K_ss) + offdiag_mean(K_rr) - 2.0 * K_sr.mean()
	return mmd2

# =====================================================
# Diagnostics helpers (added)
# =====================================================

def random_plane_project(v: torch.Tensor, rng: torch.Generator = None) -> torch.Tensor:
	"""
	v: [N,D], returns [N,2] projection onto a random 2D orthonormal basis
	"""
	with torch.no_grad():
		v = F.normalize(v, dim=1)
		N, D = v.shape
		if rng is None:
			rng = torch.Generator(device=v.device)
			rng.manual_seed(torch.randint(0, 1_000_000, (1,), device=v.device).item())
		a = torch.randn(D, generator=rng, device=v.device)
		a = F.normalize(a, dim=0)
		b = torch.randn(D, generator=rng, device=v.device)
		b = b - (b @ a) * a
		b = F.normalize(b, dim=0)
		B = torch.stack([a, b], dim=1)  # [D,2]
		return (v @ B).detach()  # [N,2]

def fig_scatter_2d(real_xy: np.ndarray, syn_xy: np.ndarray, title: str):
	"""
	real_xy, syn_xy: [N,2] numpy
	"""
	fig = plt.figure(figsize=(4.2, 4.2), dpi=120)
	ax = fig.add_subplot(111)
	if real_xy.size > 0:
		ax.scatter(real_xy[:,0], real_xy[:,1], s=6, alpha=0.6, label='real')
	if syn_xy.size > 0:
		ax.scatter(syn_xy[:,0], syn_xy[:,1], s=6, alpha=0.6, label='syn')
	ax.set_title(title)
	ax.legend(loc='best', fontsize=8)
	ax.set_xticks([]); ax.set_yticks([])
	ax.grid(True, alpha=0.2)
	plt.tight_layout()
	return fig

def fig_hist_two(dist_real: np.ndarray, dist_syn: np.ndarray, title: str, bins: int = 40):
	fig = plt.figure(figsize=(4.8, 3.4), dpi=120)
	ax = fig.add_subplot(111)
	ax.hist(dist_real, bins=bins, alpha=0.6, density=True, label='real')
	ax.hist(dist_syn,  bins=bins, alpha=0.6, density=True, label='syn')
	ax.set_title(title)
	ax.legend(loc='best', fontsize=8)
	plt.tight_layout()
	return fig

def pair_angle(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
	c = (F.normalize(a,dim=1) * F.normalize(b,dim=1)).sum(dim=1).clamp(-1+1e-6, 1-1e-6)
	return torch.acos(c)  # [B] in radians

def set_seed(seed: int = 0):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	

def make_dir(p: str):
	if not os.path.exists(p):
		os.makedirs(p)

def to_jsonable(v):
	try:
		json.dumps(v)
		return v
	except TypeError:
		return {k: to_jsonable(x) for k, x in vars(v).items()} if hasattr(v, "__dict__") else str(v)

def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
	return x / (x.norm(dim=dim, keepdim=True) + eps)


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
	a = l2_normalize(a, dim=-1)
	b = l2_normalize(b, dim=-1)
	return a @ b.t()
# =====================================================
# Main
# =====================================================

def main(args):
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	args.device = device
	set_seed(args.seed)
	
	# Save dir
	make_dir(args.log_dir)
	current_time = datetime.datetime.now().strftime("%m-%d_%H-%M-%S")
	run_dir = os.path.join(args.log_dir, f"{current_time}_{args.name}")
	make_dir(run_dir)
	
	cfg = {k: to_jsonable(v) for k, v in vars(args).items()}
	with open(os.path.join(run_dir, "args.json"), "w", encoding="utf-8") as f:
		json.dump(cfg, f, ensure_ascii=False, indent=2)

	src_file = __file__
	dst_file = os.path.join(run_dir, os.path.basename(src_file))  # run_dir 안에 저장될 경로
	shutil.copy(src_file, dst_file)
	
	writer = SummaryWriter(log_dir=run_dir)
	# 하이퍼/환경 정보 간단 기록
	writer.add_text("config/args", f"{args}", global_step=0)

	
	# Data
	train_loader, test_loader, train_dataset, test_dataset = get_dataloaders(args)
	train_sentences = train_dataset.get_all_captions()
	data = load_or_process_file('test_text', textprocess_test, args, test_loader)
	train_caption = load_or_process_file('train_text', textprocess_train, args, train_sentences)
	
	bert_test_embed = torch.from_numpy(data['bert_test_embed']).cpu()               # [N_txt_test, 768]
	print("The shape of bert_test_embed: {}".format(bert_test_embed.shape))
	train_caption_embed = torch.from_numpy(train_caption['bert_train_embed']).cpu()  # [N_txt_train, 768]
	print("The shape of train_caption_embed: {}".format(train_caption_embed.shape))

	if args.eval_it>0:
		eval_it_pool = np.arange(0, args.Iteration + 1, args.eval_it).tolist()
	else:
		eval_it_pool = []
	
	# Synthetic variables
	# image_syn, text_syn = get_images_texts(args.num_queries, train_dataset, args, init='random')

	# if os.path.isfile(os.path.join(args.eval_target_path, "synthetic_data", f"distilled_pairs_best.pt")):
	# 	eval_data = torch.load(os.path.join(args.eval_target_path, "synthetic_data", f"distilled_pairs_best.pt"))
	# else:
	# 	target_it = args.target_it
	# 	eval_data = torch.load(os.path.join(args.eval_target_path, "synthetic_data", f"distilled_pairs_{target_it}.pt"))

	# eval_data = torch.load('logs/distill_final/flickr/nfnet/bert/InfoNCE/500/mixed/11_09_2025_133938/30k_500_nce_ummd_gmmd_mixed_rerun_256/distilled_pairs_best.pt')
	# eval_data = torch.load('logs/distill_final/coco/nfnet/bert/InfoNCE/500/mixed/11_09_2025_011424/coco_500_nce_ummd_gmmd_mixed/distilled_pairs_best.pt')
	eval_data = torch.load('final_pt/11-06_01-04-58_flickr8k_lors_99_bufferseed0123_ev100-1/distilled_pairs_max.pt')
 
	image_syn = eval_data['image']  # [N_synth, 3, H, W]
	text_syn  = eval_data['text']   # [N_synth, seq_len
	eval_target_it = eval_data['iter']
	if 'similarity_mat' in eval_data:
		similarity_mat = eval_data['similarity_mat']
	else:
		similarity_mat = None
	
	# eval_models = []
	# for i in range(args.num_eval):
	# 	eval_net = CLIPModel_full(args, temperature=args.temperature)
	# 	eval_net.image_encoder.model.to('cpu')
	# 	eval_net.text_encoder.model.to('cpu')
	# 	eval_net.text_projection.to('cpu')
	# 	eval_models.append(eval_net)
	
		
	image_syn = image_syn.to(device).detach().requires_grad_(True)
	text_syn  = text_syn.to(device).detach().requires_grad_(True)
		
		
	save_this_it = False
	''' Evaluate synthetic data '''

	print('Evaluation\nimage_model_train = %s, text_model_train = %s, iteration = %d'%(args.image_encoder, args.text_encoder, eval_target_it))
	
	print('DSA augmentation strategy: \n', args.dsa_strategy)
	print('DSA augmentation parameters: \n', args.dsa_param.__dict__)
	
	multi_eval_aggr_result = defaultdict(list)  # aggregated results of multiple evaluations
	
	# LoRS 평가: 합성쌍으로 CLIP 학습 후 epoch_test + itm_eval
	for it_eval in range(args.num_eval):
		args.image_pretrained = True
		# torch.random.manual_seed(int(time.time() * 1000) % 100000)
		eval_net = CLIPModel_full(args, temperature=args.temperature)
		# eval_net = copy.deepcopy(eval_models[it_eval])
		eval_net.image_encoder.model.to('cuda')
		eval_net.text_encoder.model.to('cuda')
		eval_net.text_projection.to('cuda')
		
		
		with torch.no_grad():
			image_save = image_syn
			text_save = text_syn
		image_syn_eval, text_syn_eval = copy.deepcopy(image_save.detach()), copy.deepcopy(text_save.detach())
		
		eval_net, acc_train_list, best_val_result = evaluate_synset(
			it_eval=it_eval,
			net=eval_net,
			images_train=image_syn_eval,
			labels_train=text_syn_eval,
			testloader=test_loader,
			args=args,
			bert_test_embed=bert_test_embed,
			similarity_train=similarity_mat,
		)
		
		for k, v in best_val_result.items():
			multi_eval_aggr_result[k].append(v)

	results_table = print_results(multi_eval_aggr_result, title=f'{args.name} results for {args.dataset}')
	logger.info(results_table)
	print(results_table)
	# TensorBoard: 평가 평균/표준편차 기록
	for key, values in multi_eval_aggr_result.items():
		writer.add_scalar(f"Eval/Mean/{key}", float(np.mean(values)), eval_target_it)
		writer.add_scalar(f"Eval/Std/{key}",  float(np.std(values)), eval_target_it)

		
		print(f"[Eval it={eval_target_it}] {key}={np.mean(values):.2f} ± {np.std(values):.2f}")
	
	if save_this_it:
	
		with torch.no_grad():

			vis = image_syn.detach().cpu()
			img_log_dir = os.path.join(run_dir, 'images')
			make_dir(img_log_dir)
			save_path = os.path.join(img_log_dir, f"synthetic_images_{eval_target_it}.png")
			save_image(torch.clamp(denormalize_clip(vis)[:min(64, vis.size(0))], 0, 1), save_path, nrow=8)
			
			# TensorBoard: 이미지 그리드
			grid = to_grid_for_tb(vis[:min(64, vis.size(0))], nrow=8)
			writer.add_image("Synthetic/ImagesGrid", grid, eval_target_it)

			# 히스토그램(픽셀/텍스트)
			writer.add_histogram("Synthetic/Pixels", vis.flatten(), eval_target_it)
			writer.add_histogram("Synthetic/TextValues", text_syn.detach().cpu().flatten(), eval_target_it)

			# (선택) 최근접 캡션 16개만 텍스트로 로깅
			try:
				sentence_list = nearest_neighbor(train_sentences, text_syn.detach().cpu(), train_caption_embed)
				sentence_list = sentence_list[:16]
				writer.add_text("Synthetic/NearestCaptions", "<br>".join(sentence_list), eval_target_it)
			except Exception as _:
				pass
					

				
	
	writer.close()
	print("Done.")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Joint Prototype Bank (JPB) Distillation for Image-Text Retrieval")

	# Data/paths
	parser.add_argument('--dataset', type=str, default='flickr', choices=['flickr', 'coco', 'flickr8k'])
	parser.add_argument('--data_path', type=str, default='./data/Flickr30k/')
	parser.add_argument('--image_root', type=str, default='./data/datasets/Flickr30k/')
	parser.add_argument('--ann_root', type=str, default='./data/Flickr30k_ann/')
	parser.add_argument('--log_dir', type=str, default='./log_cross_arch')

	# Real dataset
	parser.add_argument('--no_aug', action='store_true', help='no_aug')

	# Feature dim (must match projection output)
	parser.add_argument('--feat_dim', type=int, default=768)

	# Synthetic set
	parser.add_argument('--num_queries', type=int, default=100)

	# Optimization
	parser.add_argument('--Iteration', type=int, default=3000)
	parser.add_argument('--batch_syn', type=int, default=64)
	parser.add_argument('--lr_img', type=float, default=100.0)
	parser.add_argument('--lr_txt', type=float, default=100.0)
	parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'])
	parser.add_argument('--momentum', type=float, default=0.5)
	parser.add_argument('--grad_clip', type=float, default=1.0)
	
	# DSA 
	parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate', help='differentiable Siamese augmentation strategy')

	# Init
	parser.add_argument('--pix_init', type=str, default='real', choices=['real', 'noise'])
	parser.add_argument('--txt_init', type=str, default='real', choices=['real', 'noise'])
	parser.add_argument('--image_size', type=int, default=224)

	# CLIP/Model
	parser.add_argument('--image_encoder', type=str, default='nfnet', choices=['nfnet', 'nf_resnet50', 'nf_regnet', 'vit'])
	parser.add_argument('--text_encoder',  type=str, default='bert', choices=['bert', 'clip', 'distilbert'])
	parser.add_argument('--text_pretrained',  type=bool, default=True)
	parser.add_argument('--image_pretrained', type=bool, default=True)
	parser.add_argument('--text_trainable',   type=bool, default=False)
	parser.add_argument('--image_trainable',  type=bool, default=True)
	parser.add_argument('--distill',          type=bool, default=True)
	parser.add_argument('--only_has_image_projection', type=bool, default=False, help='None')

	parser.add_argument('--temperature', type=float, default=0.07)

	# Evaluation
	parser.add_argument('--eval_it', type=int, default=100, help='evaluation interval')
	parser.add_argument('--epoch_eval_train', type=int, default=100, help='epochs to train on synthetic for eval')
	parser.add_argument('--eval_eval_freq', type=int, default=10, help='evaluation frequency during eval training')
	parser.add_argument('--num_eval', type=int, default=5, help='repeat eval training')
	parser.add_argument('--batch_size_train', type=int, default=512, help='batch_size_train')
	parser.add_argument('--batch_train', type=int, default=128, help='batch size for training networks')
	parser.add_argument('--batch_size_test', type=int, default=512, help='batch_size_test')
	parser.add_argument('--lr_teacher_img', type=float, default=0.1, help='eval-time image LR')
	parser.add_argument('--lr_teacher_txt', type=float, default=0.1, help='eval-time text LR')
	parser.add_argument('--loss_type', type=str, default="InfoNCE", help='InfoNCE or WBCE')

	
	# Teachers (buffers)
	parser.add_argument('--buffer_path', type=str, default='/mnt/hoyong3/dataset-distil/LoRS_Distill/buffer_my_seed0123/flickr30k/nfnet_bert/InfoNCE')
	parser.add_argument('--num_buffers', type=int, default=20)
	parser.add_argument('--teacher_resample', type=int, default=50)

	# Misc
	parser.add_argument('--seed', type=int, default=0)
	parser.add_argument('--name', type=str, default='cross_arch')
	parser.add_argument('--log_it', type=int, default=10)
	parser.add_argument('--save_it', type=int, default=200)
	
	# Synthetic Initialize
	parser.add_argument('--syn_init', type=str, default='kmeans', choices=['random', 'kmeans'])
	
	# Cluster DM
	parser.add_argument('--cluster_by', type=str, default='image_text', choices=['image', 'text', 'image_text'])  # 'text' 추후 지원 가능
	parser.add_argument('--cluster_mode', type=str, default='cosine', choices=['cosine','euclidean'])

	# NCE Loss weights (syn optimization)
	parser.add_argument('--w_nce', type=float, default=0.1, help='weight for image-text NCE loss')  #1
 
	# Spherical MMD Geodesic
	parser.add_argument('--w_sph_mmd', type=float, default=1.0, # 5
						help='Weight for spherical MMD loss.')
	parser.add_argument('--sph_mmd_sigma', type=float, default=0.5,
						help='Sigma of geodesic RBF kernel for spherical MMD.')

	# Conditional MMD
	# Conditional Mean Embedding losses
	parser.add_argument('--w_cgap_y', type=float, default=0.0, help='Weight for conditional GAP matching g|y.')     # 0.4
	parser.add_argument('--w_cgap_x', type=float, default=0.0, help='Weight for conditional GAP matching g|x.')     # 0.4
	parser.add_argument('--c_sigma_y', type=float, default=0.5, help='Kernel sigma for conditioning on Y.')
	parser.add_argument('--c_sigma_x', type=float, default=0.5, help='Kernel sigma for conditioning on X.')
	parser.add_argument('--cgap_it_max', type=int, default=5000, help='Maximum iteration that cgap weight increase.')
	parser.add_argument('--c_sigma_g', type=float, default=0.5, help='Kernel sigma for GAP vector G.')
	# Auxiliary losses
	parser.add_argument('--w_cgap_reg', type=float, default=0.0, help='Weight for conditional regularizer loss.')
	parser.add_argument('--w_cgap_rep', type=float, default=0.0, help='Weight for conditional representation loss.')        # 0.05
	
	parser.add_argument('--cgap_schedule', type=str, default='constant', choices=['constant', 'linear', 'log', 'exp', 'sigmoid'],
						help='cgap weight schedule that arise from 0 to w_cgap for cgap_it_max iterations')
	parser.add_argument('--cgap_log_k', type=float, default=9.0, help='log(1+k*a)')
	parser.add_argument('--cgap_exp_k', type=float, default=5.0, help='e^(kt-1) / (e^k - 1)')
	parser.add_argument('--cgap_sig_k', type=float, default=10.0, help='1/(1+e^(-k(t-0.5)))')
	parser.add_argument('--text_embed_dir', type=str, default='text_embeds', help='text embed npz file directory')
	parser.add_argument('--min_start_epoch', type=int, default=1, help='max epoch we can start at')
	parser.add_argument('--max_start_epoch', type=int, default=3, help='max epoch we can start at')
 
 
	parser.add_argument('--init_model_method', type=str, default='default', choices=['default', 'mixed', 'naive'])
 
	parser.add_argument('--eval_target_path', type=str, default='/mnt/hoyong3/dataset-distil/LoRS_Distill/log_cross_arch/11-04_03-14-54_flickr30k_evModelRand_Mws_nce1_wMMD5', help='path to load distilled data for evaluation')
	parser.add_argument('--target_it', type=int, default=3000, help='which iteration distilled data to load for evaluation')
	args = parser.parse_args()
 
		
	args.dsa_param = ParamDiffAug()
	args.dsa = False if args.dsa_strategy in ['none', 'None'] else True
 


	if args.dataset == 'flickr8k':
		args.image_root = './data/datasets/Flickr8k'
		args.ann_root = './data/annotations'
	elif args.dataset == 'flickr':
		args.image_root = './data/datasets/Flickr30k'
		args.ann_root = './data/annotations'
	elif args.dataset == 'coco':
		args.image_root = './data/datasets/COCO'
		args.ann_root = './data/annotations'
	if args.buffer_path is None:
		args.buffer_path = f'./buffer/{args.dataset}/{args.image_encoder}_{args.text_encoder}/{args.loss_type}'
	
	args.log_dir = os.path.join(args.log_dir, __file__.split('/')[-1].split('.')[0])

	torch.set_float32_matmul_precision('high')
	main(args)

# CUDA_VISIBLE_DEVICES=1 python distill_final_crossarch.py --num_queries 100 --num_eval 5 --epoch_eval_train 100 --eval_eval_freq 100 --dataset flickr8k --image_encoder nfnet --text_encoder bert --name nfnet_bert
# CUDA_VISIBLE_DEVICES=1 python distill_final_crossarch.py --num_queries 100 --num_eval 5 --epoch_eval_train 100 --eval_eval_freq 100 --dataset flickr8k --image_encoder nf_resnet50 --text_encoder bert --name nf_resnet50_bert
# CUDA_VISIBLE_DEVICES=1 python distill_final_crossarch.py --num_queries 100 --num_eval 5 --epoch_eval_train 100 --eval_eval_freq 100 --dataset flickr8k --image_encoder nf_regnet --text_encoder bert --name nf_regnet_bert
# CUDA_VISIBLE_DEVICES=1 python distill_final_crossarch.py --num_queries 100 --num_eval 5 --epoch_eval_train 100 --eval_eval_freq 100 --dataset flickr8k --image_encoder vit --text_encoder bert --name vit_bert

# CUDA_VISIBLE_DEVICES=1 python distill_final_crossarch.py --num_queries 100 --num_eval 5 --epoch_eval_train 100 --eval_eval_freq 100 --dataset flickr8k --image_encoder nfnet --text_encoder distilbert --name nfnet_distilbert    
# CUDA_VISIBLE_DEVICES=1 python distill_final_crossarch.py --num_queries 100 --num_eval 5 --epoch_eval_train 100 --eval_eval_freq 100 --dataset flickr8k --image_encoder nf_resnet50 --text_encoder distilbert --name nf_resnet50_distilbert
# CUDA_VISIBLE_DEVICES=1 python distill_final_crossarch.py --num_queries 100 --num_eval 5 --epoch_eval_train 100 --eval_eval_freq 100 --dataset flickr8k --image_encoder nf_regnet --text_encoder distilbert --name nf_regnet_distilbert
# CUDA_VISIBLE_DEVICES=1 python distill_final_crossarch.py --num_queries 100 --num_eval 5 --epoch_eval_train 100 --eval_eval_freq 100 --dataset flickr8k --image_encoder vit --text_encoder distilbert --name vit_distilbert

