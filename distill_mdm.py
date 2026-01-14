import argparse
import copy
import datetime
import glob
import json
import logging
import math
import os
import random
import shutil
import sys
import gc
import time
from collections import defaultdict, OrderedDict
from typing import List, Tuple
from dataclasses import dataclass, field
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from rich import print
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import logger
from torchvision.utils import save_image
from tqdm import tqdm
import wandb
import prettytable


# ===== Project/LoRS modules (expected in repo) =====
from data import get_dataloaders
from src.clustering_utils import KMeansCluster, OnlineCentroidBank
from src.epoch import epoch_test, itm_eval, evaluate_synset
from src.networks import CLIPModel_full
from src.utils import DiffAugment, ParamDiffAug
from src.vl_distill_utils import (
	shuffle_files, nearest_neighbor, get_images_texts, load_or_process_file, textprocess_test, textprocess_train
)


def print_memory_usage():
	# Get allocated memory
	allocated_memory = torch.cuda.memory_allocated()

	# Get reserved memory (including cached memory)
	reserved_memory = torch.cuda.memory_reserved()

	# Get peak allocated memory
	peak_allocated_memory = torch.cuda.max_memory_allocated()

	print(f"Allocated GPU Memory: {allocated_memory / (1024**2):.2f} MB | Reserved GPU Memory: {reserved_memory / (1024**2):.2f} MB | Peak Allocated GPU Memory: {peak_allocated_memory / (1024**2):.2f} MB")

	wandb.log({
		"GPUMemory/Allocated": allocated_memory / (1024**2),
		"GPUMemory/Reserved": reserved_memory / (1024**2),
		"GPUMemory/PeakAllocated": peak_allocated_memory / (1024**2),
	})
	# Reset peak memory statistics
	torch.cuda.reset_peak_memory_stats()

@dataclass
class WallClockTracker:
	metric_name: str = "r_mean"            # changeable by arg
	run_start_s: float = field(default_factory=time.perf_counter)
	best_val: float = float("-inf")
	best_time_s: float | None = None
	target_value: float | None = None      # optional: time-to-target
	target_time_s: float | None = None
	epsilon: float = 1e-9
	history: list = field(default_factory=list)   # [{it, t_s, metric}]

	def stamp_eval(self, it: int, value: float):
		# Ensure device work finished before timing
		if torch.cuda.is_available():
			torch.cuda.synchronize()
		now = time.perf_counter()
		elapsed = now - self.run_start_s
		self.history.append({"it": int(it), "t_s": float(elapsed), "metric": float(value)})

		is_new_best = (value > self.best_val + self.epsilon)
		if is_new_best:
			self.best_val = float(value)
			self.best_time_s = float(elapsed)

		if (self.target_value is not None) and (self.target_time_s is None) and (value >= self.target_value):
			self.target_time_s = float(elapsed)

		return {
			"new_best": bool(is_new_best),
			"elapsed_s": float(elapsed),
			"best_time_s": float(self.best_time_s) if self.best_time_s is not None else None,
			"best_val": float(self.best_val),
		}

	def finalize(self, out_dir: str):
		path = os.path.join(out_dir, "wallclock_times.json")
		with open(path, "w") as f:
			json.dump({
				"metric": self.metric_name,
				"best_val": self.best_val,
				"time_to_best_s": self.best_time_s,
				"target_value": self.target_value,
				"time_to_target_s": self.target_time_s,
				"history": self.history,
			}, f, indent=2)
		return path

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

class AverageMeter:
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

def compute_ratio(angle_dict, k=2):
	ratio_dict = {}
	for key in angle_dict.keys():
		angle = np.deg2rad(angle_dict[key])
		ratio_dict[key] = k*np.cos(angle) / ((k-1)*np.cos(angle)+1+EPS)

	return ratio_dict 

def compute_angle(state_dict_1, state_dict_2, ref_state_dict, add_ignore_keys=[], return_cos=False, device='cuda'):
	ignore_keys = []
	return_dict = OrderedDict()

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

			cosine_val = torch.sum(vector1 * vector2) / (torch.sqrt(torch.sum(vector1 ** 2) * torch.sum(vector2 ** 2))+EPS)
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
	cos_dict = OrderedDict()
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
def fast_merge(w1, w2, w0, ratio, device='cuda', dtype=torch.float32, non_blocking=True, alpha=1.0):
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
	
	# ratio_sum = 0 
	# ratio_count = 0

	for k in w0.keys():
		# 세 dict 모두에 있어야 하고, float tensor이면서 shape 동일해야 안전
		if (k in w1) and (k in w2) \
		   and torch.is_tensor(w1[k]) and torch.is_tensor(w2[k]) and torch.is_tensor(w0[k]) \
		   and w1[k].is_floating_point() and w2[k].is_floating_point() and w0[k].is_floating_point() \
		   and (w1[k].shape == w2[k].shape == w0[k].shape):

			t1 = w1[k].to(device=device, dtype=dtype, non_blocking=non_blocking)
			t2 = w2[k].to(device=device, dtype=dtype, non_blocking=non_blocking)
			student_net = w0[k].to(device=device, dtype=dtype, non_blocking=non_blocking)

			w12 = 0.5 * (t1 + t2)  # 평균

			# 기본은 당신 코드처럼 w12로 초기화 (ratio에 key가 없으면 평균 유지)
			outk = w12

			# ratio에 key가 있으면 w0와 보간
			if k in ratio:
				r = torch.as_tensor(ratio[k], dtype=dtype, device=device)
				# r*w12 + (1-r)*w0  ==  w0 + r*(w12 - w0)  (연산 1번으로 축약)
				outk = student_net + (r * (w12 - student_net) * alpha)
				
				# print(r)
				# ratio_sum += float(r)
				# ratio_count += 1

			w_merge[k] = outk

		else:
			# float 가 아니거나 shape가 다르면 원본 w0를 그대로 둡니다 (안전한 기본값)
			w_merge[k] = w0[k]

	# print('Average ratio used in merge:', ratio_sum / (ratio_count + EPS))

	return w_merge

def load_model_state_dict(state_dict, map_location='cpu'):
	state_dict = torch.load(state_dict, map_location=map_location)
	return state_dict

def make_distillation_model(args, student_net, merge_image=True, merge_text=True, verbose=False, merge_alpha=None):
	
	BASE_DIR = args.buffer_path
	
	total_img_buffers = args.num_buffers-1 #19# if 'flickr' in args.dataset else 11 if 'coco' in args.dataset else 0
	assert total_img_buffers > 0, "total_img_buffers is 0"

	MAX_EPOCH = args.max_start_epoch
	MIN_EPOCH = 1

	EXPERT_NUM1 = random.randint(0, total_img_buffers) # inclusive
	EPOCH_NUM1 = random.choice(range(MIN_EPOCH, MAX_EPOCH+1)) 
	img_file = os.path.join(BASE_DIR, f'img_replay_buffer_{EXPERT_NUM1}_{EPOCH_NUM1}.pth')
	txt_file = os.path.join(BASE_DIR, f'txt_replay_buffer_{EXPERT_NUM1}_{EPOCH_NUM1}.pth')
	img_expert_1 = load_model_state_dict(img_file, map_location=args.device)
	txt_expert_1 = load_model_state_dict(txt_file, map_location=args.device)
	
	EXPERT_NUM2 = random.randint(0, total_img_buffers)
	epoch_pool = [i for i in range(MIN_EPOCH, MAX_EPOCH+1) if i != EPOCH_NUM1]
	if EXPERT_NUM1 == EXPERT_NUM2:
		epoch_pool = [i for i in range(MIN_EPOCH, MAX_EPOCH+1) if i != EPOCH_NUM1]
	else:
		epoch_pool = [i for i in range(MIN_EPOCH, MAX_EPOCH+1)]
	EPOCH_NUM2 = random.choice(epoch_pool)
	img_file = os.path.join(BASE_DIR, f'img_replay_buffer_{EXPERT_NUM2}_{EPOCH_NUM2}.pth')
	txt_file = os.path.join(BASE_DIR, f'txt_replay_buffer_{EXPERT_NUM2}_{EPOCH_NUM2}.pth')
	img_expert_2 = load_model_state_dict(img_file, map_location=args.device)
	txt_expert_2 = load_model_state_dict(txt_file, map_location=args.device)
	
	print(f'[yellow]Selecting expert models: ({EXPERT_NUM1}, {EPOCH_NUM1}), ({EXPERT_NUM2}, {EPOCH_NUM2})[/yellow]')

	# check if the expert state dicts are valid
	assert isinstance(img_expert_1, dict) and isinstance(txt_expert_1, dict) and isinstance(img_expert_2, dict) and isinstance(txt_expert_2, dict)
	
	# initial model
	student_net.image_encoder.to(args.device)
	student_net.text_projection.to(args.device)
	img_initial = student_net.image_encoder.state_dict()
	txt_initial = student_net.text_projection.state_dict()
	
	if merge_image:
		cos_img = compute_cosine_dict(img_expert_1, img_expert_2, img_initial, device=args.device)
		ratio_img = compute_ratio_from_cos(cos_img, k=2.0)
		merged_img_model = fast_merge(img_expert_1, img_expert_2, img_initial, ratio_img, device=args.device, alpha=args.merge_alpha)
	else:
		merged_img_model = None

	if merge_text:
		cos_txt = compute_cosine_dict(txt_expert_1, txt_expert_2, txt_initial, device=args.device)
		ratio_txt = compute_ratio_from_cos(cos_txt, k=2.0)
		merged_txt_model = fast_merge(txt_expert_1, txt_expert_2, txt_initial, ratio_txt, device=args.device, alpha=args.merge_alpha)
	else:
		merged_txt_model = None

	assert merged_img_model is not None and merged_txt_model is not None, "merged_img_model or merged_txt_model is None"
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
	from torchvision.utils import make_grid
	return make_grid(x_denorm.cpu(), nrow=nrow, padding=0)

def _norm_t(it, T):
	if T is None or T <= 0:
		return 1.0
	t = it / float(T)
	return max(0.0, min(1.0, t))
# =====================================================
# Teachers (buffer mix)
# =====================================================

# def load_model_state_dict(state_dict):
# 	state_dict = torch.load(state_dict, map_location=args.device)
# 	return state_dict

def _load_clip_from_buffers(args):
	k = random.randint(0, args.num_buffers - 1)
	img_path = os.path.join(args.buffer_path, f'img_replay_buffer_{k}_10.pth')
	txt_path = os.path.join(args.buffer_path, f'txt_replay_buffer_{k}_10.pth')
	img_sd = torch.load(img_path, map_location=args.device)
	txt_sd = torch.load(txt_path, map_location=args.device)
 
	
	return img_sd, txt_sd, k


def _param_interpolate_(tgt_model, sd_a, sd_b, alpha):
	with torch.no_grad():
		msd = tgt_model.state_dict()
		for k in msd.keys():
			if k in sd_a and k in sd_b and torch.is_tensor(sd_a[k]) and torch.is_tensor(sd_b[k]):
				if msd[k].shape == sd_a[k].shape == sd_b[k].shape and msd[k].dtype == sd_a[k].dtype == sd_b[k].dtype:
					msd[k].copy_(alpha * sd_a[k] + (1.0 - alpha) * sd_b[k])
		tgt_model.load_state_dict(msd)

def _param_interpolate_return(sd_a, sd_b, alpha):
	interpolated_sd = {}
	with torch.no_grad():
		for k in sd_a.keys():
			if k in sd_b and torch.is_tensor(sd_a[k]) and torch.is_tensor(sd_b[k]):
				if sd_a[k].shape == sd_b[k].shape and sd_a[k].dtype == sd_b[k].dtype:
					interpolated_sd[k] = alpha * sd_a[k] + (1.0 - alpha) * sd_b[k]
	return interpolated_sd


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
	# img_expert_files = glob.glob(os.path.join(BASE_DIR, f'img_{FILE_FORMAT}_*.pth')) # list of filenames
	# txt_expert_files = glob.glob(os.path.join(BASE_DIR, f'txt_{FILE_FORMAT}_*.pth')) # list of filenames
	# total_img_buffers = len(img_expert_files)-1
 
	# EXPERT_NUM1 = random.randint(0, total_img_buffers)
	# EPOCH_NUM1 = random.choice(range(1, args.max_start_epoch+1))	
	# pretrain_img_sd = load_model_state_dict(os.path.join(BASE_DIR, F'img_{FILE_FORMAT}_{EXPERT_NUM1}.pth'))[0][EPOCH_NUM1]
	# pretrain_txt_sd = load_model_state_dict(os.path.join(BASE_DIR, F'txt_{FILE_FORMAT}_{EXPERT_NUM1}.pth'))[0][EPOCH_NUM1]
	 
	pretrain_img_sd, pretrain_txt_sd, chosen_k = _load_clip_from_buffers(args)
  
	mixed = CLIPModel_full(args, temperature=args.temperature).to(device)
	_param_interpolate_(mixed.image_encoder, pretrain_img_sd, rand_img_sd, alpha)
	_param_interpolate_(mixed.text_projection, pretrain_txt_sd, rand_txt_sd, alpha)

	mixed.eval()
	for p in mixed.parameters():
		p.requires_grad_(False)
	# return mixed, alpha, EXPERT_NUM1
	return mixed, alpha, chosen_k


def make_teachers(args, device, n_teachers: int, alpha_from=None, alpha_to=None) -> List[CLIPModel_full]:
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

def infonce(img_feats: torch.Tensor, txt_feats: torch.Tensor, temperature: float = 0.07):
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
def _pairwise_cosine(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
	"""
	Cosine similarity between rows of x and y, with L2 normalization.
	x: [N, D], y: [M, D] -> [N, M]
	"""
	x_n = F.normalize(x, dim=1)
	y_n = F.normalize(y, dim=1)
	cosv = x_n @ y_n.T
	return cosv.clamp(-1.0 + eps, 1.0 - eps)

# === add near Losses & Assignments ===
def geodesic_distance(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
	# a,b: [N,D] L2-normalized 가정(안전하게 내부 normalize)
	a = F.normalize(a, dim=1)
	b = F.normalize(b, dim=1)
	cosv = (a @ b.t()) if a.dim()==2 and b.dim()==2 else (a*b).sum(dim=-1, keepdim=False)
	cosv = cosv.clamp(-1.0 + eps, 1.0 - eps)
	return torch.acos(cosv)  # [N,N] or [N] or scalar-like

# def geodesic_loss_pair(a, b, squared: bool = True) -> torch.Tensor:
# 	theta = geodesic_distance(a, b)  # broadcast 가능
# 	return (theta * theta).mean() if squared else theta.mean()

def spherical_rbf_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float = 0.5, method='cosine') -> torch.Tensor:
	# k(x,y) = exp( - geodesic(x,y)^2 / (2 sigma^2) )

	if method == 'geodesic':
		# geodesic rbf kernel
		theta = geodesic_distance(x, y)   # [Nx, Ny]
		return torch.exp( - (theta * theta) / (2.0 * sigma * sigma) )

	elif method == 'laplacian':
		# laplacian rbf kernel
		diff = x.unsqueeze(1) - y.unsqueeze(0)     # [Nx, Ny, D]
		dist = diff.abs().sum(-1)                  # [Nx, Ny], L1 norm
		return torch.exp(-dist / sigma)
 
	elif method == 'euclidean':
		# euclidean rbf kernel
		diff = x.unsqueeze(1) - y.unsqueeze(0)  # [Nx, Ny, D]
		dist2 = (diff ** 2).sum(-1)  # [Nx, Ny]
		return torch.exp(-dist2 / (2.0 * sigma * sigma))
 
	elif method == 'chordal':
		# chordal rbf
		# d_chord(x, y)^2 = 2 - 2⟨x, y⟩   for ||x|| = ||y|| = 1
		# k(x, y) = exp(-d_chord^2 / (2 * sigma^2))
		cosv = _pairwise_cosine(x, y, eps=EPS)         # [N, M]
		dist2 = 2.0 - 2.0 * cosv                       # chordal distance squared
		return torch.exp(-dist2 / (2.0 * sigma ** 2))

	elif method == 'cosine':
		cosv = _pairwise_cosine(x, y, eps=EPS)
		d_cos = 1.0 - cosv
		dist2 = d_cos * d_cos
		return torch.exp(-dist2 / (2.0 * sigma * sigma))
 
 
 
	# diff = x.unsqueeze(1) - y.unsqueeze(0)  # [Nx, Ny, D]
	# dist2 = (diff).sum(-1)  # [Nx, Ny]
	# return torch.exp(-dist2 / (2.0 * sigma * sigma))
	

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
# key_syn:  [B,  D]  (e.g., feat_txt_syn or feat_img_syn)
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
	stopgrad_W: bool = True,
	eps: float = 1e-12,
	diversity_weight: float = 1.0,
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
	# L_reg = entropy_reg_kl_to_uniform(W)
	
	# 4) (옵션) syn key repulsion
	# keys_syn = F.normalize(key_syn, dim=1)
	# L_rep = repulsion_loss_spherical(keys_syn, sigma_key=sigma_key)
	
	if stopgrad_W:
		W = W.detach() # real-syn key gap

	# 2) g-space kernels
	Kg_sr = spherical_rbf_kernel(g_syn,  g_real, sigma=sigma_g)  # [B, Br]
	Kg_rr = spherical_rbf_kernel(g_real, g_real, sigma=sigma_g)  # [Br, Br]

	# 3) Per-sample conditional MMD^2 for each syn i:
	#    term1 = k(g_i, g_i) = 1
	#    term2 = 2 * <k(g_i,·), mu_{p(g|key_i)}> = 2 * sum_j w_{ji} k(g_i, g_j)
	#    term3 = ||mu||^2 = w_i^T K_rr w_i
	# term1 = torch.ones(B, device=device)
	
	# (Kg_sr * W^T).sum(dim=1) == sum_j w_{ji} k(g_i,g_j)
	# term2 = 2.0 * (Kg_sr * W.t()).sum(dim=1)  # [B]

	# W: [Br, B]  ->  W^T K_rr W  : [B, B]
	# diag gives each sample's w_i^T K_rr w_i
	# (mm 순서: (W^T K_rr) 먼저 해서 메모리 절약)
	# M = (W.t() @ Kg_rr) @ W         # [B, B]
	# term3 = torch.diagonal(M, dim1=-2, dim2=-1)  # [B]

	L_cgap_mmd = diversity_weight * torch.ones(B, device=device) + \
					torch.diagonal(((W.t() @ Kg_rr) @ W), dim1=-2, dim2=-1) - \
					2.0 * (Kg_sr * W.t()).sum(dim=1)	
	L_cgap_mmd = L_cgap_mmd.mean()
	# return L_cgap_mmd, L_reg, L_rep
	return L_cgap_mmd

# def entropy_and_coverage_regularizer(W: torch.Tensor, *, eps: float = 1e-12):
# 	"""
# 	W: [Br, B] (real-by-syn), columns sum to 1
# 	반환:
# 	  L_ent:  - mean entropy (값이 작아지면 엔트로피 커짐)  -> loss에 +λ_ent * L_ent
# 	  L_cov:  row-sum 균형 손실 (coverage 균형)              -> loss에 +λ_cov * L_cov
# 	"""
# 	# --- (A) Column entropy: encourage high entropy per column ---
# 	# H_i = - sum_j W_{j,i} log W_{j,i}
# 	Wc = (W + eps)
# 	H_col = -(Wc * Wc.log()).sum(dim=0)               # [B]
# 	L_ent = - H_col.mean()                             # minimize(-mean H) => maximize mean H

# 	# --- (B) Row coverage: encourage balanced usage of reals ---
# 	row_sum = W.sum(dim=1)                             # [Br]
# 	target = W.size(1) / W.size(0)                     # B/Br
# 	L_cov = torch.mean((row_sum - target) ** 2)
# 	return L_ent, L_cov

# def entropy_reg_kl_to_uniform(W: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
# 	# W: [Br, B], columns sum to 1
# 	Br = W.size(0)
# 	Wc = (W + eps)
# 	# mean over columns of sum_j w log(w*Br)
# 	kl = (Wc * (Wc * Br).log()).sum(dim=0).mean()
# 	return kl  # >= 0, 0 at perfectly uniform

# def repulsion_loss_spherical(keys_syn: torch.Tensor, sigma_key: float = 0.5):
# 	# keys_syn: [B,D], unit-norm
# 	Kss = spherical_rbf_kernel(keys_syn, keys_syn, sigma=sigma_key)  # [B,B]
# 	offdiag = (Kss.sum() - Kss.diagonal().sum()) / max(1, (Kss.numel() - Kss.size(0)))
# 	return offdiag  # minimize => push apart

# ==== Product-kernel MMD^2 for joint variables (A,B) ====
# def mmd2_product_kernel(
# 	A_syn: torch.Tensor, B_syn: torch.Tensor,
# 	A_real: torch.Tensor, B_real: torch.Tensor,
# 	sigma_A: float, sigma_B: float,
# ) -> torch.Tensor:
# 	"""
# 	A_*, B_*: [N, D] vectors (구면 비교 권장 → 내부에서 normalize는 spherical_rbf_kernel이 처리)
# 	커널: k((a,b),(a',b')) = kA(a,a') * kB(b,b')
# 	반환: unbiased MMD^2 (scalar)
# 	"""
# 	# 개별 커널들 (지오데식 RBF)
# 	KA_ss = spherical_rbf_kernel(A_syn,  A_syn,  sigma=sigma_A)  # [Ns,Ns]
# 	KA_rr = spherical_rbf_kernel(A_real, A_real, sigma=sigma_A)  # [Nr,Nr]
# 	KA_sr = spherical_rbf_kernel(A_syn,  A_real, sigma=sigma_A)  # [Ns,Nr]

# 	KB_ss = spherical_rbf_kernel(B_syn,  B_syn,  sigma=sigma_B)  # [Ns,Ns]
# 	KB_rr = spherical_rbf_kernel(B_real, B_real, sigma=sigma_B)  # [Nr,Nr]
# 	KB_sr = spherical_rbf_kernel(B_syn,  B_real, sigma=sigma_B)  # [Ns,Nr]

# 	# 제품 커널(아다마르 곱)
# 	K_ss = KA_ss * KB_ss
# 	K_rr = KA_rr * KB_rr
# 	K_sr = KA_sr * KB_sr

# 	# unbiased estimator (네 코드의 offdiag_mean 규칙과 동일)
# 	def offdiag_mean(K):
# 		n = K.size(0)
# 		if K.size(0) == K.size(1) and n > 1:
# 			return (K.sum() - K.diag().sum()) / (n * (n - 1))
# 		else:
# 			return K.mean()

# 	mmd2 = offdiag_mean(K_ss) + offdiag_mean(K_rr) - 2.0 * K_sr.mean()
# 	return mmd2

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

def make_timestamp(prefix: str="", suffix: str="") -> str:
	KST_TIMEZONE = 9
	tmstamp = datetime.datetime.now() + datetime.timedelta(hours=KST_TIMEZONE)
	tmstamp = '{:%m_%d_%Y_%H%M%S}'.format(tmstamp)
	return tmstamp
	
def cross_cov(X, Y):
	Xc = X - X.mean(0, keepdim=True)
	Yc = Y - Y.mean(0, keepdim=True)
	return (Xc.T @ Yc) / X.size(0)


# remove diagonals for unbiasedness (if square)
def offdiag_mean(K):
	n = K.size(0)
	if K.size(0) == K.size(1) and n > 1:
		return (K.sum() - K.diag().sum()) / (n*(n-1))
	else:
		return K.mean()
		

def set_seed(seed):	
	from lightning.fabric import seed_everything
	seed_everything(seed)
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.cuda.empty_cache()
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	# torch.use_deterministic_algorithms(True)
	os.environ['PYTHONHASHSEED'] = str(seed)


def clean_cache():
	import gc
	gc.collect()
	torch.cuda.empty_cache()

# Ensure the weights of student_net.image_encoder.state_dict() differ from orig_img_model (values, not just keys)
def _weights_differ(state1, state2):
	for k in state1:
		val1 = state1[k]
		val2 = state2[k]
		# Assure both are on the same device as the first tensor
		target_device = val1.device
		val1 = val1.float().to(target_device)
		val2 = val2.float().to(target_device)
		if not torch.equal(val1, val2):
			return True
	return False

def _states_differ(state1, state2):
	if set(state1.keys()) != set(state2.keys()):
		return True
	for k in state1:
		val1 = state1[k]
		val2 = state2[k]
		target_device = val1.device
		val1 = val1.float().to(target_device)
		val2 = val2.float().to(target_device)
		if not torch.equal(val1, val2):
			return True
	return False

start_datainit = torch.cuda.Event(enable_timing=True)
end_datainit = torch.cuda.Event(enable_timing=True)
start_model_init = torch.cuda.Event(enable_timing=True)
end_model_init = torch.cuda.Event(enable_timing=True)
start_distillation = torch.cuda.Event(enable_timing=True)
end_distillation = torch.cuda.Event(enable_timing=True)
start_distillation_before_backward = torch.cuda.Event(enable_timing=True)
end_distillation_before_backward = torch.cuda.Event(enable_timing=True)

EPS = 1e-12 
# =====================================================
# Main
# =====================================================
def main(args):
	args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	set_seed(args.seed)
	clean_cache()

	log_dir = f"{args.log_dir}/{args.dataset}/{args.image_encoder}/{args.text_encoder}/{args.loss_type}/{args.num_queries}/{args.init_model_method}/{make_timestamp()}/{args.name}"
	os.makedirs(log_dir, exist_ok=True, mode=0o777)
	args.log_dir = log_dir
	
	# set up logger
	filename = os.path.basename(__file__).replace('.py', '')
	formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s', datefmt='%m_%d_%Y_%H%M%S')
	logger = logging.getLogger(filename)
	logger.setLevel(logging.INFO)
	# sysout_handler = logging.StreamHandler(sys.stdout)
	# sysout_handler.setFormatter(formatter)
	# logger.addHandler(sysout_handler)
	filehandler = logging.FileHandler(os.path.join(log_dir, f"distill_{args.dataset}_{args.image_encoder}_{args.text_encoder}.log"), mode='a+')
	filehandler.setFormatter(formatter)
	logger.addHandler(filehandler)
	logger.info(to_jsonable(args))
	print("args:", sorted(args.__dict__.items(), key=lambda x: x[0]))
	with open(os.path.join(log_dir, "args.json"), "w") as f:
		json.dump(to_jsonable(args), f, indent=4)
	shutil.copy(__file__, os.path.join(log_dir, os.path.basename(__file__)))

	log_name = f'{make_timestamp()}_{args.dataset}_{args.num_queries}_nce_{args.w_nce}_sph_u_mmd_{args.w_sph_u_mmd}_sph_g_mmd_{args.w_sph_g_mmd}_cgap_x_{args.w_cgap_x}_cgap_y_{args.w_cgap_y}_cgap_u_{args.w_cgap_u}_sqrtmmd_{args.sqrtmmd}_logmmd_{args.logmmd}_diversity_weight_u_{args.diversity_weight_u}_diversity_weight_g_{args.diversity_weight_g}_merge_alpha_{args.merge_alpha}_lr_txt_{args.lr_txt}_lr_img_{args.lr_img}_modelinit_{args.init_model_method}_datainit_{args.syn_init}_{args.name}'
	if args.wandb:
		wandb.init(
			entity=args.wandb_entity,
			project=args.wandb_project,
			name=log_name+'__52.163',
			config=args,
			dir=log_dir,
			mode=args.wandb_mode,
		)
	else:
		wandb.init(mode = 'disabled')

		
	# set up tensorboard writer
	tensorboard_log_dir = os.path.join(log_dir, 'tensorboard', log_name)
	writer = SummaryWriter(log_dir=tensorboard_log_dir[:255])
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

	
	##########################################################
	# initialize synthetic data
	##########################################################
	image_syn, text_syn = get_images_texts(args.num_queries, train_dataset, args, init='random')
	start_datainit.record()
	mock_start = time.time()
	student_net_kmeans = None
	if args.syn_init == 'kmeans':
		print('[yellow]Initializing synthetic images/texts using k-means clustering.[/yellow]')
		# k-means clustering for image initialization
		if args.init_model_method == 'default':
			model_for_cluster, alphas_for_cluster = make_teachers(args, args.device, n_teachers=1, alpha_from=0.5, alpha_to=0.5)
			model_for_cluster = model_for_cluster[0]
			model_for_cluster.eval()
			
			# ---- P0: use the same teacher for training (fixed) ----
			student_net_kmeans = model_for_cluster     # 고정 teacher
			a0 = alphas_for_cluster[0] # 로그용

		elif args.init_model_method == 'mixed':
			student_net = CLIPModel_full(args, temperature=args.temperature)
			student_net.eval()
			orig_img_model = copy.deepcopy(student_net.image_encoder.state_dict())
			orig_txt_model = copy.deepcopy(student_net.text_projection.state_dict())

			merged_img_model, merged_txt_model = make_distillation_model(args, student_net, merge_image=True, merge_text=True)
			student_net.image_encoder.load_state_dict(copy.deepcopy(merged_img_model))
			student_net.text_projection.load_state_dict(copy.deepcopy(merged_txt_model))

			assert _weights_differ(student_net.image_encoder.state_dict(), orig_img_model), "student_net.image_encoder.state_dict() weights are still equal to orig_img_model"
			assert _weights_differ(student_net.text_projection.state_dict(), orig_txt_model), "student_net.text_projection.state_dict() weights are still equal to orig_txt_model"
			assert _states_differ(student_net.image_encoder.state_dict(), orig_img_model), "student_net.image_encoder.state_dict() is still equal to orig_img_model"
			assert _states_differ(student_net.text_projection.state_dict(), orig_txt_model), "student_net.text_projection.state_dict() is still equal to orig_txt_model"

			student_net_kmeans = copy.deepcopy(student_net)
			a0 = 0	

		elif args.init_model_method == 'naive':
			student_net = CLIPModel_full(args, temperature=args.temperature)
			student_net.eval()
			orig_img_model = copy.deepcopy(student_net.image_encoder.state_dict())
			orig_txt_model = copy.deepcopy(student_net.text_projection.state_dict())

			# random expert
			EXPERT_NUM = random.randint(0, args.num_buffers-1)
			EPOCH_NUM = random.randint(args.min_start_epoch, args.max_start_epoch)
			img_file = os.path.join(args.buffer_path, F'img_replay_buffer_{EXPERT_NUM}_{EPOCH_NUM}.pth')
			txt_file = os.path.join(args.buffer_path, F'txt_replay_buffer_{EXPERT_NUM}_{EPOCH_NUM}.pth')
			img_expert = torch.load(img_file)
			txt_expert = torch.load(txt_file)
			
			# naive mix
			min_ratio, max_ratio = args.naive_mix_min_ratio, args.naive_mix_max_ratio
			ratio = random.uniform(min_ratio, max_ratio)
			for key in img_expert.keys():
				img_expert[key] = img_expert[key].clone().to('cpu') * ratio + orig_img_model[key].clone().to('cpu') * (1.0 - ratio)
			for key in txt_expert.keys():
				txt_expert[key] = txt_expert[key].clone().to('cpu') * ratio + orig_txt_model[key].clone().to('cpu') * (1.0 - ratio)

			# update student_net
			student_net.image_encoder.load_state_dict(copy.deepcopy(img_expert))
			student_net.text_projection.load_state_dict(copy.deepcopy(txt_expert))
	
			student_net_kmeans = student_net
			a0 = 0
			last_refresh = 0

		elif args.init_model_method == 'none':
			student_net_kmeans = CLIPModel_full(args, temperature=args.temperature)
			student_net_kmeans = student_net_kmeans.to(args.device).eval()
			a0 = 0
			last_refresh = 0

		elif args.init_model_method == 'expert':
			student_net = CLIPModel_full(args, temperature=args.temperature)
			student_net = student_net.to(args.device).eval()
			EXPERT_NUM = random.randint(0, args.num_buffers-1)
			EPOCH_NUM = random.randint(args.min_start_epoch, args.max_start_epoch)
			img_file = os.path.join(args.buffer_path, F'img_replay_buffer_{EXPERT_NUM}_{EPOCH_NUM}.pth')
			txt_file = os.path.join(args.buffer_path, F'txt_replay_buffer_{EXPERT_NUM}_{EPOCH_NUM}.pth')
			img_expert = torch.load(img_file)
			txt_expert = torch.load(txt_file)
			
			# update student_net
			student_net_kmeans.image_encoder.load_state_dict(copy.deepcopy(img_expert))
			student_net_kmeans.text_projection.load_state_dict(copy.deepcopy(txt_expert))

			student_net_kmeans = student_net
			a0 = 0
			last_refresh = 0

		assert student_net_kmeans is not None, "student_net_kmeans is None"

		student_net_kmeans = student_net_kmeans.to(args.device)
		student_net_kmeans.eval()
		for p in student_net_kmeans.parameters():
			p.requires_grad_(False)
		
		# test student_net accuracy
		score_val_i2t, score_val_t2i = epoch_test(test_loader, student_net_kmeans, args.device, bert_test_embed)
		val_result = itm_eval(score_val_i2t, score_val_t2i, test_loader.dataset.txt2img, test_loader.dataset.img2txt) 
		print("[Eval][k-means] | Text-to-Image R@1={img_r1:.2f} R@5={img_r5:.2f} R@10={img_r10:.2f} R_Mean={img_r_mean:.2f} | Image-to-Text R@1={txt_r1:.2f} R@5={txt_r5:.2f} R@10={txt_r10:.2f} R_Mean={txt_r_mean:.2f} | R_Mean={r_mean:.2f}".format(
			**val_result
		))

		all_feats = []
		all_dataset_indices = []   
		
		kmeans_train_loader = torch.utils.data.DataLoader(
			train_dataset,
			batch_size=512, #2048,
			shuffle=False,  # no shuffle to keep indices
			drop_last=False,
			num_workers=4,
		)

		with torch.no_grad():
			for img, _, real_ds_inds in tqdm(kmeans_train_loader, desc='Extracting features for k-means', ncols=60):
				img = img.to(args.device, non_blocking=True)
				feat_img = student_net_kmeans.image_encoder(img)
				if hasattr(student_net_kmeans, 'image_projection'):
					feat_img = student_net_kmeans.image_projection(feat_img.float())
				feat_img = F.normalize(feat_img.float(), dim=1)
				
				feat_txt = student_net_kmeans.text_projection(train_caption_embed[real_ds_inds].to(args.device).float())
				feat_txt = F.normalize(feat_txt.float(), dim=1)
				
				if args.cluster_by == 'image':
					all_feats.append(feat_img.cpu()) 
				elif args.cluster_by == 'text':
					all_feats.append(feat_txt.cpu())
				elif args.cluster_by == 'image_text':
					# all_feats.append(F.normalize(feat_img + feat_txt, dim=1).cpu())
					all_feats.append(F.normalize(torch.cat([feat_img, feat_txt], dim=1), dim=1).cpu()) # concatenated features
				all_dataset_indices.append(torch.as_tensor(real_ds_inds, dtype=torch.long))

		
		# cat은 feature만
		all_feats = torch.cat(all_feats, dim=0)            # [N, D], CPU tensor
		all_dataset_indices = torch.cat(all_dataset_indices, 0)    # [N]

		with KMeansCluster(all_feats.to(args.device), mode=args.cluster_mode) as kmeans_unit: # GPU kmeans
			query_feat_indices = kmeans_unit.query_incluster_n(args.num_queries).cpu()  # indices of feature array

			selected_ds_indices = all_dataset_indices[query_feat_indices]  # indices of original dataset

			image_syn = torch.stack([kmeans_train_loader.dataset[i][0] for i in selected_ds_indices.tolist()]).to(args.device)
			text_syn = train_caption_embed[selected_ds_indices].to(args.device).float()
	
			if args.kmeans_viz:
				kmeans_unit.visualize(
					selected_indices=query_feat_indices.to(args.device),
					filename=os.path.join(log_dir, 'kmeans_init_clusters.png'),
					title=f'K-means init (k={args.num_queries})',
					max_points=80000
				)
				print(f'[yellow]viz saved to {log_dir}/kmeans_init_clusters.png[/yellow]')	
		del kmeans_train_loader

	elif args.syn_init == 'noise':
		mean = torch.tensor([-0.0626, -0.0221,  0.0680])
		std  = torch.tensor([1.0451, 1.0752, 1.0539])
		image_syn = torch.randn([args.num_queries, 3, 224, 224])
		for c in range(3):
			image_syn[:, c] = image_syn[:, c] * std[c] + mean[c]
		print('Initialized synthetic images from random noise.')
	
		text_syn = torch.normal(mean=-0.0094, std=0.5253, size=(args.num_queries, 768))
		print('Initialized synthetic texts from random noise.')

	 
	 
	time_taken_time = time.time()-mock_start
	print(f"data initialization (time module): {time_taken_time:.2f} seconds")
	end_datainit.record()
	torch.cuda.synchronize()
	data_init_times = []
	time_taken = start_datainit.elapsed_time(end_datainit)/1000
	data_init_times.append(time_taken)	
	print(f"Time taken to apply data initialization: {time_taken:.2f} seconds")

	# Save the initialized synthetic image and text embeddings and original real text captions to file for reproducibility/debugging
	init_save_path = os.path.join(log_dir, f"init_image_text_embeddings_{make_timestamp()}.pt")
	torch.save({
		"image_syn": image_syn.detach().cpu(),
		"text_syn": text_syn.detach().cpu(),
		"real_train_caption_embed": train_caption_embed.detach().cpu(),
		"real_train_captions": train_sentences if 'train_sentences' in locals() else [],
		"args": to_jsonable(args)
	}, init_save_path)
	print(f"[yellow]Saved initialized synthetic image/text embeddings and original real text captions to {init_save_path}[/yellow]")

	# Save snapshot plot of initial synthetic images and show the nearest neighboring text captions
	# Save in batches of 4 (2x2 grid) for all images
	# try:
	# 	import matplotlib.pyplot as plt
	# 	
	# 	import textwrap

	# 	# Get images
	# 	imgs = image_syn.detach().cpu()
	# 	num_imgs = imgs.shape[0]
	# 	batch_size = 4  # 2x2 grid per batch
	# 	num_batches = (num_imgs + batch_size - 1) // batch_size
	# 	plot_cap_len = 30  # Max caption chars per line

	# 	# Find nearest neighboring captions using text_syn (same as line 1489)
	# 	syn_captions = [""] * num_imgs
	# 	if 'train_sentences' in locals() and 'train_caption_embed' in locals():
	# 		try:
	# 			syn_captions = nearest_neighbor(train_sentences, text_syn.detach().cpu(), train_caption_embed)
	# 		except Exception as e:
	# 			print(f"[yellow]Warning: Could not find nearest captions: {e}[/yellow]")
	# 			if 'train_sentences' in locals():
	# 				syn_captions = train_sentences[:num_imgs] if len(train_sentences) >= num_imgs else [""] * num_imgs
	# 	elif 'train_sentences' in locals():
	# 		syn_captions = train_sentences[:num_imgs] if len(train_sentences) >= num_imgs else [""] * num_imgs

	# 	# Denormalize images for visualization
	# 	# imgs is [N, 3, H, W] in CLIP-normalized space
	# 	imgs_denorm = denormalize_clip(imgs)  # [N, 3, H, W] -> [N, 3, H, W] in [0, 1] range
	# 	imgs_denorm = torch.clamp(imgs_denorm, 0.0, 1.0)
	# 	imgs_denorm = imgs_denorm.permute(0, 2, 3, 1).numpy()  # [N, H, W, C]

	# 	# Process in batches of 4
	# 	for batch_idx in range(num_batches):
	# 		start_idx = batch_idx * batch_size
	# 		end_idx = min(start_idx + batch_size, num_imgs)
	# 		batch_imgs = imgs_denorm[start_idx:end_idx]
	# 		batch_captions = syn_captions[start_idx:end_idx]
	# 		batch_num = len(batch_imgs)

	# 		# Create 2x2 grid for this batch
	# 		fig, axs = plt.subplots(2, 2, figsize=(8, 8))
	# 		axs = axs.flatten()

	# 		for i in range(4):  # Always 4 subplots
	# 			ax = axs[i]
	# 			if i < batch_num:
	# 				# Show image
	# 				ax.imshow(batch_imgs[i])
	# 				ax.axis('off')
					
	# 				# Caption: word wrap if too long
	# 				cap = batch_captions[i] if batch_captions[i] is not None else ""
	# 				if cap:
	# 					wrapped_cap = textwrap.fill(cap, width=plot_cap_len)
	# 					ax.set_title(wrapped_cap, fontsize=9, pad=5)
	# 				else:
	# 					ax.set_title("", fontsize=9)
	# 			else:
	# 				# Empty subplot
	# 				ax.axis('off')
	# 				ax.set_title("", fontsize=9)

	# 		fig.suptitle(f"Initial Synthetic Images & Associated Captions (Batch {batch_idx+1}/{num_batches})", fontsize=12)
	# 		plt.tight_layout(rect=[0, 0, 1, 0.96])
			
	# 		# Save batch with zero-padded batch number
	# 		os.makedirs(os.path.join(log_dir, f"initial_0"), exist_ok=True)
	# 		img_fig_path = os.path.join(log_dir, f"initial_0", f"0_init_image_syn_captions_grid_batch{batch_idx+1:03d}_{num_batches:03d}_{make_timestamp()}.png")
	# 		plt.savefig(img_fig_path, dpi=150, bbox_inches='tight')
	# 		plt.close(fig)
	# 		print(f"[yellow]Saved batch {batch_idx+1}/{num_batches} ({batch_num} images) to {img_fig_path}[/yellow]")

	# 	print(f"[yellow]Saved all {num_imgs} synthetic images in {num_batches} batches of 4[/yellow]")

	# except Exception as e:
	# 	print(f"[red]Error visualizing initial image syn with captions: {e}[/red]")
	# 	import traceback
	# 	traceback.print_exc()  
	# print(f'saved initial synthetic images and captions to {log_dir}')

	image_syn = image_syn.to(args.device).detach().requires_grad_(True)
	text_syn  = text_syn.to(args.device).detach().requires_grad_(True)

	##########################################################
	# optimizer
	##########################################################
	params = [
		{"params": [image_syn], "lr": args.lr_img},
		{"params": [text_syn],  "lr": args.lr_txt},
	]
	if args.optimizer == 'sgd':
		opt = torch.optim.SGD(params, lr=0.0, momentum=args.momentum)
	else:
		opt = torch.optim.Adam(params, lr=0.0, betas=(0.5, 0.999))
	grad_clip = args.grad_clip

	##########################################################
	# training loop
	##########################################################
	train_iter = iter(train_loader)
	last_refresh = 0
	
	best_aggr_results = defaultdict(list)
	best_aggr_results['r_mean'] = [-1000]

	model_init_times = []
	distillation_times = []
	distillation_before_backward_times = []

	print(f'[yellow]Starting distillation loop[/yellow]')
	stopping_eval_iterations_after_best = 20
	no_improvement_count = 0  # Track consecutive iterations without improvement
	for it in tqdm(range(args.Iteration), desc='Distillation', ncols=100, total=args.Iteration):
		student_net = None
		teachers = None
		a0 = None
		batch = None
		
		save_this_it = False

		##########################################################
		# Evaluate synthetic data
		##########################################################
		if it > 0 and (it > 0 and (it % args.eval_it == 0 or it == args.Iteration-1)):
			# print('Evaluation\nimage_model_train = %s, text_model_train = %s, iteration = %d'%(args.image_encoder, args.text_encoder, it))
			
			# print('DSA augmentation strategy: \n', args.dsa_strategy)
			# print('DSA augmentation parameters: \n', args.dsa_param.__dict__)
			
			multi_eval_aggr_result = defaultdict(list)  # aggregated results of multiple evaluations
			best_updated_this_eval = False  # Track if best_aggr_results was updated
			
			for it_eval in range(args.num_eval):
				args.image_pretrained = True
				torch.random.manual_seed(int(time.time() * 1000) % 100000)
				eval_net = CLIPModel_full(args, temperature=args.temperature)
	
				eval_net.image_encoder.model.to(args.device)
				eval_net.text_encoder.model.to(args.device)
				eval_net.text_projection.to(args.device)
								
				with torch.no_grad():
					image_save = image_syn
					text_save = text_syn

				image_syn_eval, text_syn_eval = copy.deepcopy(image_save.detach()), copy.deepcopy(text_save.detach())
				
				eval_net, acc_train_list, best_val_result = evaluate_synset(
					it_eval, eval_net, image_syn_eval, text_syn_eval, test_loader, args, bert_test_embed, similarity_train=None
				)
				
				for k, v in best_val_result.items():
					multi_eval_aggr_result[k].append(v)

				print('best val results at current iteration: ', best_val_result)
				current_eval_result = defaultdict(list)
				for k, v in best_val_result.items():
					current_eval_result[k].append(v)
				results_table = print_results(current_eval_result, title=f'[{it+1}/{args.Iteration}] eval #[{it_eval+1}/{args.num_eval}] results for {args.dataset}')
				logger.info(results_table)
				print(results_table)

			
			##########################################################
			# logging summary
			##########################################################
			for key, values in multi_eval_aggr_result.items():
				writer.add_scalar(f"Eval/Mean/{key}", float(np.mean(values)), it)
				writer.add_scalar(f"Eval/Std/{key}",  float(np.std(values)), it)

				wandb.log({
					f"Eval/Mean/{key}": float(np.mean(values)),
					f"Eval/Std/{key}": float(np.std(values)),
				}, step=it)
				# print(f"[Eval it={it}] {key}={np.mean(values):.2f} ± {np.std(values):.2f}")
			
			results_table = print_results(multi_eval_aggr_result, title=f'image-text retrieval results (avg {args.num_eval} evals) for {args.dataset} at iteration {it}')
			logger.info(results_table)
			print(results_table)
			print(f'saved logs to {log_dir}')
	
			# update best results for final print
			if np.mean(best_aggr_results['r_mean']) < np.mean(multi_eval_aggr_result['r_mean']):
				best_aggr_results = copy.deepcopy(multi_eval_aggr_result)
				best_updated_this_eval = True  # Mark that best was updated
				no_improvement_count = 0  # Reset counter on improvement
				for key, values in best_aggr_results.items():
					writer.add_scalar(f"BestEval/Mean/{key}", float(np.mean(values)), it)
					writer.add_scalar(f"BestEval/Std/{key}",  float(np.std(values)), it)

					wandb.log({"BestEval/Mean/{}".format(key): float(np.mean(values)), 
							"BestEval/Std/{}".format(key): float(np.std(values))}, step=it)
				
				torch.save({
					'image': image_syn.detach().cpu(),
					'text': text_syn.detach().cpu(),
					'iter': it,
				}, os.path.join(log_dir, f"distilled_pairs_best.pt"))
				logger.info(f"Best results at iteration {it} saved to {log_dir}/distilled_pairs_best.pt")
				
				results_table = print_results(best_aggr_results, title=f'<best> image-text retrieval results (avg {args.num_eval} evals) for {args.dataset} at iteration {it}')
				logger.info(results_table)
	
				# === Wall-clock stamping & new-best printf ===
				current_metric = float(np.mean(best_aggr_results[args.wall_clock_tracker.metric_name]))
				wc = args.wall_clock_tracker.stamp_eval(it=it, value=current_metric)

				# log current & elapsed
				writer.add_scalar(f"WallClock/current_{args.wall_clock_tracker.metric_name}", current_metric, it)
				writer.add_scalar("WallClock/elapsed_s", wc["elapsed_s"], it)
				wandb.log({
					f"WallClock/current_{args.wall_clock_tracker.metric_name}": current_metric,
					"WallClock/elapsed_s": wc["elapsed_s"],
				}, step=it)

				# on new best: print and log the time-to-best
				if wc["new_best"]:
					msg = (f"[BEST] it=[yellow]{it}[/yellow] | {args.wall_clock_tracker.metric_name}=[yellow]{wc['best_val']:.4f}[/yellow] | "
						f"time_to_best=[yellow]{wc['best_time_s']:.2f}[/yellow] seconds")
					print(msg)
					# try:
					# 	tqdm.write(msg)   # plays nice with progress bars
					# except Exception:
					# 	print(msg)
					writer.add_scalar("WallClock/time_to_best_s", wc["best_time_s"], it)
					wandb.log({
						"WallClock/time_to_best_s": wc["best_time_s"],
						f"WallClock/best_{args.wall_clock_tracker.metric_name}": wc["best_val"],
					}, step=it)

				# optional: when first crossing a fixed-quality target, record once
				if args.wall_clock_tracker.target_value is not None and args.wall_clock_tracker.target_time_s is not None:
					writer.add_scalar("WallClock/time_to_target_s", args.wall_clock_tracker.target_time_s, it)
					wandb.log({"WallClock/time_to_target_s": args.wall_clock_tracker.target_time_s}, step=it)
	 
	 
	 
				# print(f'saving best snapshot image and text syn to {log_dir}')
				
				# # Save snapshot plot of synthetic images and show the nearest neighboring text captions
				# # Save in batches of 4 (2x2 grid) for all images
				# try:
				# 	import matplotlib.pyplot as plt
				# 	
				# 	import textwrap

				# 	# Get images
				# 	imgs = image_syn.detach().cpu()
				# 	num_imgs = imgs.shape[0]
				# 	batch_size = 4  # 2x2 grid per batch
				# 	num_batches = (num_imgs + batch_size - 1) // batch_size
				# 	plot_cap_len = 30  # Max caption chars per line

				# 	# Find nearest neighboring captions using text_syn (same as line 1489)
				# 	syn_captions = [""] * num_imgs
				# 	if 'train_sentences' in locals() and 'train_caption_embed' in locals():
				# 		try:
				# 			syn_captions = nearest_neighbor(train_sentences, text_syn.detach().cpu(), train_caption_embed)
				# 		except Exception as e:
				# 			print(f"[yellow]Warning: Could not find nearest captions: {e}[/yellow]")
				# 			if 'train_sentences' in locals():
				# 				syn_captions = train_sentences[:num_imgs] if len(train_sentences) >= num_imgs else [""] * num_imgs
				# 	elif 'train_sentences' in locals():
				# 		syn_captions = train_sentences[:num_imgs] if len(train_sentences) >= num_imgs else [""] * num_imgs

				# 	# Denormalize images for visualization
				# 	# imgs is [N, 3, H, W] in CLIP-normalized space
				# 	imgs_denorm = denormalize_clip(imgs)  # [N, 3, H, W] -> [N, 3, H, W] in [0, 1] range
				# 	imgs_denorm = torch.clamp(imgs_denorm, 0.0, 1.0)
				# 	imgs_denorm = imgs_denorm.permute(0, 2, 3, 1).numpy()  # [N, H, W, C]

				# 	# Process in batches of 4
				# 	for batch_idx in range(num_batches):
				# 		start_idx = batch_idx * batch_size
				# 		end_idx = min(start_idx + batch_size, num_imgs)
				# 		batch_imgs = imgs_denorm[start_idx:end_idx]
				# 		batch_captions = syn_captions[start_idx:end_idx]
				# 		batch_num = len(batch_imgs)

				# 		# Create 2x2 grid for this batch
				# 		fig, axs = plt.subplots(2, 2, figsize=(8, 8))
				# 		axs = axs.flatten()

				# 		for i in range(4):  # Always 4 subplots
				# 			ax = axs[i]
				# 			if i < batch_num:
				# 				# Show image
				# 				ax.imshow(batch_imgs[i])
				# 				ax.axis('off')
								
				# 				# Caption: word wrap if too long
				# 				cap = batch_captions[i] if batch_captions[i] is not None else ""
				# 				if cap:
				# 					wrapped_cap = textwrap.fill(cap, width=plot_cap_len)
				# 					ax.set_title(wrapped_cap, fontsize=9, pad=5)
				# 				else:
				# 					ax.set_title("", fontsize=9)
				# 			else:
				# 				# Empty subplot
				# 				ax.axis('off')
				# 				ax.set_title("", fontsize=9)

				# 		fig.suptitle(f"{it}th iteration Synthetic Images & Associated Captions (Batch {batch_idx+1}/{num_batches})", fontsize=12)
				# 		plt.tight_layout(rect=[0, 0, 1, 0.96])
						
				# 		# Save batch with zero-padded batch number
				# 		os.makedirs(os.path.join(log_dir, f"{it}"), exist_ok=True)
				# 		img_fig_path = os.path.join(log_dir, f"{it}/{it}_image_syn_captions_grid_batch{batch_idx+1:03d}_{num_batches:03d}_{make_timestamp()}.png")
				# 		plt.savefig(img_fig_path, dpi=150, bbox_inches='tight')
				# 		plt.close(fig)
				# 		print(f"[yellow]Saved batch {batch_idx+1}/{num_batches} ({batch_num} images) to {img_fig_path}[/yellow]")

				# 	print(f"[yellow]Saved all {num_imgs} synthetic images in {num_batches} batches of 4 at {it}th iteration[/yellow]")

				# except Exception as e:
				# 	print(f"[red]Error visualizing {it}th iteration image syn with captions: {e}[/red]")
				# 	import traceback
				# 	traceback.print_exc()
				# print(f'saved {it}th iteration best snapshot image and text syn to {log_dir}')
			
			else:
				# No improvement this evaluation
				no_improvement_count += 1

			##########################################################
			# save
			##########################################################
			if save_this_it:			
				with torch.no_grad():
					vis = image_syn.detach().cpu()
					img_save_dir = os.path.join(log_dir, 'images')
					os.make_dirs(img_save_dir, exist_ok=True)
					save_path = os.path.join(img_save_dir, f"synthetic_images_{it}.png")

					save_image(torch.clamp(denormalize_clip(vis)[:min(64, vis.size(0))], 0, 1), save_path, nrow=8)
					# TensorBoard: 이미지 그리드
					grid = to_grid_for_tb(vis[:min(64, vis.size(0))], nrow=8)
					writer.add_image("Synthetic/ImagesGrid", grid, it)
					wandb.log({"Synthetic/ImagesGrid": wandb.Image(save_path)}, step=it)
										# 히스토그램(픽셀/텍스트)
					writer.add_histogram("Synthetic/Pixels", vis.flatten(), it)
					writer.add_histogram("Synthetic/TextValues", text_syn.detach().cpu().flatten(), it)
					wandb.log({
						"Synthetic/Pixels": wandb.Histogram(vis.flatten().numpy()),
						"Synthetic/TextValues": wandb.Histogram(text_syn.detach().cpu().flatten().numpy(), step=it),
					}, step=it)

					# (선택) 최근접 캡션 16개만 텍스트로 로깅
					try:
						sentence_list = nearest_neighbor(train_sentences, text_syn.detach().cpu(), train_caption_embed)
						sentence_list = sentence_list[:16]
						writer.add_text("Synthetic/NearestCaptions", "<br>".join(sentence_list), it)
						columns = len(sentence_list)
						wandb.log({"Synthetic/NearestCaptions": wandb.Table(data=sentence_list, columns=columns)}, step=it)
					except Exception as e:
						print(f"Error in nearest neighbor: {e}")
						pass
					
					data_save_dir = os.path.join(log_dir, 'synthetic_data')
					os.make_dirs(data_save_dir, exist_ok=True)
					torch.save({
						'image': image_syn.detach().cpu(),
						'text': text_syn.detach().cpu(),
						'iter': it,
					}, os.path.join(data_save_dir, f"distilled_pairs_{it}.pt"))
			
			
			# Early stopping: if no improvement for 10 consecutive evaluation iterations
			if no_improvement_count >= stopping_eval_iterations_after_best:
				print(f"[Early Stopping] No improvement in r_mean for {no_improvement_count} consecutive evaluation iterations. Stopping training.")
				logger.info(f"[Early Stopping] No improvement in r_mean for {no_improvement_count} consecutive evaluation iterations. Stopping training at iteration {it}.")
				
				# Save final state as in line 1235
				torch.save({
					'image': image_syn.detach().cpu(),
					'text': text_syn.detach().cpu(),
					'iter': it,
				}, os.path.join(log_dir, f"distilled_pairs_early_stop_iter_{it}.pt"))
				logger.info(f"Early stop results at iteration {it} saved to {log_dir}/distilled_pairs_early_stop_iter_{it}.pt")
				
				results_table = print_results(best_aggr_results, title=f'<best> image-text retrieval results (avg {args.num_eval} evals) for {args.dataset} at iteration {it}')
				logger.info(results_table)
				print(results_table)
				# Break out of the training loop
				break
		
	

		##########################################################
		# initialize distillation model
		##########################################################
		clean_cache()
		start_model_init.record()
		
		if args.init_model_method == 'default':
			if it == 0 or (it - last_refresh) >= args.teacher_resample:
				# Periodically refresh teachers
				student_net, alphas = make_teachers(args, args.device, n_teachers=1, alpha_from=0.2, alpha_to=0.5)
				student_net = student_net[0]
				student_net = student_net.eval()
				a0 = alphas[0]
				last_refresh = it
	
		elif args.init_model_method == 'mixed':	
			# print(f'[yellow]Initializing distillation model using mixed method[/yellow]')
			student_net = CLIPModel_full(args, temperature=args.temperature)
			student_net.eval()
			orig_img_model = copy.deepcopy(student_net.image_encoder.state_dict())
			orig_txt_model = copy.deepcopy(student_net.text_projection.state_dict())
   
			merged_img_model, merged_txt_model = make_distillation_model(args, student_net, merge_image=True, merge_text=True)
			student_net.image_encoder.load_state_dict(copy.deepcopy(merged_img_model))
			student_net.text_projection.load_state_dict(copy.deepcopy(merged_txt_model))
   
			assert _weights_differ(student_net.image_encoder.state_dict(), orig_img_model), "student_net.image_encoder.state_dict() weights are still equal to orig_img_model"
			assert _weights_differ(student_net.text_projection.state_dict(), orig_txt_model), "student_net.text_projection.state_dict() weights are still equal to orig_txt_model"
			assert _states_differ(student_net.image_encoder.state_dict(), orig_img_model), "student_net.image_encoder.state_dict() is still equal to orig_img_model"
			assert _states_differ(student_net.text_projection.state_dict(), orig_txt_model), "student_net.text_projection.state_dict() is still equal to orig_txt_model"
   
			a0 = 0
			last_refresh = 0
   
		elif args.init_model_method == 'naive':
			student_net = CLIPModel_full(args, temperature=args.temperature)
			student_net.eval()
			orig_img_model = copy.deepcopy(student_net.image_encoder.state_dict())
			orig_txt_model = copy.deepcopy(student_net.text_projection.state_dict())
   
			
			# random expert
			EXPERT_NUM = random.randint(0, args.num_buffers-1)
			EPOCH_NUM = random.randint(args.min_start_epoch, args.max_start_epoch)
			img_file = os.path.join(args.buffer_path, F'img_replay_buffer_{EXPERT_NUM}_{EPOCH_NUM}.pth')
			txt_file = os.path.join(args.buffer_path, F'txt_replay_buffer_{EXPERT_NUM}_{EPOCH_NUM}.pth')
			img_expert = torch.load(img_file)
			txt_expert = torch.load(txt_file)
			
			
			# naive mix
			min_ratio, max_ratio = args.naive_mix_min_ratio, args.naive_mix_max_ratio
			ratio = random.uniform(min_ratio, max_ratio)
			for key in img_expert.keys():
				img_expert[key] = img_expert[key].clone().to('cpu') * ratio + orig_img_model[key].clone().to('cpu') * (1.0 - ratio)
			for key in txt_expert.keys():
				txt_expert[key] = txt_expert[key].clone().to('cpu') * ratio + orig_txt_model[key].clone().to('cpu') * (1.0 - ratio)

			# update student_net
			student_net.image_encoder.load_state_dict(copy.deepcopy(img_expert))
			student_net.text_projection.load_state_dict(copy.deepcopy(txt_expert))
		
			a0 = 0
			last_refresh = 0
   
		elif args.init_model_method == 'none':
			student_net = CLIPModel_full(args, temperature=args.temperature)
			student_net = student_net.to(args.device).eval()
			a0 = 0
			last_refresh = 0
   
		elif args.init_model_method == 'expert':
			student_net = CLIPModel_full(args, temperature=args.temperature)
			student_net = student_net.to(args.device).eval()
			EXPERT_NUM = random.randint(0, args.num_buffers-1)
			EPOCH_NUM = random.randint(args.min_start_epoch, args.max_start_epoch)
			img_file = os.path.join(args.buffer_path, F'img_replay_buffer_{EXPERT_NUM}_{EPOCH_NUM}.pth')
			txt_file = os.path.join(args.buffer_path, F'txt_replay_buffer_{EXPERT_NUM}_{EPOCH_NUM}.pth')
			img_expert = torch.load(img_file)
			txt_expert = torch.load(txt_file)
			
			# update student_net
			student_net.image_encoder.load_state_dict(copy.deepcopy(img_expert))
			student_net.text_projection.load_state_dict(copy.deepcopy(txt_expert))

			a0 = 0
			last_refresh = 0

		student_net = student_net.to(args.device)
		student_net.train()
  
		# student_net.eval()
		# for param in student_net.parameters():
		# 	param.requires_grad = False
  
  
		end_model_init.record()
		torch.cuda.synchronize()
		print(f"Time taken to prepare distillation model: {start_model_init.elapsed_time(end_model_init)/1000:.2f} seconds")
		model_init_times.append(start_model_init.elapsed_time(end_model_init)/1000)
  
  
		##########################################################
		# distillation
		##########################################################
		torch.cuda.empty_cache()
		gc.collect()
  
  
  		####################################################################################################
		# --- Synthetic Minibatch ---
		####################################################################################################
		start_distillation.record()
		start_distillation_before_backward.record()
		try:
			batch = next(train_iter)
		except StopIteration:
			train_iter = iter(train_loader)
			batch = next(train_iter)


		real_imgs, _, real_txt_inds = batch
		real_imgs = real_imgs.to(args.device)
		real_txt_embed = train_caption_embed[real_txt_inds].to(args.device).float()

		# Synthetic minibatch
		B_real = real_imgs.size(0)
		B_txt  = real_txt_embed.size(0)
  
		if args.num_queries > 100:
			B = min(B_real, B_txt, 64, args.num_queries)
		else:
			B = min(B_real, B_txt, args.num_queries)
		B = 64
		syn_idx = torch.randperm(args.num_queries)[:B]
		
		syn_imgs = image_syn[syn_idx]
		syn_txts = text_syn[syn_idx]
		# breakpoint()
		loss_total = torch.tensor(0.0, device=args.device)

		# fix
		# student_net.image_encoder = student_net.image_encoder.to(args.device)
		# student_net.text_projection = student_net.text_projection.to(args.device)
		# ---------------------------
		# 1) 실배치 feature로 중심 업데이트
		# ---------------------------
		with torch.no_grad():
			# student_net.eval()
			# student_net 기준 실 이미지 feature 추출 (image_projection 포함, L2 norm)
			feat_img_real = student_net.image_encoder(real_imgs)
			if hasattr(student_net, 'image_projection'):
				feat_img_real = student_net.image_projection(feat_img_real.float())
			feat_img_real = F.normalize(feat_img_real.float(), dim=1)

			feat_txt_real = student_net.text_projection(real_txt_embed.float())
			feat_txt_real = F.normalize(feat_txt_real.float(), dim=1)
		
		
		# student_net.train()

		# ---------------------------
		# 2) 합성 배치 feature & 손실
		# ---------------------------
		feat_img_syn, feat_txt_syn = get_clip_feats(student_net, syn_imgs, syn_txts, args)
		
		loss_dict = OrderedDict()
		loss_meter = AverageMeter()
		####################################################################################################
		# --- InfoNCE ---
		####################################################################################################
		if args.w_nce > 0.0:
			loss_infonce = args.w_nce * infonce(feat_img_syn, feat_txt_syn, temperature=args.temperature)
		else:
			loss_infonce = torch.tensor(0.0, device=args.device)
		loss_dict['InfoNCE'] = loss_infonce.item()


		# construct u, g vectors
		u_real = F.normalize(feat_img_real + feat_txt_real, dim=1)   # [Br, D]
		u_syn  = F.normalize(feat_img_syn + feat_txt_syn, dim=1)     # [B,  D]
		g_real = F.normalize(feat_img_real - feat_txt_real, dim=1)
		g_syn  = F.normalize(feat_img_syn - feat_txt_syn, dim=1)

  
		####################################################################################################
		# --- Geodesic MMD (u) ---
		####################################################################################################
		if args.w_sph_u_mmd > 0.0:
			# ablation
			# concat_real = torch.cat([feat_img_real, feat_txt_real], dim=1)
			# concat_syn = torch.cat([feat_img_syn, feat_txt_syn], dim=1)
			# u_real = concat_real
			# u_syn = concat_syn

			# ablation
			# u_real = feat_img_real
			# u_syn = feat_img_syn

			K_ss = spherical_rbf_kernel(u_syn,  u_syn,  sigma=args.sph_u_mmd_sigma)
			K_rr = spherical_rbf_kernel(u_real, u_real, sigma=args.sph_u_mmd_sigma)
			K_sr = spherical_rbf_kernel(u_syn,  u_real, sigma=args.sph_u_mmd_sigma)

			# add: diversity weight to encourage separation between synthetic samples
			mmd2 = args.diversity_weight_u * offdiag_mean(K_ss) + offdiag_mean(K_rr) - 2.0 * K_sr.mean()
						# record for logging
			with torch.no_grad():
				kss_u = offdiag_mean(K_ss).item()
				krr_u = offdiag_mean(K_rr).item()
				ksr_u = K_sr.mean().item()
				wandb.log({
					"GeodesicMMD_u/DiagU/offdiag_Kss": kss_u,
					"GeodesicMMD_u/DiagU/offdiag_Krr": krr_u,
					"GeodesicMMD_u/DiagU/mean_Ksr": ksr_u,
					"GeodesicMMD_u/DiagU/mmd2_u_raw": args.diversity_weight_u*kss_u + krr_u - 2.0*ksr_u
				}, step=it)
	
			if args.sqrtmmd:
				loss_sph_u_mmd = torch.sqrt(torch.clamp(mmd2, min=0.0) + EPS)
			elif args.logmmd:	
				loss_sph_u_mmd = torch.log(torch.clamp(mmd2, min=1.0) + 1.0)
			else:
				loss_sph_u_mmd = mmd2
	
			loss_sph_u_mmd = args.w_sph_u_mmd * loss_sph_u_mmd
		else:
			loss_sph_u_mmd = torch.tensor(0.0, device=args.device)
		loss_dict['GeodesicMMD_u'] = loss_sph_u_mmd.item()

		####################################################################################################
		# --- Geodesic MMD (g) ---
		####################################################################################################
		
		if args.w_sph_g_mmd > 0.0:
			# ablation
			# g_real = feat_txt_real
			# g_syn = feat_txt_syn
			
			K_ss_g = spherical_rbf_kernel(g_syn,  g_syn,  sigma=args.sph_g_mmd_sigma)
			K_rr_g = spherical_rbf_kernel(g_real, g_real, sigma=args.sph_g_mmd_sigma)
			K_sr_g = spherical_rbf_kernel(g_syn,  g_real, sigma=args.sph_g_mmd_sigma)

			# add: diversity weight to encourage separation between synthetic samples
			mmd2_gmmd = args.diversity_weight_g * offdiag_mean(K_ss_g) + offdiag_mean(K_rr_g) - 2.0 * K_sr_g.mean()
   
			with torch.no_grad():
				kss_g = offdiag_mean(K_ss_g).item()
				krr_g = offdiag_mean(K_rr_g).item()
				ksr_g = K_sr_g.mean().item()
				wandb.log({
					"GeodesicMMD_g/DiagG/offdiag_Kss": kss_g,
					"GeodesicMMD_g/DiagG/offdiag_Krr": krr_g,
					"GeodesicMMD_g/DiagG/mean_Ksr": ksr_g,
					"GeodesicMMD_g/DiagG/mmd2_g_raw": args.diversity_weight_g*kss_g + krr_g - 2.0*ksr_g
				}, step=it)

			if args.sqrtmmd:
				loss_sph_g_mmd = torch.sqrt(torch.clamp(mmd2_gmmd, min=0.0) + EPS)
			elif args.logmmd:	
				loss_sph_g_mmd = torch.log(torch.clamp(mmd2_gmmd, min=1.0) + 1.0)
			else:
				loss_sph_g_mmd = mmd2_gmmd

			loss_sph_g_mmd = args.w_sph_g_mmd * loss_sph_g_mmd
		else:
			loss_sph_g_mmd = torch.tensor(0.0, device=args.device)
		loss_dict['GeodesicMMD_g'] = loss_sph_g_mmd.item()
		####################################################################################################
		# --- Conditional Gap Matching / CG-MMD ---
		####################################################################################################
		# image modality conditional gap matching
		if args.w_cgap_x > 0.0:			
			# === (D) Conditional Gap Matching: g | x ===
			loss_cgap_mmd_x = conditional_kernel_mmd(
				g_real=g_real, g_syn=g_syn,
				key_real=feat_img_real, key_syn=feat_img_syn,
				sigma_g=getattr(args, "c_sigma_g", 0.5),
				sigma_key=getattr(args, "c_sigma_x", 0.5),
				weight_mode='normalize',
				stopgrad_W=True,
				diversity_weight=args.diversity_weight_g,
			)

			if args.sqrtmmd:
				loss_cgap_mmd_x = torch.sqrt(torch.clamp(loss_cgap_mmd_x, min=0.0) + EPS)
			elif args.logmmd:	
				loss_cgap_mmd_x = torch.log(torch.clamp(loss_cgap_mmd_x, min=1.0) + 1.0)
			else:
				loss_cgap_mmd_x = loss_cgap_mmd_x
	
			loss_cgap_mmd_x = args.w_cgap_x * loss_cgap_mmd_x
		else:
			loss_cgap_mmd_x = torch.tensor(0.0, device=args.device)
		loss_dict['ConditionalGapMMD_g_x'] = loss_cgap_mmd_x.item()
		# text modality conditional gap matching
		if args.w_cgap_y > 0.0:
			# === (C) Conditional Gap Matching: g | y ===
			loss_cgap_mmd_y = conditional_kernel_mmd(
				g_real=g_real, g_syn=g_syn,
				key_real=feat_txt_real, key_syn=feat_txt_syn,
				sigma_g=getattr(args, "c_sigma_g", 0.5),
				sigma_key=getattr(args, "c_sigma_x", 0.5),
				weight_mode='normalize',
				stopgrad_W=True,
				diversity_weight=args.diversity_weight_g,
			)

			if args.sqrtmmd:
				loss_cgap_mmd_y = torch.sqrt(torch.clamp(loss_cgap_mmd_y, min=0.0) + EPS)
			elif args.logmmd:	
				loss_cgap_mmd_y = torch.log(torch.clamp(loss_cgap_mmd_y, min=1.0) + 1.0)
			else:
				loss_cgap_mmd_y = loss_cgap_mmd_y
	
			loss_cgap_mmd_y = args.w_cgap_y * loss_cgap_mmd_y
		else:
			loss_cgap_mmd_y = torch.tensor(0.0, device=args.device)
		loss_dict['ConditionalGapMMD_g_y'] = loss_cgap_mmd_y.item()
   
		# orthogonal u,g vector manipulation
		if args.w_cgap_u > 0.0:
			# project g to tangent of u (i.e., orthogonal to u)
			g_real_proj = F.normalize(g_real - (g_real*u_real).sum(1,keepdim=True)*u_real, dim=1)
			g_syn_proj  = F.normalize(g_syn  - (g_syn *u_syn ).sum(1,keepdim=True)*u_syn , dim=1)

			loss_cgap_u_mmd = conditional_kernel_mmd(
				g_real=g_real_proj, g_syn=g_syn_proj,
				key_real=u_real, key_syn=u_syn,
				sigma_g=args.c_sigma_g, sigma_key=getattr(args,"c_sigma_u",0.5),
				weight_mode='normalize', stopgrad_W=True,
				diversity_weight=args.diversity_weight_ortho,
			)
			if args.sqrtmmd:
				loss_cgap_u_mmd = torch.sqrt(torch.clamp(loss_cgap_u_mmd,0)+1e-8)
			elif args.logmmd:
				loss_cgap_u_mmd = torch.log(torch.clamp(loss_cgap_u_mmd,0)+1e-8)
			else:
				loss_cgap_u_mmd = loss_cgap_u_mmd

			loss_cgap_u_mmd = args.w_cgap_u * loss_cgap_u_mmd
		else:
			loss_cgap_u_mmd = torch.tensor(0.0, device=args.device)
		loss_dict['ConditionalGapMMD_ortho_u_g'] = loss_cgap_u_mmd.item()
   
   
   
		# total loss
		loss_total = loss_total.to(args.device)
		loss_total += loss_infonce + loss_sph_u_mmd + loss_sph_g_mmd + loss_cgap_mmd_x + loss_cgap_mmd_y + loss_cgap_u_mmd
		loss_dict['total'] = loss_total.item()
   
		# if it > 0 and it % (args.eval_it*2) == 0:
		# 	with torch.no_grad():
		# 		# --- Kernel regression quality on Y ---
		# 		if args.w_cgap_y > 0.0:
		# 			W_y = spherical_rbf_weights(feat_txt_syn, feat_txt_real, sigma=getattr(args, "c_sigma_y", 0.5))  # [B,Br]
		# 			Neff_y = 1.0 / (W_y.pow(2).sum(dim=1) + 1e-8)                     # [B]
		# 			H_y = -(W_y.clamp_min(1e-12) * (W_y.clamp_min(1e-12)).log()).sum(dim=1)
		# 			g_real = F.normalize(feat_img_real - feat_txt_real, dim=1)
		# 			mu_g_y_vec = W_y @ g_real
		# 			mu_g_y = F.normalize(mu_g_y_vec, dim=1)
		# 			mu_norm_y = mu_g_y_vec.norm(dim=1)
		# 			g_syn = F.normalize(feat_img_syn - feat_txt_syn, dim=1)
		# 			ang_align_y = torch.acos((g_syn * mu_g_y).sum(dim=1).clamp(-1+1e-6, 1-1e-6))

		# 			writer.add_scalar("DiagY/Neff_mean",  Neff_y.mean().item(), it)
		# 			writer.add_scalar("DiagY/Neff_std",   Neff_y.std().item(),  it)
		# 			writer.add_scalar("DiagY/H_mean",     H_y.mean().item(),    it)
		# 			writer.add_scalar("DiagY/mu_norm_mean", mu_norm_y.mean().item(), it)
		# 			writer.add_scalar("DiagY/angle_g_mu_mean", ang_align_y.mean().item(), it)
		# 			writer.add_histogram("DiagY/Neff_hist",  Neff_y.detach().cpu(), it)
		# 			writer.add_histogram("DiagY/ang_hist",   ang_align_y.detach().cpu(), it)
		# 			wandb.log({
		# 			"DiagY/Neff_mean":  Neff_y.mean().item(),
		# 			"DiagY/Neff_std":   Neff_y.std().item(),
		# 			"DiagY/H_mean":     H_y.mean().item(),
		# 			"DiagY/mu_norm_mean": mu_norm_y.mean().item(),
		# 			"DiagY/angle_g_mu_mean": ang_align_y.mean().item(),
		# 			}, step=it)

		# 		# --- Angle histogram (real vs syn) ---
		# 		th_real = pair_angle(feat_img_real, feat_txt_real).detach().cpu().numpy()
		# 		th_syn  = pair_angle(feat_img_syn, feat_txt_syn).detach().cpu().numpy()
		# 		fig_th = fig_hist_two(th_real, th_syn, title=f"Theta(i,t) real vs syn @it={it}", bins=40)
		# 		writer.add_figure("Diag/AngleTheta_hist", fig_th, global_step=it)
		# 		plt.close(fig_th)
		# 		wandb.log({"Diag/AngleTheta_hist": wandb.Image(fig_th)}, step=it)

		# 		# --- Random 2D scatter of u and g ---
		# 		# (샘플링으로 가볍게)
		# 		max_show = min(512, feat_img_syn.size(0), feat_img_real.size(0))
		# 		idx_s = torch.randperm(feat_img_syn.size(0))[:max_show]
		# 		idx_r = torch.randperm(feat_img_real.size(0))[:max_show]

		# 		u_syn  = F.normalize(feat_img_syn + feat_txt_syn, dim=1)[idx_s]
		# 		u_real = F.normalize(feat_img_real + feat_txt_real, dim=1)[idx_r]
		# 		g_syn  = F.normalize(feat_img_syn - feat_txt_syn, dim=1)[idx_s]
		# 		g_real = F.normalize(feat_img_real - feat_txt_real, dim=1)[idx_r]

		# 		P_u_syn  = random_plane_project(u_syn)
		# 		P_u_real = random_plane_project(u_real)
		# 		P_g_syn  = random_plane_project(g_syn)
		# 		P_g_real = random_plane_project(g_real)

		# 		fig_u = fig_scatter_2d(P_u_real.detach().cpu().numpy(), P_u_syn.detach().cpu().numpy(), title=f"u random-2D real vs syn @it={it}")
		# 		writer.add_figure("Diag/u_scatter2D", fig_u, global_step=it)
		# 		plt.close(fig_u)
		# 		wandb.log({"Diag/u_scatter2D": wandb.Image(fig_u)}, step=it)
				
		# 		fig_g = fig_scatter_2d(P_g_real.detach().cpu().numpy(), P_g_syn.detach().cpu().numpy(), title=f"g random-2D real vs syn @it={it}")
		# 		writer.add_figure("Diag/g_scatter2D", fig_g, global_step=it)
		# 		plt.close(fig_g)
				
		# 		wandb.log({"Diag/g_scatter2D": wandb.Image(fig_g)}, step=it)
				
		# 		# (선택) Plane principal angles 히스토그램도 가능하나, 비용이 있어 기본은 비활성화.
		# 		# 필요 시 여기서 Q=two-ortho → 2x2 SVD → atan2로 주각 분포 로그.

		# ====================
		end_distillation_before_backward.record()
		torch.cuda.synchronize()
		print(f"Time taken to apply distill step before backward: {start_distillation_before_backward.elapsed_time(end_distillation_before_backward)/1000:.2f} seconds")
		distillation_before_backward_times.append(start_distillation_before_backward.elapsed_time(end_distillation_before_backward)/1000)
  
		opt.zero_grad()
		
		loss_total.backward()
		torch.nn.utils.clip_grad_norm_([image_syn, text_syn], grad_clip)
		opt.step()
  
		loss_meter.update(loss_total.item(), B)
  
		end_distillation.record()
		torch.cuda.synchronize()
		print(f"Time taken to apply distill step: {start_distillation.elapsed_time(end_distillation)/1000:.2f} seconds")
		distillation_times.append(start_distillation.elapsed_time(end_distillation)/1000)

		if it > 0 and it % args.log_freq == 0:
			print_str = f'Batch: [yellow]{it+1:06d}/{len(train_loader):06d}[/yellow] || Avg total loss: [yellow]{loss_meter.avg:.5f}[/yellow] || '
			for key, value in loss_dict.items():
				if value > 0.0:
					writer.add_scalar(f"Distillation/{key}", value, it)
					wandb.log({f'Distillation/{key}': value}, step=it)
					if key != 'total':
						print_str += f'\t[blue]{key}: {value:.5f}[/blue]'
			print(print_str)
			logger.info(print_str)
			print_memory_usage()
   
			
			print(f'---------------[yellow]{it}th iteration[/yellow]-----------------')
			print(f'time taken to get data: {np.mean(data_init_times):.2f} seconds')
			print(f'average time taken to initialize model: {np.mean(model_init_times):.2f} seconds')
			print(f'average time taken to distillation before backward: {np.mean(distillation_before_backward_times):.2f} seconds')
			print(f'average time taken to distillation: {np.mean(distillation_times):.2f} seconds')
			print(f'Batch size: {B}')
			print(f'Number of queries: {args.num_queries}')
			print(f'---------------[yellow]End of {it}th iteration[/yellow]-----------------')
		
			with torch.no_grad():
				writer.add_scalar("Synthetic/ImgParamNormMean", image_syn.detach().flatten(start_dim=1).norm(dim=1).mean().item(), it)
				writer.add_scalar("Synthetic/ImgParamNormStd", image_syn.detach().flatten(start_dim=1).norm(dim=1).std().item(), it)
				writer.add_scalar("Synthetic/TxtParamNormMean", text_syn.detach().norm(dim=1).mean().item(), it)
				writer.add_scalar("Synthetic/TxtParamNormStd", text_syn.detach().norm(dim=1).std().item(), it)
				
				wandb.log({
					"Synthetic/ImgParamNormMean": image_syn.detach().flatten(start_dim=1).norm(dim=1).mean().item(),
					"Synthetic/ImgParamNormStd": image_syn.detach().flatten(start_dim=1).norm(dim=1).std().item(),
					"Synthetic/TxtParamNormMean": text_syn.detach().norm(dim=1).mean().item(),
					"Synthetic/TxtParamNormStd": text_syn.detach().norm(dim=1).std().item(),
				}, step=it)

			# if args.save_snapshot_every_iter:
			# 	# Save snapshot plot of synthetic images and show the nearest neighboring text captions
			# 	# Save in batches of 4 (2x2 grid) for all images
			# 	try:
			# 		import matplotlib.pyplot as plt
					
			# 		import textwrap

			# 		# Get images
			# 		imgs = image_syn.detach().cpu()
			# 		num_imgs = imgs.shape[0]
			# 		batch_size = 4  # 2x2 grid per batch
			# 		num_batches = (num_imgs + batch_size - 1) // batch_size
			# 		plot_cap_len = 30  # Max caption chars per line

			# 		# Find nearest neighboring captions using text_syn (same as line 1489)
			# 		syn_captions = [""] * num_imgs
			# 		if 'train_sentences' in locals() and 'train_caption_embed' in locals():
			# 			try:
			# 				syn_captions = nearest_neighbor(train_sentences, text_syn.detach().cpu(), train_caption_embed)
			# 			except Exception as e:
			# 				print(f"[yellow]Warning: Could not find nearest captions: {e}[/yellow]")
			# 				if 'train_sentences' in locals():
			# 					syn_captions = train_sentences[:num_imgs] if len(train_sentences) >= num_imgs else [""] * num_imgs
			# 		elif 'train_sentences' in locals():
			# 			syn_captions = train_sentences[:num_imgs] if len(train_sentences) >= num_imgs else [""] * num_imgs

			# 		# Denormalize images for visualization
			# 		# imgs is [N, 3, H, W] in CLIP-normalized space
			# 		imgs_denorm = denormalize_clip(imgs)  # [N, 3, H, W] -> [N, 3, H, W] in [0, 1] range
			# 		imgs_denorm = torch.clamp(imgs_denorm, 0.0, 1.0)
			# 		imgs_denorm = imgs_denorm.permute(0, 2, 3, 1).numpy()  # [N, H, W, C]

			# 		# Process in batches of 4
			# 		for batch_idx in range(num_batches):
			# 			start_idx = batch_idx * batch_size
			# 			end_idx = min(start_idx + batch_size, num_imgs)
			# 			batch_imgs = imgs_denorm[start_idx:end_idx]
			# 			batch_captions = syn_captions[start_idx:end_idx]
			# 			batch_num = len(batch_imgs)

			# 			# Create 2x2 grid for this batch
			# 			fig, axs = plt.subplots(2, 2, figsize=(8, 8))
			# 			axs = axs.flatten()

			# 			for i in range(4):  # Always 4 subplots
			# 				ax = axs[i]
			# 				if i < batch_num:
			# 					# Show image
			# 					ax.imshow(batch_imgs[i])
			# 					ax.axis('off')
								
			# 					# Caption: word wrap if too long
			# 					cap = batch_captions[i] if batch_captions[i] is not None else ""
			# 					if cap:
			# 						wrapped_cap = textwrap.fill(cap, width=plot_cap_len)
			# 						ax.set_title(wrapped_cap, fontsize=9, pad=5)
			# 					else:
			# 						ax.set_title("", fontsize=9)
			# 				else:
			# 					# Empty subplot
			# 					ax.axis('off')
			# 					ax.set_title("", fontsize=9)

			# 			fig.suptitle(f"{it}th iteration Synthetic Images & Associated Captions (Batch {batch_idx+1}/{num_batches})", fontsize=12)
			# 			plt.tight_layout(rect=[0, 0, 1, 0.96])
						
			# 			# Save batch with zero-padded batch number
			# 			os.makedirs(os.path.join(log_dir, f"{it}"), exist_ok=True)
			# 			img_fig_path = os.path.join(log_dir, f"{it}/{it}_image_syn_captions_grid_batch{batch_idx+1:03d}_{num_batches:03d}_{make_timestamp()}.png")
			# 			plt.savefig(img_fig_path, dpi=150, bbox_inches='tight')
			# 			plt.close(fig)
			# 			print(f"[yellow]Saved batch {batch_idx+1}/{num_batches} ({batch_num} images) to {img_fig_path}[/yellow]")

			# 		print(f"[yellow]Saved all {num_imgs} synthetic images in {num_batches} batches of 4 at {it}th iteration[/yellow]")

			# 	except Exception as e:
			# 		print(f"[red]Error visualizing {it}th iteration image syn with captions: {e}[/red]")
			# 		import traceback
			# 		traceback.print_exc()
			# 	print(f'saved {it}th iteration best snapshot image and text syn to {log_dir}')
		

	if np.mean(best_aggr_results['r_mean']) != -1000:
		results_table = print_results(best_aggr_results, title=f'best image-text retrieval results for {args.dataset}')
		logger.info(results_table)


	print(f'time taken to get data: {np.mean(data_init_times):.2f} seconds')
	print(f'average time taken to initialize model: {np.mean(model_init_times):.2f} seconds')
	print(f'average time taken to distillation: {np.mean(distillation_times):.2f} seconds')
	
	print(f'saving distilled pairs to {os.path.join(log_dir, f"distilled_pairs_final_iter_{it}.pt")}')
	torch.save({
		'image': image_syn.detach().cpu(),
		'text': text_syn.detach().cpu(),
		'iter': it,
	}, os.path.join(log_dir, f"distilled_pairs_final_iter_{it}.pt"))
	json_path = args.wall_clock_tracker.finalize(log_dir)
	print(f"[WallClock] best {args.wall_clock_tracker.metric_name}={args.wall_clock_tracker.best_val:.4f} | "
		f"time_to_best={args.wall_clock_tracker.best_time_s:.2f}s | "
		f"time_to_target={args.wall_clock_tracker.target_time_s if args.wall_clock_tracker.target_time_s is not None else 'N/A'}s")
	print(f"[WallClock] history saved -> {json_path}")

	writer.close()
	wandb.finish()
	logger.info("Training completed successfully")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Image-Text Retrieval")

	# -----------------------------
	# Data & Paths
	# -----------------------------
	parser.add_argument('--dataset', type=str, default='flickr', choices=['flickr', 'coco', 'flickr8k'])
	parser.add_argument('--image_root', type=str, default='./data/datasets/Flickr30k/')
	parser.add_argument('--ann_root', type=str, default='./data/annotations/')
	parser.add_argument('--log_dir', type=str, default='./log_final')
	parser.add_argument('--wandb', action='store_true')
	parser.add_argument('--wandb_entity', type=str, default='test')
	parser.add_argument('--wandb_project', type=str, default='test')
	parser.add_argument('--wandb_mode', type=str, default='online')
	parser.add_argument('--save_snapshot_every_iter', type=bool, default=False)


	# -----------------------------
	# Dataset Setup
	# -----------------------------
	parser.add_argument('--no_aug', action='store_true', help='no_aug')
	parser.add_argument('--feat_dim', type=int, default=768)
	parser.add_argument('--image_size', type=int, default=224)

	# -----------------------------
	# Synthetic Data
	# -----------------------------
	parser.add_argument('--num_queries', type=int, default=100)
	parser.add_argument('--pix_init', type=str, default='real', choices=['real', 'noise'])
	parser.add_argument('--txt_init', type=str, default='real', choices=['real', 'noise'])
	parser.add_argument('--syn_init', type=str, default='kmeans', choices=['random', 'kmeans'])


	# -----------------------------
	# Model Initialization
	# -----------------------------
	parser.add_argument('--naive_mix_min_ratio', type=float, default=0.1, help='min ratio we can start at')
	parser.add_argument('--naive_mix_max_ratio', type=float, default=0.9, help='max ratio we can start at')
	# -----------------------------
	# Optimization & Training
	# -----------------------------
	parser.add_argument('--Iteration', type=int, default=3000)
	
	parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'])
	parser.add_argument('--lr_img', type=float, default=100.0)
	parser.add_argument('--lr_txt', type=float, default=100.0)
	parser.add_argument('--momentum', type=float, default=0.5)
	parser.add_argument('--grad_clip', type=float, default=1.0)
	parser.add_argument('--log_freq', type=int, default=5)
	parser.add_argument('--save_it', type=int, default=200)
	parser.add_argument('--seed', type=int, default=0)
	parser.add_argument('--name', type=str, default='')

	# -----------------------------
	# Augmentation
	# -----------------------------
	parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate', help='differentiable Siamese augmentation strategy')

	# -----------------------------
	# Model Architecture
	# -----------------------------
	parser.add_argument('--image_encoder', type=str, default='nfnet')
	parser.add_argument('--text_encoder', type=str, default='bert', choices=['bert', 'clip', 'distilbert'])
	parser.add_argument('--text_pretrained', type=bool, default=True)
	parser.add_argument('--image_pretrained', type=bool, default=True)
	parser.add_argument('--text_trainable', type=bool, default=False)
	parser.add_argument('--image_trainable', type=bool, default=True)
	parser.add_argument('--distill', type=bool, default=True)
	parser.add_argument('--only_has_image_projection', type=bool, default=False, help='None')
	parser.add_argument('--temperature', type=float, default=0.07)


	# -----------------------------
	# Teacher Buffer/Ensemble
	# -----------------------------
	parser.add_argument('--buffer_path', type=str, default=None, required=True)
	parser.add_argument('--num_buffers', type=int, default=20)
	parser.add_argument('--teacher_resample', type=int, default=50)

	# -----------------------------
	# Evaluation
	# -----------------------------
	parser.add_argument('--eval_it', type=int, default=100, help='evaluation interval')
	parser.add_argument('--epoch_eval_train', type=int, default=100, help='epochs to train on synthetic for eval')
	parser.add_argument('--eval_eval_freq', type=int, default=100, help='evaluation frequency during eval training')
	parser.add_argument('--num_eval', type=int, default=1, help='repeat eval training')
	parser.add_argument('--batch_size_train', type=int, default=512, help='batch_size_train')		# optimize real train batch
	parser.add_argument('--batch_size_test', type=int, default=512, help='batch_size_test')	# eval real test batch
	parser.add_argument('--lr_teacher_img', type=float, default=0.1, help='eval-time image LR')
	parser.add_argument('--lr_teacher_txt', type=float, default=0.1, help='eval-time text LR')
	parser.add_argument('--loss_type', type=str, default="InfoNCE", help='InfoNCE or WBCE')
	parser.add_argument('--text_embed_dir', type=str, default='text_embeds', help='text embed npz file directory')
	parser.add_argument('--min_start_epoch', type=int, default=1, help='max epoch we can start at')
	parser.add_argument('--max_start_epoch', type=int, default=3, help='max epoch we can start at')
 

	# -----------------------------
	# Visualization
	# -----------------------------
	parser.add_argument('--kmeans_viz', type=bool, default=False, help='Visualize k-means initialization.')

	# -----------------------------
	# Clustering for DM
	# -----------------------------
	parser.add_argument('--cluster_by', type=str, default='image_text', choices=['image', 'text', 'image_text'])
	parser.add_argument('--cluster_mode', type=str, default='cosine', choices=['cosine', 'euclidean'])

	# -----------------------------
	# Loss Weights & Adanced Losses
	# -----------------------------

	# NCE Loss
	parser.add_argument('--w_nce', type=float, default=0.1, help='weight for image-text NCE loss')

	# Spherical MMD
	parser.add_argument('--w_sph_u_mmd', type=float, default=1.0, help='Weight for spherical MMD loss (u).')
	parser.add_argument('--sph_u_mmd_sigma', type=float, default=0.5, help='Sigma of geodesic RBF kernel for spherical MMD (u).')
	parser.add_argument('--w_sph_g_mmd', type=float, default=1.0, help='Weight for spherical MMD loss (g).')
	parser.add_argument('--sph_g_mmd_sigma', type=float, default=0.5, help='Sigma of geodesic RBF kernel for spherical MMD (g).')
	parser.add_argument('--sqrtmmd', type=bool, default=False, help='Use square root of MMD loss.')
	parser.add_argument('--logmmd', type=bool, default=False, help='Use log of MMD loss.')

	# Cross Covariance
	parser.add_argument('--w_cross_cov', type=float, default=0.0, help='Weight for cross covariance matching loss.')

	# Cross Matching
	parser.add_argument('--w_cgap_u', type=float, default=0.0, help='Weight for conditional MMD for u.')
	parser.add_argument('--c_sigma_u', type=float, default=0.5, help='Kernel sigma for conditioning on u.')

	# Conditional Gap Matching
	parser.add_argument('--w_cgap_y', type=float, default=0.0, help='Weight for conditional GAP matching g|y.')     # 0.4
	parser.add_argument('--w_cgap_x', type=float, default=0.0, help='Weight for conditional GAP matching g|x.')     # 0.4
	parser.add_argument('--c_sigma_y', type=float, default=0.5, help='Kernel sigma for conditioning on Y.')
	parser.add_argument('--c_sigma_x', type=float, default=0.5, help='Kernel sigma for conditioning on X.')
	parser.add_argument('--c_sigma_g', type=float, default=0.5, help='Kernel sigma for GAP vector G.')

	# -----------------------------
	# Merging & Diversity (advanced model init/ensemble)
	# -----------------------------
	parser.add_argument('--init_model_method', type=str, default='default', choices=['default', 'mixed', 'naive', 'none', 'expert'])
	parser.add_argument('--merge_alpha', type=float, default=1.0, help='merging alpha for model merge')
	parser.add_argument('--diversity_weight_u', type=float, default=1, help='diversity weight for diversity loss')
	parser.add_argument('--diversity_weight_g', type=float, default=1, help='diversity weight for diversity loss')
	parser.add_argument('--diversity_weight_ortho', type=float, default=1, help='diversity weight for diversity loss')
	

	# -----------------------------
	# Wall Clock Tracker
	# -----------------------------
	parser.add_argument('--wall_clock_tracker', action='store_true', help='Enable wall clock tracker')
	parser.add_argument('--primary_metric', type=str, default='r_mean',
						help='Metric key to track at eval (e.g., r_mean, img_r1, txt_r1).')
	parser.add_argument('--target_value', type=float, default=None,
						help='Optional fixed-quality target for reporting time-to-target.')

	args = parser.parse_args()
		
	args.dsa_param = ParamDiffAug()
	args.dsa = False if args.dsa_strategy in ['none', 'None'] else True

	if args.wall_clock_tracker:
		args.wall_clock_tracker = WallClockTracker(metric_name=getattr(args, "primary_metric", "r_mean"))
		if getattr(args, "target_value", None) is not None:
			args.wall_clock_tracker.target_value = float(args.target_value)


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

