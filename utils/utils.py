import datetime
import torch
import json
import gc
import os
import lightning as L
from lightning.fabric import seed_everything
import random
from src.networks import CLIPModel_full
import copy
from typing import List
import torch.nn.functional as F
from torchvision.utils import make_grid
from lightning.fabric import Fabric
from rich import print


__all__ = ['launch_fabric', 'make_timestamp', 'to_jsonable', 'set_seed', 'clean_cache', 'make_log_dir', 'l2_normalize', 'save_script_copy', 'naive_mixing', 'denormalize_clip', 'to_grid_for_tb', 'nearest_neighbor', '_load_clip_from_buffers', '_param_interpolate_', '_make_mixed_teacher', '_make_teachers', 'clip_symmetric_nce_loss', 'get_clip_feats', 'geodesic_distance', 'geodesic_loss_pair', 'spherical_rbf_kernel']

def launch_fabric(precision='32-true', strategy='ddp', devices='auto', accelerator='auto'):
	fabric = Fabric(precision=precision, strategy=strategy, devices=devices, accelerator=accelerator)
	if not fabric._launched: 
		fabric.launch()
	return fabric

def make_timestamp(prefix: str="", suffix: str="") -> str:
	KST_TIMEZONE = 9
	tmstamp = datetime.datetime.now() + datetime.timedelta(hours=KST_TIMEZONE)
	tmstamp = '{:%m_%d_%Y_%H%M%S}'.format(tmstamp)
	
	return prefix + tmstamp + suffix

def to_jsonable(v):
	import json
	try:
		json.dumps(v, indent=4, sort_keys=True)
		return v
	except TypeError:
		return {k: to_jsonable(x) for k, x in vars(v).items()} if hasattr(v, "__dict__") else str(v)

def set_seed(seed):	
	seed_everything(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	# torch.use_deterministic_algorithms(True)
	os.environ['PYTHONHASHSEED'] = str(seed)

def clean_cache():
	import gc
	gc.collect()
	torch.cuda.empty_cache()

def make_log_dir(filename, log_dir, args):
	log_dir = f"{log_dir}/{args.dataset}/{args.image_encoder}/{args.text_encoder}/{args.loss_type}/{args.num_queries}/{filename}/{make_timestamp()}"
	os.makedirs(log_dir, exist_ok=True, mode=0o777)
	return log_dir

def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
	return x / (x.norm(dim=dim, keepdim=True) + eps)

def save_script_copy(src_file, dst_file):
	# Keep a copy of the current script file at log_dir
	try:
		import shutil, os
		if not os.path.exists(dst_file):
			shutil.copyfile(src_file, dst_file)
	except Exception as e:
		print(f"Warning: Failed to copy script file at {dst_file}): {e}")


def naive_mixing(w1, w0, ratio):
	"""
 	Args:
		w1 (dict): pretrained model weights
		w0 (dict): random model weights
		ratio (float): mixing ratio
	"""
	
	# alpha*pretrained + (1-alpha) *random
	w_merge = {}
	for key in w1.keys():
		w_merge[key] = w1[key].clone().to('cpu') * ratio + w0[key].clone().to('cpu') * (1.0 - ratio)
	
	return w_merge


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

#################################################################
# fetch neighboring text captions for the synthetic text embeddings
#################################################################
def nearest_neighbor(sentences, query_embeddings, database_embeddings):
	"""Find the nearest neighbors for a batch of embeddings.

	Args:
	sentences: The original sentences from which the embeddings were computed.
	query_embeddings: A batch of embeddings for which to find the nearest neighbors.
	database_embeddings: All pre-computed embeddings.

	Returns:
	A list of the most similar sentences for each embedding in the batch.
	"""
	import numpy as np
	from sklearn.metrics.pairwise import cosine_similarity

	nearest_neighbors = []
	for query in query_embeddings:
		similarities = cosine_similarity(query.reshape(1, -1), database_embeddings)
		
		most_similar_index = np.argmax(similarities)
		
		nearest_neighbors.append(sentences[most_similar_index])
		
	return nearest_neighbors


###########
# load buffer
###########

# =====================================================
# Teachers (buffer mix)
# =====================================================

def _load_clip_from_buffers(args):
	k = random.randint(0, args.num_buffers - 1)
	img_path = os.path.join(args.buffer_root, f'img_replay_buffer_{k}_10.pth')
	txt_path = os.path.join(args.buffer_root, f'txt_replay_buffer_{k}_10.pth')
	img_sd = torch.load(img_path, map_location='cpu')
	txt_sd = torch.load(txt_path, map_location='cpu')
	
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
		
	rand_model = CLIPModel_full(args, temperature=args.temperature) # pretrained model
	rand_img_sd = copy.deepcopy(rand_model.image_encoder.state_dict())
	rand_txt_sd = copy.deepcopy(rand_model.text_projection.state_dict())
	
	pretrain_img_sd, pretrain_txt_sd, chosen_k = _load_clip_from_buffers(args) # real trained model
	
	mixed = CLIPModel_full(args, temperature=args.temperature).to(device) # pretrained model
	
	# mixed = alpha * buffer + (1.0 - alpha) * random init
	_param_interpolate_(mixed.image_encoder, pretrain_img_sd, rand_img_sd, alpha) 
	_param_interpolate_(mixed.text_projection, pretrain_txt_sd, rand_txt_sd, alpha)

	mixed.eval()
	for p in mixed.parameters():
		p.requires_grad_(False)
	return mixed, alpha, chosen_k


def _make_teachers(args, device, n_teachers: int, alpha_from=None, alpha_to=None) -> List[CLIPModel_full]:
	teachers = []
	alphas = []
	for _ in range(n_teachers):
		# t, _, _ = _make_mixed_teacher(args, device, alpha=1.0)
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
