import torch
import random
import numpy as np
import os
import json
from torchvision.utils import make_grid
from src.model_utils import CLIPModel_full
from typing import List
import copy
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
# =====================================================
# Utils
# =====================================================

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

# =============================
# Image utils
# =============================


def _norm_t(it, T):
	if T is None or T <= 0:
		return 1.0
	t = it / float(T)
	return max(0.0, min(1.0, t))
# =====================================================
# Teachers (buffer mix)
# =====================================================

def _load_clip_from_buffers(args):
	import glob
	
	BASE_DIR = args.buffer_path
	# BASE_DIR = 'buffer_my_seed0123/flickr8k/nfnet_bert/InfoNCE'
	FILE_FORMAT = 'replay_buffer'
	# k = random.randint(0, args.num_buffers - 1)
	img_expert_files = glob.glob(os.path.join(BASE_DIR, f'img_{FILE_FORMAT}_*.pt')) # list of filenames
	txt_expert_files = glob.glob(os.path.join(BASE_DIR, f'txt_{FILE_FORMAT}_*.pt')) # list of filenames
	total_img_buffers = len(img_expert_files)-1
	EXPERT_NUM1 = random.randint(0, total_img_buffers)

	MAX_EPOCH = args.max_start_epoch
	EPOCH_NUM1 = random.choice(range(1, MAX_EPOCH+1))

	# img_path = os.path.join(args.buffer_path, f'img_replay_buffer_{k}_10.pth')
	# txt_path = os.path.join(args.buffer_path, f'txt_replay_buffer_{k}_10.pth')
	# img_sd = torch.load(img_path, map_location='cpu')
	# txt_sd = torch.load(txt_path, map_location='cpu')
	img_file = os.path.join(BASE_DIR, F'img_{FILE_FORMAT}_{EXPERT_NUM1}.pt')
	txt_file = os.path.join(BASE_DIR, F'txt_{FILE_FORMAT}_{EXPERT_NUM1}.pt')
	img_expert_1 = torch.load(img_file, map_location='cpu')[0][EPOCH_NUM1]
	txt_expert_1 = torch.load(txt_file, map_location='cpu')[0][EPOCH_NUM1]
	
	return img_expert_1, txt_expert_1, EXPERT_NUM1


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
		
	rand_model = CLIPModel_full(args, temperature=args.temperature)
	rand_img_sd = copy.deepcopy(rand_model.image_encoder.state_dict())
	rand_txt_sd = copy.deepcopy(rand_model.text_projection.state_dict())
	
	pretrain_img_sd, pretrain_txt_sd, chosen_expert_idx = _load_clip_from_buffers(args)
	
	mixed = CLIPModel_full(args, temperature=args.temperature).to(device)
	_param_interpolate_(mixed.image_encoder, pretrain_img_sd, rand_img_sd, alpha)
	_param_interpolate_(mixed.text_projection, pretrain_txt_sd, rand_txt_sd, alpha)

	mixed.eval()
	for p in mixed.parameters():
		p.requires_grad_(False)
	return mixed, alpha, chosen_expert_idx


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
		W = W.detach()

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

# unbiased estimator (네 코드의 offdiag_mean 규칙과 동일)
def offdiag_mean(K):
	n = K.size(0)
	if K.size(0) == K.size(1) and n > 1:
		return (K.sum() - K.diag().sum()) / (n * (n - 1))
	else:
		return K.mean()

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

# ==============================