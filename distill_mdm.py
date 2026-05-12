import argparse
import copy
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from data import get_dataloaders
from src.clustering_utils import KMeansCluster
from src.networks import CLIPModel_full
from src.utils import ParamDiffAug
from src.vl_distill_utils import (
	get_images_texts, load_or_process_file, textprocess_test, textprocess_train
)

EPS = 1e-12

def compute_cosine_dict(state1, state2, ref, device='cuda'):
	cos_dict = {}
	s1 = state1
	s2 = state2
	r = ref
	for k in r.keys():
		if k not in s1 or k not in s2:
			continue
		v1, v2, vr = s1[k], s2[k], r[k]
		if (not torch.is_tensor(v1)) or (not torch.is_tensor(v2)) or (not torch.is_tensor(vr)):
			continue
		if (not v1.is_floating_point()) or (not v2.is_floating_point()) or (not vr.is_floating_point()):
			continue
		if v1.shape != v2.shape or v1.shape != vr.shape:
			continue
		a = (v1 - vr).clone().detach()
		b = (v2 - vr).clone().detach()
		dot = torch.dot(a.reshape(-1), b.reshape(-1))
		na = torch.linalg.vector_norm(a)
		nb = torch.linalg.vector_norm(b)
		cos = (dot / (na * nb + EPS)).clamp(-1.0, 1.0)
		cos_dict[k] = cos.detach().to('cpu')
	return cos_dict

def compute_ratio_from_cos(cos_dict, k=2.0):
	out = {}
	for k_ in cos_dict.keys():
		c = float(cos_dict[k_])
		out[k_] = (k * c) / (((k - 1) * c) + 1.0 + EPS)
	return out

@torch.inference_mode()
def fast_merge(w1, w2, w0, ratio, device='cuda', dtype=torch.float32, non_blocking=True, alpha=1.0):
	w_merge = {}
	for k in w0.keys():
		if (k in w1) and (k in w2) \
			and torch.is_tensor(w1[k]) and torch.is_tensor(w2[k]) and torch.is_tensor(w0[k]) \
			and w1[k].is_floating_point() and w2[k].is_floating_point() and w0[k].is_floating_point() \
			and (w1[k].shape == w2[k].shape == w0[k].shape):
			t1 = w1[k].to(device=device, dtype=dtype, non_blocking=non_blocking)
			t2 = w2[k].to(device=device, dtype=dtype, non_blocking=non_blocking)
			student_net = w0[k].to(device=device, dtype=dtype, non_blocking=non_blocking)
			w12 = 0.5 * (t1 + t2)
			outk = w12
			if k in ratio:
				r = torch.as_tensor(ratio[k], dtype=dtype, device=device)
				outk = student_net + (r * (w12 - student_net) * alpha)
			w_merge[k] = outk
		else:
			w_merge[k] = w0[k]
	return w_merge

def load_model_state_dict(state_dict, map_location='cpu'):
	state_dict = torch.load(state_dict, map_location=map_location)
	return state_dict

def make_distillation_model(args, student_net, merge_image=True, merge_text=True, verbose=False):
	BASE_DIR = args.buffer_path
	total_img_buffers = args.num_buffers - 1
	assert total_img_buffers > 0, "total_img_buffers is 0"
	MAX_EPOCH = args.max_start_epoch
	MIN_EPOCH = 1
	EXPERT_NUM1 = random.randint(0, total_img_buffers)
	EPOCH_NUM1 = random.choice(range(MIN_EPOCH, MAX_EPOCH+1))
	img_file = os.path.join(BASE_DIR, f'img_replay_buffer_{EXPERT_NUM1}_{EPOCH_NUM1}.pth')
	txt_file = os.path.join(BASE_DIR, f'txt_replay_buffer_{EXPERT_NUM1}_{EPOCH_NUM1}.pth')
	img_expert_1 = load_model_state_dict(img_file, map_location=args.device)
	txt_expert_1 = load_model_state_dict(txt_file, map_location=args.device)
	EXPERT_NUM2 = random.randint(0, total_img_buffers)
	epoch_pool = [i for i in range(MIN_EPOCH, MAX_EPOCH+1) if i != EPOCH_NUM1] if EXPERT_NUM1 == EXPERT_NUM2 else [i for i in range(MIN_EPOCH, MAX_EPOCH+1)]
	EPOCH_NUM2 = random.choice(epoch_pool)
	img_file = os.path.join(BASE_DIR, f'img_replay_buffer_{EXPERT_NUM2}_{EPOCH_NUM2}.pth')
	txt_file = os.path.join(BASE_DIR, f'txt_replay_buffer_{EXPERT_NUM2}_{EPOCH_NUM2}.pth')
	img_expert_2 = load_model_state_dict(img_file, map_location=args.device)
	txt_expert_2 = load_model_state_dict(txt_file, map_location=args.device)
	assert isinstance(img_expert_1, dict) and isinstance(txt_expert_1, dict) and isinstance(img_expert_2, dict) and isinstance(txt_expert_2, dict)
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

def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.cuda.empty_cache()
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	os.environ['PYTHONHASHSEED'] = str(seed)

def clean_cache():
	import gc
	gc.collect()
	torch.cuda.empty_cache()

def offdiag_mean(K):
	n = K.size(0)
	if K.size(0) == K.size(1) and n > 1:
		return (K.sum() - K.diag().sum()) / (n*(n-1))
	else:
		return K.mean()
		

def geodesic_distance(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
	a = F.normalize(a, dim=1)
	b = F.normalize(b, dim=1)
	cosv = (a @ b.t()) if a.dim()==2 and b.dim()==2 else (a*b).sum(dim=-1, keepdim=False)
	cosv = cosv.clamp(-1.0 + eps, 1.0 - eps)
	return torch.acos(cosv)  # [N,N] or [N] or scalar-like

def spherical_rbf_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float = 0.5, method='cosine') -> torch.Tensor:
	if method == 'geodesic':
		theta = geodesic_distance(x, y)   # [Nx, Ny]
		return torch.exp(- (theta * theta) / (2.0 * sigma * sigma))
	raise NotImplementedError("Only 'geodesic' method implemented in this kernel.")

def main(args):
	args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	set_seed(args.seed)
	clean_cache()

	# Standardize log_dir, but remove timestamp for reproducibility and publishing
	log_dir = f"{args.log_dir}/{args.dataset}/{args.image_encoder}/{args.text_encoder}/{args.loss_type}/{args.num_queries}/{args.init_model_method}/{args.name}"
	os.makedirs(log_dir, exist_ok=True, mode=0o777)
	args.log_dir = log_dir

	# Data loading
	train_loader, test_loader, train_dataset, test_dataset = get_dataloaders(args)
	train_sentences = train_dataset.get_all_captions()
	data = load_or_process_file('test_text', textprocess_test, args, test_loader)
	train_caption = load_or_process_file('train_text', textprocess_train, args, train_sentences)
	bert_test_embed = torch.from_numpy(data['bert_test_embed']).cpu()
	train_caption_embed = torch.from_numpy(train_caption['bert_train_embed']).cpu()
	image_syn, text_syn = get_images_texts(args.num_queries, train_dataset, args, init='random')


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
		
		def _load_clip_from_buffers(args):
			k = random.randint(0, args.num_buffers - 1)
			img_path = os.path.join(args.buffer_path, f'img_replay_buffer_{k}_10.pth')
			txt_path = os.path.join(args.buffer_path, f'txt_replay_buffer_{k}_10.pth')
			img_sd = torch.load(img_path, map_location=args.device)
			txt_sd = torch.load(txt_path, map_location=args.device)
		
		
			return img_sd, txt_sd


	def _param_interpolate_(tgt_model, sd_a, sd_b, alpha):
		with torch.no_grad():
			msd = tgt_model.state_dict()
			for k in msd.keys():
				if k in sd_a and k in sd_b and torch.is_tensor(sd_a[k]) and torch.is_tensor(sd_b[k]):
					if msd[k].shape == sd_a[k].shape == sd_b[k].shape and msd[k].dtype == sd_a[k].dtype == sd_b[k].dtype:
						msd[k].copy_(alpha * sd_a[k] + (1.0 - alpha) * sd_b[k])
			tgt_model.load_state_dict(msd)

		pretrain_img_sd, pretrain_txt_sd, chosen_k = _load_clip_from_buffers(args)
	
		mixed = CLIPModel_full(args, temperature=args.temperature).to(device)
		_param_interpolate_(mixed.image_encoder, pretrain_img_sd, rand_img_sd, alpha)
		_param_interpolate_(mixed.text_projection, pretrain_txt_sd, rand_txt_sd, alpha)

		mixed.eval()
		for p in mixed.parameters():
			p.requires_grad_(False)
		# return mixed, alpha, EXPERT_NUM1
		return mixed, alpha, chosen_k

	student_net_kmeans = None
	if args.syn_init == 'kmeans':
		if args.init_model_method == 'default':
			model_for_cluster, alphas_for_cluster = make_teachers(args, args.device, n_teachers=1, alpha_from=0.5, alpha_to=0.5)
			model_for_cluster = model_for_cluster[0]
			model_for_cluster.eval()
			student_net_kmeans = model_for_cluster
		elif args.init_model_method == 'mixed':
			student_net = CLIPModel_full(args, temperature=args.temperature)
			student_net.eval()
			merged_img_model, merged_txt_model = make_distillation_model(args, student_net, merge_image=True, merge_text=True)
			student_net.image_encoder.load_state_dict(copy.deepcopy(merged_img_model))
			student_net.text_projection.load_state_dict(copy.deepcopy(merged_txt_model))
			student_net_kmeans = copy.deepcopy(student_net)
		elif args.init_model_method == 'naive':
			student_net = CLIPModel_full(args, temperature=args.temperature)
			student_net.eval()
			orig_img_model = copy.deepcopy(student_net.image_encoder.state_dict())
			orig_txt_model = copy.deepcopy(student_net.text_projection.state_dict())
			EXPERT_NUM = random.randint(0, args.num_buffers-1)
			EPOCH_NUM = random.randint(args.min_start_epoch, args.max_start_epoch)
			img_file = os.path.join(args.buffer_path, F'img_replay_buffer_{EXPERT_NUM}_{EPOCH_NUM}.pth')
			txt_file = os.path.join(args.buffer_path, F'txt_replay_buffer_{EXPERT_NUM}_{EPOCH_NUM}.pth')
			img_expert = torch.load(img_file)
			txt_expert = torch.load(txt_file)
			min_ratio, max_ratio = args.naive_mix_min_ratio, args.naive_mix_max_ratio
			ratio = random.uniform(min_ratio, max_ratio)
			for key in img_expert.keys():
				img_expert[key] = img_expert[key].clone().to('cpu') * ratio + orig_img_model[key].clone().to('cpu') * (1.0 - ratio)
			for key in txt_expert.keys():
				txt_expert[key] = txt_expert[key].clone().to('cpu') * ratio + orig_txt_model[key].clone().to('cpu') * (1.0 - ratio)
			student_net.image_encoder.load_state_dict(copy.deepcopy(img_expert))
			student_net.text_projection.load_state_dict(copy.deepcopy(txt_expert))
			student_net_kmeans = student_net
		elif args.init_model_method == 'none':
			student_net_kmeans = CLIPModel_full(args, temperature=args.temperature)
			student_net_kmeans = student_net_kmeans.to(args.device).eval()
		elif args.init_model_method == 'expert':
			student_net = CLIPModel_full(args, temperature=args.temperature)
			student_net = student_net.to(args.device).eval()
			EXPERT_NUM = random.randint(0, args.num_buffers-1)
			EPOCH_NUM = random.randint(args.min_start_epoch, args.max_start_epoch)
			img_file = os.path.join(args.buffer_path, F'img_replay_buffer_{EXPERT_NUM}_{EPOCH_NUM}.pth')
			txt_file = os.path.join(args.buffer_path, F'txt_replay_buffer_{EXPERT_NUM}_{EPOCH_NUM}.pth')
			img_expert = torch.load(img_file)
			txt_expert = torch.load(txt_file)
			student_net_kmeans.image_encoder.load_state_dict(copy.deepcopy(img_expert))
			student_net_kmeans.text_projection.load_state_dict(copy.deepcopy(txt_expert))
			student_net_kmeans = student_net
		assert student_net_kmeans is not None, "student_net_kmeans is None"
		student_net_kmeans = student_net_kmeans.to(args.device)
		student_net_kmeans.eval()
		for p in student_net_kmeans.parameters():
			p.requires_grad_(False)
		# Feature extraction for clustering
		all_feats = []
		all_dataset_indices = []
		kmeans_train_loader = torch.utils.data.DataLoader(
			train_dataset,
			batch_size=512,
			shuffle=False,
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
					all_feats.append(F.normalize(torch.cat([feat_img, feat_txt], dim=1), dim=1).cpu())
				all_dataset_indices.append(torch.as_tensor(real_ds_inds, dtype=torch.long))
		all_feats = torch.cat(all_feats, dim=0)
		all_dataset_indices = torch.cat(all_dataset_indices, 0)
		with KMeansCluster(all_feats.to(args.device), mode=args.cluster_mode) as kmeans_unit:
			query_feat_indices = kmeans_unit.query_incluster_n(args.num_queries).cpu()
			selected_ds_indices = all_dataset_indices[query_feat_indices]
			image_syn = torch.stack([kmeans_train_loader.dataset[i][0] for i in selected_ds_indices.tolist()]).to(args.device)
			text_syn = train_caption_embed[selected_ds_indices].to(args.device).float()
		del kmeans_train_loader
	elif args.syn_init == 'noise':
		mean = torch.tensor([-0.0626, -0.0221,  0.0680])
		std  = torch.tensor([1.0451, 1.0752, 1.0539])
		image_syn = torch.randn([args.num_queries, 3, 224, 224])
		for c in range(3):
			image_syn[:, c] = image_syn[:, c] * std[c] + mean[c]
		text_syn = torch.normal(mean=-0.0094, std=0.5253, size=(args.num_queries, 768))

	image_syn = image_syn.to(args.device).detach().requires_grad_(True)
	text_syn  = text_syn.to(args.device).detach().requires_grad_(True)
	params = [
		{"params": [image_syn], "lr": args.lr_img},
		{"params": [text_syn],  "lr": args.lr_txt},
	]
	if args.optimizer == 'sgd':
		opt = torch.optim.SGD(params, lr=0.0, momentum=args.momentum)
	else:
		opt = torch.optim.Adam(params, lr=0.0, betas=(0.5, 0.999))
	grad_clip = args.grad_clip
	train_iter = iter(train_loader)
	last_refresh = 0


	def make_teachers(args, device, n_teachers: int, alpha_from=None, alpha_to=None):
		teachers = []
		alphas = []
		for _ in range(n_teachers):
			t, alpha, _ = _make_mixed_teacher(args, device, alpha_from=alpha_from, alpha_to=alpha_to)
			teachers.append(t)
			alphas.append(alpha)
		return teachers, alphas


	for it in tqdm(range(args.Iteration), desc='Distillation', ncols=100, total=args.Iteration):
		if args.init_model_method == 'default':
			if it == 0 or (it - last_refresh) >= args.teacher_resample:
				student_net, alphas = make_teachers(args, args.device, n_teachers=1, alpha_from=0.2, alpha_to=0.5)
				student_net = student_net[0]
				student_net = student_net.eval()
				last_refresh = it
		elif args.init_model_method == 'mixed':
			student_net = CLIPModel_full(args, temperature=args.temperature)
			student_net.eval()
			merged_img_model, merged_txt_model = make_distillation_model(args, student_net, merge_image=True, merge_text=True)
			student_net.image_encoder.load_state_dict(copy.deepcopy(merged_img_model))
			student_net.text_projection.load_state_dict(copy.deepcopy(merged_txt_model))
		elif args.init_model_method == 'naive':
			student_net = CLIPModel_full(args, temperature=args.temperature)
			student_net.eval()
			orig_img_model = copy.deepcopy(student_net.image_encoder.state_dict())
			orig_txt_model = copy.deepcopy(student_net.text_projection.state_dict())
			EXPERT_NUM = random.randint(0, args.num_buffers-1)
			EPOCH_NUM = random.randint(args.min_start_epoch, args.max_start_epoch)
			img_file = os.path.join(args.buffer_path, F'img_replay_buffer_{EXPERT_NUM}_{EPOCH_NUM}.pth')
			txt_file = os.path.join(args.buffer_path, F'txt_replay_buffer_{EXPERT_NUM}_{EPOCH_NUM}.pth')
			img_expert = torch.load(img_file)
			txt_expert = torch.load(txt_file)
			min_ratio, max_ratio = args.naive_mix_min_ratio, args.naive_mix_max_ratio
			ratio = random.uniform(min_ratio, max_ratio)
			for key in img_expert.keys():
				img_expert[key] = img_expert[key].clone().to('cpu') * ratio + orig_img_model[key].clone().to('cpu') * (1.0 - ratio)
			for key in txt_expert.keys():
				txt_expert[key] = txt_expert[key].clone().to('cpu') * ratio + orig_txt_model[key].clone().to('cpu') * (1.0 - ratio)
			student_net.image_encoder.load_state_dict(copy.deepcopy(img_expert))
			student_net.text_projection.load_state_dict(copy.deepcopy(txt_expert))
		elif args.init_model_method == 'none':
			student_net = CLIPModel_full(args, temperature=args.temperature)
			student_net = student_net.to(args.device).eval()
		elif args.init_model_method == 'expert':
			student_net = CLIPModel_full(args, temperature=args.temperature)
			student_net = student_net.to(args.device).eval()
			EXPERT_NUM = random.randint(0, args.num_buffers-1)
			EPOCH_NUM = random.randint(args.min_start_epoch, args.max_start_epoch)
			img_file = os.path.join(args.buffer_path, F'img_replay_buffer_{EXPERT_NUM}_{EPOCH_NUM}.pth')
			txt_file = os.path.join(args.buffer_path, F'txt_replay_buffer_{EXPERT_NUM}_{EPOCH_NUM}.pth')
			img_expert = torch.load(img_file)
			txt_expert = torch.load(txt_file)
			student_net.image_encoder.load_state_dict(copy.deepcopy(img_expert))
			student_net.text_projection.load_state_dict(copy.deepcopy(txt_expert))

		student_net = student_net.to(args.device)
		student_net.train()
		try:
			batch = next(train_iter)
		except StopIteration:
			train_iter = iter(train_loader)
			batch = next(train_iter)
		real_imgs, _, real_txt_inds = batch
		real_imgs = real_imgs.to(args.device)
		real_txt_embed = train_caption_embed[real_txt_inds].to(args.device).float()
		B_real = real_imgs.size(0)
		B_txt  = real_txt_embed.size(0)
		B = min(B_real, B_txt, 64, args.num_queries)
		syn_idx = torch.randperm(args.num_queries)[:B]
		syn_imgs = image_syn[syn_idx]
		syn_txts = text_syn[syn_idx]
		loss_total = torch.tensor(0.0, device=args.device)
		with torch.no_grad():
			feat_img_real = student_net.image_encoder(real_imgs)
			if hasattr(student_net, 'image_projection'):
				feat_img_real = student_net.image_projection(feat_img_real.float())
			feat_img_real = F.normalize(feat_img_real.float(), dim=1)
			feat_txt_real = student_net.text_projection(real_txt_embed.float())
			feat_txt_real = F.normalize(feat_txt_real.float(), dim=1)
		
		def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
			return x / (x.norm(dim=dim, keepdim=True) + eps)

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

		feat_img_syn, feat_txt_syn = get_clip_feats(student_net, syn_imgs, syn_txts, args)
		if args.w_nce > 0.0:
			def infonce(img_feats: torch.Tensor, txt_feats: torch.Tensor, temperature: float = 0.07):
				logits = (img_feats @ txt_feats.t()) / temperature
				targets = torch.arange(img_feats.size(0), device=img_feats.device)
				loss_i2t = F.cross_entropy(logits, targets)
				loss_t2i = F.cross_entropy(logits.t(), targets)
				return 0.5 * (loss_i2t + loss_t2i)
			loss_infonce = args.w_nce * infonce(feat_img_syn, feat_txt_syn, temperature=args.temperature)
		else:
			loss_infonce = torch.tensor(0.0, device=args.device)
		u_real = F.normalize(feat_img_real + feat_txt_real, dim=1)
		u_syn = F.normalize(feat_img_syn + feat_txt_syn, dim=1)
		g_real = F.normalize(feat_img_real - feat_txt_real, dim=1)
		g_syn = F.normalize(feat_img_syn - feat_txt_syn, dim=1)
		if args.w_sph_u_mmd > 0.0:
			K_ss = spherical_rbf_kernel(u_syn,  u_syn,  sigma=args.sph_u_mmd_sigma)
			K_rr = spherical_rbf_kernel(u_real, u_real, sigma=args.sph_u_mmd_sigma)
			K_sr = spherical_rbf_kernel(u_syn,  u_real, sigma=args.sph_u_mmd_sigma)
			mmd2 = args.diversity_weight_u * offdiag_mean(K_ss) + offdiag_mean(K_rr) - 2.0 * K_sr.mean()
			if args.sqrtmmd:
				loss_sph_u_mmd = torch.sqrt(torch.clamp(mmd2, min=0.0) + EPS)
			elif args.logmmd:
				loss_sph_u_mmd = torch.log(torch.clamp(mmd2, min=1.0) + 1.0)
			else:
				loss_sph_u_mmd = mmd2
			loss_sph_u_mmd = args.w_sph_u_mmd * loss_sph_u_mmd
		else:
			loss_sph_u_mmd = torch.tensor(0.0, device=args.device)
		if args.w_sph_g_mmd > 0.0:
			K_ss_g = spherical_rbf_kernel(g_syn,  g_syn,  sigma=args.sph_g_mmd_sigma)
			K_rr_g = spherical_rbf_kernel(g_real, g_real, sigma=args.sph_g_mmd_sigma)
			K_sr_g = spherical_rbf_kernel(g_syn,  g_real, sigma=args.sph_g_mmd_sigma)
			mmd2_gmmd = args.diversity_weight_g * offdiag_mean(K_ss_g) + offdiag_mean(K_rr_g) - 2.0 * K_sr_g.mean()
			if args.sqrtmmd:
				loss_sph_g_mmd = torch.sqrt(torch.clamp(mmd2_gmmd, min=0.0) + EPS)
			elif args.logmmd:
				loss_sph_g_mmd = torch.log(torch.clamp(mmd2_gmmd, min=1.0) + 1.0)
			else:
				loss_sph_g_mmd = mmd2_gmmd
			loss_sph_g_mmd = args.w_sph_g_mmd * loss_sph_g_mmd
		else:
			loss_sph_g_mmd = torch.tensor(0.0, device=args.device)
		loss_total = loss_total.to(args.device)
		loss_total += loss_infonce + loss_sph_u_mmd + loss_sph_g_mmd

		opt.zero_grad()
		loss_total.backward()
		torch.nn.utils.clip_grad_norm_([image_syn, text_syn], grad_clip)
		opt.step()

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Image-Text Retrieval")
	parser.add_argument('--dataset', type=str, default='flickr', choices=['flickr', 'coco', 'flickr8k'])
	parser.add_argument('--image_root', type=str, default='./data/datasets/Flickr30k/')
	parser.add_argument('--ann_root', type=str, default='./data/annotations/')
	parser.add_argument('--log_dir', type=str, default='./log_final')
	parser.add_argument('--save_snapshot_every_iter', type=bool, default=False)
	parser.add_argument('--no_aug', action='store_true', help='no_aug')
	parser.add_argument('--feat_dim', type=int, default=768)
	parser.add_argument('--image_size', type=int, default=224)
	parser.add_argument('--num_queries', type=int, default=100)
	parser.add_argument('--pix_init', type=str, default='real', choices=['real', 'noise'])
	parser.add_argument('--txt_init', type=str, default='real', choices=['real', 'noise'])
	parser.add_argument('--syn_init', type=str, default='kmeans', choices=['random', 'kmeans'])
	parser.add_argument('--naive_mix_min_ratio', type=float, default=0.1, help='min ratio we can start at')
	parser.add_argument('--naive_mix_max_ratio', type=float, default=0.9, help='max ratio we can start at')
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
	parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate', help='differentiable Siamese augmentation strategy')
	parser.add_argument('--image_encoder', type=str, default='nfnet')
	parser.add_argument('--text_encoder', type=str, default='bert', choices=['bert', 'clip', 'distilbert'])
	parser.add_argument('--text_pretrained', type=bool, default=True)
	parser.add_argument('--image_pretrained', type=bool, default=True)
	parser.add_argument('--text_trainable', type=bool, default=False)
	parser.add_argument('--image_trainable', type=bool, default=True)
	parser.add_argument('--distill', type=bool, default=True)
	parser.add_argument('--only_has_image_projection', type=bool, default=False, help='None')
	parser.add_argument('--temperature', type=float, default=0.07)
	parser.add_argument('--buffer_path', type=str, default=None, required=True)
	parser.add_argument('--num_buffers', type=int, default=20)
	parser.add_argument('--teacher_resample', type=int, default=50)
	parser.add_argument('--eval_it', type=int, default=100, help='evaluation interval')
	parser.add_argument('--epoch_eval_train', type=int, default=100, help='epochs to train on synthetic for eval')
	parser.add_argument('--eval_eval_freq', type=int, default=100, help='evaluation frequency during eval training')
	parser.add_argument('--num_eval', type=int, default=1, help='repeat eval training')
	parser.add_argument('--batch_size_train', type=int, default=512, help='batch_size_train')
	parser.add_argument('--batch_size_test', type=int, default=512, help='batch_size_test')
	parser.add_argument('--lr_teacher_img', type=float, default=0.1, help='eval-time image LR')
	parser.add_argument('--lr_teacher_txt', type=float, default=0.1, help='eval-time text LR')
	parser.add_argument('--loss_type', type=str, default="InfoNCE", help='InfoNCE or WBCE')
	parser.add_argument('--text_embed_dir', type=str, default='text_embeds', help='text embed npz file directory')
	parser.add_argument('--min_start_epoch', type=int, default=1, help='max epoch we can start at')
	parser.add_argument('--max_start_epoch', type=int, default=3, help='max epoch we can start at')
	parser.add_argument('--kmeans_viz', type=bool, default=False, help='Visualize k-means initialization.')
	parser.add_argument('--cluster_by', type=str, default='image_text', choices=['image', 'text', 'image_text'])
	parser.add_argument('--cluster_mode', type=str, default='cosine', choices=['cosine', 'euclidean'])
	parser.add_argument('--w_nce', type=float, default=0.1, help='weight for image-text NCE loss')
	parser.add_argument('--w_sph_u_mmd', type=float, default=1.0, help='Weight for spherical MMD loss (u).')
	parser.add_argument('--sph_u_mmd_sigma', type=float, default=0.5, help='Sigma of geodesic RBF kernel for spherical MMD (u).')
	parser.add_argument('--w_sph_g_mmd', type=float, default=1.0, help='Weight for spherical MMD loss (g).')
	parser.add_argument('--sph_g_mmd_sigma', type=float, default=0.5, help='Sigma of geodesic RBF kernel for spherical MMD (g).')
	parser.add_argument('--sqrtmmd', type=bool, default=False, help='Use square root of MMD loss.')
	parser.add_argument('--logmmd', type=bool, default=False, help='Use log of MMD loss.')
	parser.add_argument('--w_cross_cov', type=float, default=0.0, help='Weight for cross covariance matching loss.')
	parser.add_argument('--init_model_method', type=str, default='default', choices=['default', 'mixed', 'naive', 'none', 'expert'])
	parser.add_argument('--merge_alpha', type=float, default=1.0, help='merging alpha for model merge')
	parser.add_argument('--diversity_weight_u', type=float, default=1)
	parser.add_argument('--diversity_weight_g', type=float, default=1)
	parser.add_argument('--wall_clock_tracker', action='store_true', help='Enable wall clock tracker')
	parser.add_argument('--primary_metric', type=str, default='r_mean',
						help='Metric key to track at eval (e.g., r_mean, img_r1, txt_r1).')
	parser.add_argument('--target_value', type=float, default=None,
						help='Optional fixed-quality target for reporting time-to-target.')

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

	main(args)
