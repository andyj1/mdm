'''
 * part of the code (i.e. def epoch_test() and itm_eval()) is from: https://github.com/salesforce/BLIP/blob/main/train_retrieval.py#L69
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
import datetime
import time
from rich import print
from math import ceil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import wandb

from src.utils import *
from data import SimilarityDataloader, SimilarityDataLoaderWrapper

from src.utils import DiffAugment, augment

def epoch(e, dataloader, net, optimizer, args, aug=True):
	"""
	Perform a training epoch on the given dataloader.

	Args:
		dataloader (torch.utils.data.DataLoader): The dataloader for iterating over the dataset.
		net: The model.
		optimizer_img: The optimizer for image parameters.
		optimizer_txt: The optimizer for text parameters.
		args (object): The arguments specifying the training configuration.

	Returns:
		Tuple of average loss and average accuracy.
	"""
	net = net.to(args.device)
	net.train()
	loss_avg, acc_avg, num_exp = 0, 0, 0

	require_similarity = isinstance(dataloader, (SimilarityDataloader, SimilarityDataLoaderWrapper))    # Expert training: False

	for i, data in (enumerate(dataloader)):
		if args.distill:
			if require_similarity:
				image, caption, similarity = data
			else:
				image, caption = data[:2]
		else:
			if require_similarity:
				image, caption, index, similarity = data
			else:
				image, caption, index = data[:3]

		image = image.to(args.device)
		n_b = image.shape[0]

		if aug:
			if args.dsa:
				image = DiffAugment(image, args.dsa_strategy, param=args.dsa_param)
			else:
				image = augment(image, args.dc_aug_param, device=args.device)
				
		if require_similarity:
			loss, acc = net(image, caption, e, similarity)
		else:
			loss, acc = net(image, caption, e)

		loss_avg += loss.item() * n_b
		acc_avg += acc
		num_exp += n_b

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	loss_avg /= num_exp
	acc_avg /= num_exp

	return loss_avg, acc_avg

@torch.no_grad()
def epoch_test(dataloader, model, device, bert_test_embed):
	model.eval() 
	logit_scale = model.logit_scale.detach()
	start_time = time.time()  


	txt_embed = model.text_projection(bert_test_embed.float().to('cuda')) 
	text_embeds = txt_embed / txt_embed.norm(dim=1, keepdim=True) #torch.Size([5000, 768])
	text_embeds = text_embeds.to(device)

	image_embeds = []
	for image, img_id in dataloader: 
		image_feat = model.image_encoder(image.to(device))
		im_embed = image_feat / image_feat.norm(dim=1, keepdim=True)
		image_embeds.append(im_embed)
	image_embeds = torch.cat(image_embeds,dim=0)
	use_image_projection = False
	if use_image_projection:
		im_embed = model.image_projection(image_embeds.float())
		image_embeds = im_embed / im_embed.norm(dim=1, keepdim=True)
	else:
		image_embeds = image_embeds / image_embeds.norm(dim=1, keepdim=True)
		
	sims_matrix = logit_scale.exp() * image_embeds @ text_embeds.t()
	score_matrix_i2t = torch.full((len(image_embeds),len(text_embeds)),-100.0).to(device) #torch.Size([1000, 5000])
	for i, sims in enumerate(sims_matrix[0:sims_matrix.size(0) + 1]): 
		topk = 128 if sims.shape[0] > 128 else sims.shape[0]
		topk_sim, topk_idx = sims.topk(k=topk, dim=0)
		score_matrix_i2t[i,topk_idx] = topk_sim #i:0-999, topk_idx:0-4999, find top k (k=128) similar text for each image
	
	sims_matrix = sims_matrix.t()
	score_matrix_t2i = torch.full((len(text_embeds),len(image_embeds)),-100.0).to(device)
	for i,sims in enumerate(sims_matrix[0:sims_matrix.size(0) + 1]): 
		topk = 128 if sims.shape[0] > 128 else sims.shape[0]
		topk_sim, topk_idx = sims.topk(k=topk, dim=0)
		score_matrix_t2i[i,topk_idx] = topk_sim

	total_time = time.time() - start_time
	total_time_str = str(datetime.timedelta(seconds=int(total_time)))
	# print('Evaluation time {}'.format(total_time_str)) 

	return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()


@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, txt2img, img2txt):
	
	#Images->Text 
	ranks = np.zeros(scores_i2t.shape[0])
	# print("TR: ", len(ranks))
	for index, score in enumerate(scores_i2t):
		inds = np.argsort(score)[::-1]
		# Score
		rank = 1e20
		for i in img2txt[index]:
			tmp = np.where(inds == i)[0][0]
			if tmp < rank:
				rank = tmp
		ranks[index] = rank

	# Compute metrics
	tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
	tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
	tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
  
	#Text->Images 
	ranks = np.zeros(scores_t2i.shape[0])
	# print("IR: ", len(ranks))
	
	for index,score in enumerate(scores_t2i):
		inds = np.argsort(score)[::-1]
		ranks[index] = np.where(inds == txt2img[index])[0][0]

	# Compute metrics
	ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
	ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
	ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)        

	tr_mean = (tr1 + tr5 + tr10) / 3
	ir_mean = (ir1 + ir5 + ir10) / 3
	r_mean = (tr_mean + ir_mean) / 2

	eval_result =  {'txt_r1': tr1,
					'txt_r5': tr5,
					'txt_r10': tr10,
					'txt_r_mean': tr_mean,
					'img_r1': ir1,
					'img_r5': ir5,
					'img_r10': ir10,
					'img_r_mean': ir_mean,
					'r_mean': r_mean}
	return eval_result


def evaluate_synset(it_eval, net, images_train, labels_train, testloader, args, bert_test_embed, return_loss=False, similarity_train=None):
	
	net = net.to(args.device)
	images_train = images_train.to(args.device)
	labels_train = labels_train.to(args.device)
	Epoch = int(args.epoch_eval_train)
	optimizer = torch.optim.SGD([
		{'params': net.image_encoder.parameters(), 'lr': args.lr_teacher_img},
		{'params': net.text_projection.parameters(), 'lr': args.lr_teacher_txt},
	], lr=0, momentum=0.9, weight_decay=0.0005)
	lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[Epoch//2+1], gamma=0.1)

	dst_train = TensorDataset(images_train, labels_train)
	trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_size_train, shuffle=True, num_workers=0)

	start = time.time()
	acc_train_list = []
	loss_train_list = []

	best_val_result = None
	for ep in tqdm(range(args.epoch_eval_train), ncols=60):
		loss_train, acc_train = epoch(ep, trainloader, net, optimizer, args)
		acc_train_list.append(acc_train)
		loss_train_list.append(loss_train)
  
		if ep > 0 and ep % args.eval_eval_freq == 0 or ep == Epoch-1:
			with torch.no_grad():
				score_val_i2t, score_val_t2i = epoch_test(testloader, net, args.device, bert_test_embed)
				val_result = itm_eval(score_val_i2t, score_val_t2i, testloader.dataset.txt2img, testloader.dataset.img2txt) 
				
				print("[Eval_{it:02d}] Ep{ep} || Image R@1={img_r1:.2f} R@5={img_r5:.2f} R@10={img_r10:.2f} R_Mean={r_mean:.2f} || Text R@1={txt_r1:.2f} R@5={txt_r5:.2f} R@10={txt_r10:.2f} R_Mean={txt_r_mean:.2f} || Mean={r_mean:.2f}".format(
					it=it_eval, ep=ep, **val_result
				))
				if best_val_result is None or val_result["r_mean"] > best_val_result["r_mean"]:
					best_val_result = val_result
					
				# if args.wandb:
				# 	for k, v in val_result.items():
				# 		wandb.log({f"Eval/{k}": v})
		lr_scheduler.step()

	time_train = time.time() - start
	return net, acc_train_list, best_val_result


def evaluate_synset_with_similarity(it_eval, net, images_train, labels_train, lr_img, lr_txt, similarity_train, testloader, args, bert_test_embed, mom=0.9, l2=0.0005):
	"""Added for LoRS, see Algorithm 2 in the paper"""

	net = net.to(args.device)
	images_train = images_train.to(args.device)
	labels_train = labels_train.to(args.device)
	similarity_train = similarity_train.to(args.device)
	Epoch = int(args.epoch_eval_train)
	optimizer = torch.optim.SGD([
		{'params': net.image_encoder.parameters(), 'lr': lr_img},
		{'params': net.text_projection.parameters(), 'lr': lr_txt},
	], lr=0, momentum=mom, weight_decay=l2)
	lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[Epoch//2+1], gamma=0.1)

	trainloader = SimilarityDataloader(images_train, labels_train, similarity_train, batch_size=args.batch_size_train)

	start = time.time()
	acc_train_list = []
	loss_train_list = []

	best_val_result = None
	for ep in tqdm(range(args.epoch_eval_train), ncols=60):
		loss_train, acc_train = epoch(ep, trainloader, net, optimizer, args)
		acc_train_list.append(acc_train)
		loss_train_list.append(loss_train)
		if ep > 0 and ep % args.eval_eval_freq == 0 or ep == Epoch-1:
			with torch.no_grad():
				score_val_i2t, score_val_t2i = epoch_test(testloader, net, args.device, bert_test_embed)
				val_result = itm_eval(score_val_i2t, score_val_t2i, testloader.dataset.txt2img, testloader.dataset.img2txt)

				print("[Eval_{it:02d}] Ep{ep} || Image R@1={img_r1:.2f} R@5={img_r5:.2f} R@10={img_r10:.2f} R_Mean={r_mean:.2f} || Text R@1={txt_r1:.2f} R@5={txt_r5:.2f} R@10={txt_r10:.2f} R_Mean={txt_r_mean:.2f} || Mean={r_mean:.2f}".format(
					it=it_eval, ep=ep, **val_result
				))
				if best_val_result is None or val_result["r_mean"] > best_val_result["r_mean"]:
					best_val_result = val_result

				# if args.wandb:
				# 	for k, v in val_result.items():
				# 		wandb.log({f"Eval/{k}": v})
		lr_scheduler.step()
	
	time_train = time.time() - start
	assert best_val_result is not None
	return net, acc_train_list, best_val_result


# def evaluate_synset_aug(it_eval, net, images_train, labels_train, testloader, args, bert_test_embed, return_loss=False, similarity_train=None):
	
#     net = net.to(args.device)
#     images_train = images_train.to(args.device)
#     labels_train = labels_train.to(args.device)
#     Epoch = int(args.epoch_eval_train)
#     optimizer = torch.optim.SGD([
#         {'params': net.image_encoder.parameters(), 'lr': args.lr_teacher_img},
#         {'params': net.text_projection.parameters(), 'lr': args.lr_teacher_txt},
#     ], lr=0, momentum=0.9, weight_decay=0.0005)
#     lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[Epoch//2+1], gamma=0.1)

#     if similarity_train is None:
#         dst_train = TensorDataset(images_train, labels_train)
#         trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_size_train, shuffle=True, num_workers=16)
#     else:
#         trainloader = SimilarityDataloader(images_train, labels_train, similarity_train, batch_size=args.batch_size_train)
		
#     start = time.time()
#     acc_train_list = []
#     loss_train_list = []

#     best_val_result = None
#     for ep in tqdm(range(Epoch), ncols=60):
#         loss_train, acc_train = epoch(ep, trainloader, net, optimizer, args, aug=True)
#         acc_train_list.append(acc_train)
#         loss_train_list.append(loss_train)
#         if ep > 0 and ep % args.eval_eval_freq == 0 or ep == Epoch:
#             with torch.no_grad():
#                 score_val_i2t, score_val_t2i = epoch_test(testloader, net, args.device, bert_test_embed)
#                 val_result = itm_eval(score_val_i2t, score_val_t2i, testloader.dataset.txt2img, testloader.dataset.img2txt) 
				
#                 print("[Eval_{it:02d}] Ep{ep} | Image R@1={img_r1:.2f} R@5={img_r5:.2f} R@10={img_r10:.2f} | Text R@1={txt_r1:.2f} R@5={txt_r5:.2f} R@10={txt_r10:.2f} | Mean={r_mean:.2f}".format(
#                     it=it_eval, ep=ep, **val_result
#                 ))
#                 if best_val_result is None or val_result["r_mean"] > best_val_result["r_mean"]:
#                     best_val_result = val_result
					
#         lr_scheduler.step()

#     time_train = time.time() - start
#     return net, acc_train_list, best_val_result