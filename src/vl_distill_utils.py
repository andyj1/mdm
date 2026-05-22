"""Move some basic utils in distill.py in VL-Distill here"""
import os
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

from src.networks import TextEncoder, CLIPModel_full
from tqdm import tqdm
__all__ = [
	"shuffle_files",
	"nearest_neighbor",
	"get_images_texts",
	"load_or_process_file",
	"textprocess",
	"textprocess_train",
	"coreset",
]


#################################################################
# coreset
#################################################################
def coreset(method, dataset, num_syn, args):
	if method == 'herding':
		image_syn = torch.randn([num_syn, 3, args.image_size, args.image_size])
		text_syn = torch.randn([num_syn, args.text_dim])
		
	elif method == 'kcenter':
		image_syn = torch.randn([num_syn, 3, args.image_size, args.image_size])
		text_syn = torch.randn([num_syn, args.text_dim])
		
	elif method == 'coreset':
		image_syn = torch.randn([num_syn, 3, args.image_size, args.image_size])
		text_syn = torch.randn([num_syn, args.text_dim])
	else:
		raise NotImplementedError(f"Method {method} not implemented")

	return image_syn, text_syn

#################################################################
# shuffle expert buffers
#################################################################

def shuffle_files(img_expert_files, txt_expert_files):
	# Check if both lists have the same length and if the lists are not empty
	assert len(img_expert_files) == len(txt_expert_files), "Number of image files and text files does not match"
	assert len(img_expert_files) != 0, "No files to shuffle"
	shuffled_indices = np.random.permutation(len(img_expert_files))

	# Apply the shuffled indices to both lists
	img_expert_files = np.take(img_expert_files, shuffled_indices)
	txt_expert_files = np.take(txt_expert_files, shuffled_indices)
	return img_expert_files, txt_expert_files

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
	nearest_neighbors = []
	
	for query in query_embeddings:
		similarities = cosine_similarity(query.reshape(1, -1), database_embeddings)
		
		most_similar_index = np.argmax(similarities)
		
		nearest_neighbors.append(sentences[most_similar_index])
		
	return nearest_neighbors

#################################################################
# fetch random samples from the real dataset
#################################################################

def get_images_texts(n, dataset, args, text_encoder=None, i_have_indices=None, init='random'):
	"""Get random n images and corresponding texts from the dataset.

	Args:
	n: Number of images and texts to retrieve.
	dataset: The dataset containing image-text pairs.

	Returns:
	A tuple containing two elements:
	  - A tensor of randomly selected images.
	  - A tensor of the corresponding texts, encoded as floats.
	"""
	if init == 'random':
		# Generate n unique random indices
		if i_have_indices is not None:
			idx_shuffle = i_have_indices
		else:
			idx_shuffle = np.random.permutation(len(dataset))[:n]

		# Initialize the text encoder
		if text_encoder is None:
			text_encoder = TextEncoder(args).to(args.device)

		image_syn = torch.stack([dataset[i][0] for i in idx_shuffle])
		
		# text_syn = text_encoder([dataset[i][1] for i in idx_shuffle], device='cpu')
		text_syn = text_encoder([dataset[i][1] for i in idx_shuffle])

	elif init == 'noise':
		mean = torch.tensor([-0.0626, -0.0221,  0.0680])
		std  = torch.tensor([1.0451, 1.0752, 1.0539])
		image_syn = torch.randn([args.num_queries, 3, 224, 224])
		for c in range(3):
			image_syn[:, c] = image_syn[:, c] * std[c] + mean[c]
	
		text_syn = torch.normal(mean=-0.0094, std=0.5253, size=(args.num_queries, 768))
	
	else:
		raise NotImplementedError(f"Initialization method {init} not implemented")

	return image_syn, text_syn.float()

#################################################################
# text caption loading
#################################################################
@torch.no_grad()
def textprocess_test(args, testloader, device='cuda'):
	net = CLIPModel_full(args).to(device)
	net.eval() 
	captions = testloader.dataset.text 

	chunk_size = 2000
	chunks = []
	for i in tqdm(range(0, len(captions), chunk_size)):
		chunk = net.text_encoder(captions[i:i + chunk_size]).cpu()
		chunks.append(chunk)
		del chunk
		torch.cuda.empty_cache()  # free up memory
	bert_test_embed = torch.cat(chunks, dim=0)
	bert_test_embed = bert_test_embed.cpu().numpy()
	np.savez(f'{args.text_embed_dir}/{args.dataset}_{args.text_encoder}_test_text_embed.npz', bert_test_embed=bert_test_embed) 
	
	return 

@torch.no_grad()
def textprocess_train(args, train_sentences, device='cuda'):
	net = CLIPModel_full(args).to(device)
	net.eval() 
	chunk_size = 2000
	chunks = []
	for i in tqdm(range(0, len(train_sentences), chunk_size)):
		chunk = net.text_encoder(train_sentences[i:i + chunk_size]).cpu()
		chunks.append(chunk)
		del chunk
		torch.cuda.empty_cache()  # free up memory
	bert_train_embed = torch.cat(chunks, dim=0)
	bert_train_embed = bert_train_embed.numpy()
	
	np.savez(f'{args.text_embed_dir}/{args.dataset}_{args.text_encoder}_train_text_embed.npz', bert_train_embed=bert_train_embed)

	return 

def load_or_process_file(file_type, process_func, args, data_source):
	"""
	Load the processed file if it exists, otherwise process the data source and create the file.

	Args:
	file_type: The type of the file (e.g., 'train', 'test').
	process_func: The function to process the data source.
	args: The arguments required by the process function and to build the filename.
	data_source: The source data to be processed.

	Returns:
	The loaded data from the file.
	"""
	os.makedirs(args.text_embed_dir, exist_ok=True)
	filename = f'{args.text_embed_dir}/{args.dataset}_{args.text_encoder}_{file_type}_embed.npz'


	if not os.path.exists(filename):
		print(f'Creating {filename}')
		process_func(args, data_source)
	else:
		print(f'Loading {filename}')
	
	return np.load(filename)

def get_LC_images_texts(n, dataset, args):
	"""Get random n images and corresponding texts from the dataset.

	Args:
	n: Number of images and texts to retrieve.
	dataset: The dataset containing image-text pairs.

	Returns:
	A tuple containing two elements:
	  - A tensor of randomly selected images.
	  - A tensor of the corresponding texts, encoded as floats.
	"""
	# Generate n unique random indices
	idx_shuffle = np.random.permutation(len(dataset))[:n]

	# Initialize the text encoder
	text_encoder = TextEncoder(args)

	image_syn = torch.stack([dataset[i][0] for i in idx_shuffle])
	
	text_syn = text_encoder([dataset[i][1] for i in idx_shuffle], device="cpu")

	return image_syn, text_syn.float()

