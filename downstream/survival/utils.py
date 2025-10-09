import pickle
import torch
import numpy as np
import torch.nn as nn
import pdb
import os
import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler, sampler
from transformers import get_cosine_schedule_with_warmup
import torch.optim as optim
import pdb
import torch.nn.functional as F
import math
from itertools import islice
import collections
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name='unk', fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class SubsetSequentialSampler(Sampler):
	"""Samples elements sequentially from a given list of indices, without replacement.

	Arguments:
		indices (sequence): a sequence of indices
	"""
	def __init__(self, indices):
		self.indices = indices

	def __iter__(self):
		return iter(self.indices)

	def __len__(self):
		return len(self.indices)

def collate_MIL_mtl_sex(batch):
	img = torch.cat([item[0] for item in batch], dim = 0)
	label = torch.LongTensor([item[1] for item in batch])
	site = torch.LongTensor([item[2] for item in batch])
	sex = torch.LongTensor([item[3] for item in batch])
	# for item in batch:
	# 	print(item)
	return [img, label, site, sex]

def collate_MIL_mtl(batch):
	img = torch.cat([item[0] for item in batch], dim = 0)
	label_task1 = torch.LongTensor([item[1] for item in batch])
	label_task2 = torch.LongTensor([item[2] for item in batch])
	label_task3 = torch.LongTensor([item[3] for item in batch])
	# for item in batch:
	# 	print(item)
	return [img, label_task1, label_task2, label_task3]

def collate_MIL(batch):
	img = torch.cat([item[0] for item in batch], dim = 0)
	label = torch.LongTensor([item[1] for item in batch])
	return [img, label]

def collate_features(batch):
	img = torch.cat([item[0] for item in batch], dim = 0)
	coords = np.vstack([item[1] for item in batch])
	return [img, coords]


collate_dict = {'MIL': collate_MIL, 'MIL_mtl': collate_MIL_mtl, 'MIL_mtl_sex': collate_MIL_mtl_sex, 'MIL_sex': collate_MIL_mtl}

def get_simple_loader(dataset, batch_size=1, collate_fn='MIL'):
	kwargs = {'num_workers': 32} if device.type == "cuda" else {}
	collate = collate_dict[collate_fn]
	loader = DataLoader(dataset, batch_size=batch_size, sampler = sampler.SequentialSampler(dataset), collate_fn = collate, **kwargs)
	return loader

def get_split_loader(split_dataset, training = False, testing = False, weighted = False, collate_fn='MIL'):
	"""
		return either the validation loader or training loader
	"""
	collate = collate_dict[collate_fn]

	kwargs = {'num_workers': 4} if device.type == "cuda" else {}
	if not testing:
		if training:
			if weighted:
				weights = make_weights_for_balanced_classes_split(split_dataset)
				loader = DataLoader(split_dataset, batch_size=1, sampler = WeightedRandomSampler(weights, len(weights)), collate_fn = collate, **kwargs)
			else:
				loader = DataLoader(split_dataset, batch_size=1, sampler = RandomSampler(split_dataset), collate_fn = collate, **kwargs)
		else:
			loader = DataLoader(split_dataset, batch_size=1, sampler = SequentialSampler(split_dataset), collate_fn = collate, **kwargs)

	else:
		ids = np.random.choice(np.arange(len(split_dataset)), int(len(split_dataset)*0.01), replace = False)
		loader = DataLoader(split_dataset, batch_size=1, sampler = SubsetSequentialSampler(ids), collate_fn = collate, **kwargs )

	return loader

def get_optim(model, args):
	if args.opt == "adam":
		optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.reg)
	elif args.opt == 'sgd':
		optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.reg)
	else:
		raise NotImplementedError
	return optimizer

def print_network(net):
	num_params = 0
	num_params_train = 0
	print(net)

	for param in net.parameters():
		n = param.numel()
		num_params += n
		if param.requires_grad:
			num_params_train += n

	print('Total number of parameters: %d' % num_params)
	print('Total number of trainable parameters: %d' % num_params_train)


def generate_split(cls_ids, val_num, test_num, samples, n_splits = 5,
	seed = 7, label_frac = 1.0, custom_test_ids = None):
	indices = np.arange(samples).astype(int)

	if custom_test_ids is not None:
		indices = np.setdiff1d(indices, custom_test_ids)

	np.random.seed(seed)
	for i in range(n_splits):
		all_val_ids = []
		all_test_ids = []
		sampled_train_ids = []

		if custom_test_ids is not None: # pre-built test split, do not need to sample
			all_test_ids.extend(custom_test_ids)

		for c in range(len(val_num)):
			if c == 38:
				pdb.set_trace()
			possible_indices = np.intersect1d(cls_ids[c], indices) #all indices of this class
			remaining_ids = possible_indices

			if val_num[c] > 0:
				val_ids = np.random.choice(possible_indices, val_num[c], replace = False) # validation ids
				remaining_ids = np.setdiff1d(possible_indices, val_ids) #indices of this class left after validation
				all_val_ids.extend(val_ids)

			if custom_test_ids is None and test_num[c] > 0: # sample test split

				test_ids = np.random.choice(remaining_ids, test_num[c], replace = False)
				remaining_ids = np.setdiff1d(remaining_ids, test_ids)
				all_test_ids.extend(test_ids)

			if label_frac == 1:
				sampled_train_ids.extend(remaining_ids)

			else:
				sample_num  = math.ceil(len(remaining_ids) * label_frac)
				slice_ids = np.arange(sample_num)
				sampled_train_ids.extend(remaining_ids[slice_ids])

		yield sampled_train_ids, all_val_ids, all_test_ids


def nth(iterator, n, default=None):
	if n is None:
		return collections.deque(iterator, maxlen=0)
	else:
		return next(islice(iterator,n, None), default)

def calculate_error(Y_hat, Y):
	error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()

	return error

def make_weights_for_balanced_classes_split(dataset):
	N = float(len(dataset))
	weight_per_class = [N/len(dataset.slide_cls_ids[c]) for c in range(len(dataset.slide_cls_ids))]
	weight = [0] * int(N)
	for idx in range(len(dataset)):
		y = dataset.getlabel(idx)
		weight[idx] = weight_per_class[y]

	return torch.DoubleTensor(weight)

def initialize_weights(module):
	for m in module.modules():
		if isinstance(m, nn.Linear):
			nn.init.xavier_normal_(m.weight)
			m.bias.data.zero_()

		elif isinstance(m, nn.BatchNorm1d):
			nn.init.constant_(m.weight, 1)
			nn.init.constant_(m.bias, 0)


def log_dict_tensorboard(writer, results, str_prefix, step = 0, verbose = False):
    for k, v in results.items():
        if verbose:
            print(f'{k}: {v:.4f}')
        writer.add_scalar(f'{str_prefix}{k}', v, step)
    return writer

def get_lr_scheduler(args, optimizer, dataloader):
    warmup_steps = args.warmup_steps
    warmup_epochs = args.warmup_epochs
    
    epochs = args.max_epochs if hasattr(args, 'max_epochs') else args.epochs
    
    assert not (warmup_steps > 0 and warmup_epochs > 0)
    
    accum_steps = args.accum_steps
    
    if warmup_steps > 0:
        warmup_steps = warmup_steps
    elif warmup_epochs > 0:
        warmup_steps = warmup_epochs * (len(dataloader)//accum_steps)
    else:
        warmup_steps = 0
        
        
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer = optimizer, num_warmup_steps = warmup_steps,
                                                  num_training_steps = (len(dataloader)//accum_steps * epochs))
    
    return lr_scheduler


def save_checkpoint(model, optimizer, epoch, val_loss, train_loss, c_index_train, c_index_val, args):
	checkpoint = {
		'epoch': epoch,
		'model_state_dict': model.state_dict(),
		'optimizer_state_dict': optimizer.state_dict(),
		'val_loss': val_loss,
		'val_c_index': c_index_val,
		'train_loss': train_loss,
		'train_c_index': c_index_train,
		'args': args
	}

	checkpoint_path = os.path.join(args.results_dir, 
								f'checkpoint_epoch_{epoch}_val_{val_loss}_c_ind_{c_index_val}.pth')
	
	torch.save(checkpoint, checkpoint_path)

	log_path = os.path.join(args.results_dir, 'training_val_log.csv')
	if not os.path.exists(log_path):
		with open(log_path, 'w') as f:
			f.write('epoch,train_loss,c_index_train,val_loss,c_index_val\n')

	with open(log_path, 'a') as f:
		f.write(f'{epoch},{train_loss:.4f},{c_index_train:.4f},{val_loss:.4f},{c_index_val:4f}\n')

