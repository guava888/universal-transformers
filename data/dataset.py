from torch.utils.data import Dataset
from .utils import load_corpus
import torch 

class HPcorpus(Dataset):
	def __init__(self,data_root,char_dict):
		self.sents = load_corpus(data_root,char_dict)

	def __getitem__(self,index):
		ids = self.sents[index]
		
		input_ids = ids[:-1]
		input_ids = torch.LongTensor(input_ids)

		target_ids = ids[1:]
		target_ids = torch.LongTensor(target_ids)

		return input_ids,target_ids
	def __len__(self):
		return len(self.sents)