import torch
from torch.nn.utils.rnn import pad_sequence
from model.model import ModelConfig
import argparse


def ZeroPadMask(*input_tensor):
    if len(input_tensor) == 1:
        input_tensor = input_tensor[0]
        masks_tensor = torch.zeros_like(input_tensor,dtype = torch.long)
        masks_tensor = masks_tensor.masked_fill(input_tensor != 0, 1)
        masks_tensor = (1.0 - masks_tensor) * -10000.0
        return masks_tensor[:,None,None,:]
    else:
        target_tensor,input_tensor = input_tensor
        masks_tensor_i = torch.zeros_like(input_tensor,dtype = torch.long)
        masks_tensor_i = masks_tensor_i.masked_fill(input_tensor != 0, 1)
        masks_tensor_t = torch.zeros_like(target_tensor,dtype = torch.long)
        masks_tensor_t = masks_tensor_t.masked_fill(target_tensor != 0, 1)
        masks_tensor = torch.matmul(masks_tensor_t.unsqueeze(-1),masks_tensor_i.unsqueeze(1))
        masks_tensor = (1.0 - masks_tensor) * -10000.0
        return masks_tensor[:,None,:,:]

def FutureMask(input_tensor):
	device = input_tensor.device
	masks_tensor = torch.zeros_like(input_tensor,dtype = torch.long,device = device) # (B,S)
	masks_tensor = masks_tensor.masked_fill(input_tensor != 0, 1)
	batch_size ,seq_len = input_tensor.size()
	seq_ids = torch.arange(seq_len,device = device)
	causal_mask = seq_ids[None,None, :].repeat(batch_size,seq_len, 1) <= seq_ids[:, None]
	causal_mask = causal_mask.to(torch.long)
	causal_mask = (causal_mask[:,None, :, :]*masks_tensor[:,None, None, :])
	causal_mask = (1.0 - causal_mask) * -10000.0
	return causal_mask


def collate_fn(batch):
	input_tensor = [i[0] for i in batch]
	target_tensor = [i[1] for i in batch]

	input_tensor = pad_sequence(input_tensor,batch_first=True)
	target_tensor = pad_sequence(target_tensor,batch_first=True)

	masks_tensor = FutureMask(input_tensor)
	
	return input_tensor,target_tensor,masks_tensor



def parse_args():
	parser = argparse.ArgumentParser()

	# data
	parser.add_argument('--data_root', type = str, default = './data/corpus_v2')
	parser.add_argument('--char_dict', type = str, default = './data/HPchars.txt')

	# training contig
	parser.add_argument('--batch_size', type = int, default = 16)
	parser.add_argument('--epochs', type = int, default = 20)
	parser.add_argument('--output_folder', type = str, default = 'output_3_2')
	parser.add_argument('--resume_path', type = str, default = '') #./output/output_3_2/weight/save_epoch16.pth
	parser.add_argument('--max_len', type = int, default = 100)
	parser.add_argument('--temperature', type = float, default = 0.4)


	# model config
	parser.add_argument('--vocab_size', type = int, default = 4096)
	parser.add_argument('--embedding_size', type = int, default = 128)
	parser.add_argument('--num_hidden_layers', nargs='+', type = int, default = [6,6],
						help = '')
	parser.add_argument('--position_embedding_mode', type = str, default = 'BERT',
						help = '(BERT|Transformer)')
	parser.add_argument('--max_position_embeddings', type = int, default = 512,
						help = 'Max input constraint')
	parser.add_argument('--is_share', type = str, default = 'True',
						help = 'the embedding in UT is shared or not ')
	parser.add_argument('--dropout_p', type = float, default = 0.1,
						help = 'Dropout probability')
	parser.add_argument('--act_epsilon', type = float, default = 0.01,
						help = 'ACT parameter')
	parser.add_argument('--num_heads', type = int, default = 8,
						help = 'Number of attention heads')
	parser.add_argument('--transition_type', type = str, default = 'fc',
						help = '(fc|conv)')
	parser.add_argument('--act_type', type = str, default = 'accumulated',
						help = '(accumulated|basic)')
	parser.add_argument('--time_penalty', type = float, default = 0.001,
						help = 'ACT parameter')

	# optimizer config
	parser.add_argument('--optimizer', type = str, default = 'AdamW',
						help = 'only support: (Adam|AdamW)')
	parser.add_argument('--learning_rate', type = float, default = 0.0001)
	parser.add_argument('--beta1', type=float, default=0.9,
	                    help='Optimizer parameter: beta1.')
	parser.add_argument('--beta2', type=float, default=0.999,
	                    help='Optimizer parameter: beta2.')
	parser.add_argument('--eps', type=float, default=1e-6,
	                    help='Optimizer parameter: eps.')
	parser.add_argument('--weight_decay', type=float, default=0.01,
	                    help='Optimizer parameter: eps.')

	return parser.parse_args()


def config_setting(args):
	is_share = True if (args.is_share).lower() in ['true','t','y'] else False

	return ModelConfig(vocab_size = args.vocab_size,
					   embedding_size = args.embedding_size,
					   num_hidden_layers = args.num_hidden_layers,
					   position_embedding_mode = args.position_embedding_mode,
					   max_position_embeddings = args.max_position_embeddings,
					   is_share = is_share,
					   dropout_p = args.dropout_p,
					   act_epsilon = args.act_epsilon,
					   num_heads = args.num_heads,
					   transition_type = args.transition_type,
					   act_type = args.act_type,
					   time_penalty = args.time_penalty)

