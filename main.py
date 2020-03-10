from data.dataset import HPcorpus
from torch.utils.data import DataLoader
from utils import collate_fn,parse_args,config_setting
from model.model import UniversalTransformerDecoder
import torch
import torch.nn as nn
import time
import tqdm 
import os
from tensorboardX import SummaryWriter


if __name__ == '__main__':
	# argument setting
	args = parse_args()
	model_config = config_setting(args)

	# output path setting
	if not os.path.exists('./output'):
		os.mkdir('./output')

	output_path = os.path.join('./output',args.output_folder)
	if not os.path.exists(output_path):
		os.mkdir(output_path)

	output_w_path = os.path.join(output_path,'weight')
	if not os.path.exists(output_w_path):
		os.mkdir(output_w_path)

	output_TB_path = os.path.join(output_path,'tensorboard')
	if not os.path.exists(output_TB_path):
		os.mkdir(output_TB_path)

	# data
	data_root,char_dict = args.data_root,args.char_dict
	train_data = HPcorpus(data_root,char_dict)
	train_loader = DataLoader(train_data,
							  batch_size=args.batch_size,
	            			  shuffle=True,
	            			  pin_memory=True,
	            			  collate_fn = collate_fn)

	# model 
	model = UniversalTransformerDecoder(model_config)
	no_decay = ["bias", "LayerNorm.weight"]
	optimizer_grouped_parameters = [{"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
	         						 "weight_decay": args.weight_decay,},
	        						{"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},]

	# optimizer 
	if args.optimizer == 'Adam':
		optimizer = torch.optim.Adam(optimizer_grouped_parameters,
									 lr = args.learning_rate,
									 betas = (args.beta1, args.beta2), 
									 eps = args.eps, 
									 weight_decay = args.weight_decay)
	elif args.optimizer == 'AdamW':
		optimizer = torch.optim.AdamW(optimizer_grouped_parameters,
									  lr = args.learning_rate,
									  betas = (args.beta1, args.beta2), 
									  eps = args.eps)

	begin_epoch = 0

	# resume model
	if args.resume_path:
		checkpoint = torch.load(args.resume_path)
		begin_epoch = checkpoint['epoch']
		model.load_state_dict(checkpoint['state_dict'])
		
		for state in optimizer.state.values():
			for k, v in state.items():
				if isinstance(v, torch.Tensor):
					state[k] = v.cuda()

	# device setting
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(device)
	model = model.to(device)


	# tensorboard writer
	writer_path = output_TB_path
	writer = SummaryWriter(writer_path)


	# loss 
	loss_fn = nn.CrossEntropyLoss()

	def adjust_learning_rate(optimizer, epoch):
		"""Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
		lr = args.learning_rate * (0.1 ** (epoch//100))
		print(f'learning rate = {lr}') 
		for param_group in optimizer.param_groups:
			param_group['lr'] = lr

	# training
	best_loss = 9999.0
	stop_early = 0
	for i in range(begin_epoch,begin_epoch + args.epochs):
		adjust_learning_rate(optimizer,i)
		
		train_loss = 0.0
		start_time = time.time()
		model.train()
		for j, batch_data in enumerate(tqdm.tqdm(train_loader)):
			input_tensor,target_tensor,masks_tensor = [t.to(device) for t in batch_data]

			outputs,num_updates,remainders = model(input_ids=input_tensor,
												   input_mask=masks_tensor
										)
			active_pos = target_tensor != 0
			_,_,V = outputs.size()
			outputs = outputs[active_pos].view(-1,V)
			target_tensor = target_tensor[active_pos].view(-1)
			num_updates = num_updates[active_pos].sum()
			remainders = remainders[active_pos].sum()
			ponder_cost = num_updates + remainders
			loss = loss_fn(outputs,target_tensor) 
			if j%10 == 0:
				print()
				print(f'Epoch {i} Batch {j} Training Loss = {loss}')
			loss = loss + (model_config.time_penalty*ponder_cost)
			if j%10 == 0:
				print(f'Epoch {i} Batch {j} Training Loss adds ponder cost= {loss}')
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			train_loss += loss.item()

			

		print(f'Epoch {i} Training Loss = {train_loss/(j+1)}')
		print(f'Epoch {i} Training Time = {time.time()-start_time}')
		writer.add_scalar('train/loss',(train_loss/(j+1)),i)

		
		if i % 10 == 9:
			save_file_path = os.path.join(output_w_path,'save_epoch{}.pth'.format(i))
			states = {'epoch': i + 1,
					  'state_dict': model.state_dict(),
					  'optimizer': optimizer.state_dict(),}
			torch.save(states, save_file_path)

	# save final state
	save_file_path = os.path.join(output_w_path,'save_epoch{}.pth'.format(i))
	states = {'epoch': i + 1,
			  'state_dict': model.state_dict(),
			   'optimizer': optimizer.state_dict(),}
	torch.save(states, save_file_path)