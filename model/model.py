import torch
import torch.nn as nn
import math

# config:
# vocab_size = 23149 (this is default, if need add other token must change this parameter)
# embedding_size = 768 (can try small one, 128)
# num_hidden_layers = [num_encoder_layers,num_decoder_layers]
# position_embedding_mode \in {'BERT','Transformer'}
# max_position_embeddings = 512 (restrict the length of input and output)
# is_share = True/False,must set when position_embedding_mode == 'BERT', 
#            whether share positional embedding and time embedding or not


class ModelConfig(object):
	def __init__(self,
				vocab_size = 4096,#4095
				embedding_size = 128,
				num_hidden_layers = [6,6],
				position_embedding_mode = 'BERT',
				max_position_embeddings = 512,
				is_share = True,
				dropout_p = 0.1,
				act_epsilon = 0.01,
				num_heads = 8,
				transition_type = 'fc',
				act_type = 'accumulated',
				time_penalty = 0 #0.01
				):
		self.vocab_size = vocab_size
		self.embedding_size = embedding_size
		
		if len(num_hidden_layers) == 2:
			self.num_encoder_layers = num_hidden_layers[0]
			self.num_decoder_layers = num_hidden_layers[1]
		else:
			raise ValueError

		if position_embedding_mode in ['BERT','Transformer']:
			self.position_embedding_mode = position_embedding_mode
		else:
			raise ValueError(f'Do not support the mode {position_embedding_mode}')

		self.max_position_embeddings = max_position_embeddings
		self.is_share = is_share
		self.dropout_p = dropout_p
		self.act_epsilon = act_epsilon
		self.num_heads = num_heads
		if transition_type in ['fc','conv']:
			self.transition_type = transition_type
		else:
			raise ValueError

		if act_type in ['basic','accumulated']:
			self.act_type = act_type
		self.time_penalty = time_penalty



class PositionalEmbedding(nn.Module):
	def __init__(self,max_len,embedding_size):
		super(PositionalEmbedding, self).__init__()

		position = torch.arange(float(max_len)).unsqueeze(1)
		scale = 1/(torch.pow(10000.0,torch.arange(0.0,embedding_size,2)/embedding_size).unsqueeze(0))
		position = torch.mm(position,scale)
		self.PE = torch.cat([torch.sin(position),torch.cos(position)],dim=1)
		
	def forward(self,input_ids):
		'''
		Args:
			input_ids (B,S)

		Return:
			position_embedding (B,S,D)

		Example:
		max_len, embedding_dim = 6,10
		PE = PositionalEmbedding(max_len, embedding_dim)
		input_ids = torch.rand(3,4)
		input_pe = PE(input_ids) #with size (3,4,10)

		'''
		batch_size, seq_len = input_ids.size() 
		position_embedding = self.PE[:seq_len,:].unsqueeze(0).expand(batch_size,-1,-1)

		# if device is setting 'cuda:1' might occur error.
		if input_ids.is_cuda:
			position_embedding = position_embedding.cuda()

		return position_embedding



class UTEmbedding(nn.Module):
	def __init__(self,config):
		super(UTEmbedding, self).__init__()

		self.WordEmbedding = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=0)
		if config.position_embedding_mode == 'BERT':
			if config.is_share:
				self.PositionEmbedding = nn.Embedding(config.max_position_embeddings,config.embedding_size)
				self.TimeEmbedding = nn.Embedding(config.num_encoder_layers,config.embedding_size)
			else:
				self.PositionEmbedding = [nn.Embedding(config.max_position_embeddings,config.embedding_size),
										   nn.Embedding(config.max_position_embeddings,config.embedding_size)]
				self.TimeEmbedding = nn.Embedding(config.num_encoder_layers+config.num_decoder_layers,config.embedding_size)
		else:
			self.PositionEmbedding = PositionalEmbedding(config.max_position_embeddings,config.embedding_size)
			if config.is_share:
				self.TimeEmbedding = PositionalEmbedding(config.num_encoder_layers,config.embedding_size)
			else:
				self.TimeEmbedding = PositionalEmbedding(config.num_encoder_layers+config.num_decoder_layers,config.embedding_size)

		self.num_encoder_layers = config.num_encoder_layers
		self.num_decoder_layers = config.num_decoder_layers
		self.is_share = config.is_share

	def forward(self,input_ids,target_ids):
		'''
		Args:
			input_ids (B,S_i)
			target_ids (B,S_t)

		Return:
			output_emds_i (B,S_i,D)			
			output_emds_t (B,S_t,D)			
			time_embedding_i (1,num_encoder_layers+num_decoder_layers,D)

		'''
		batch_size, seq_len_i = input_ids.size() 
		_, seq_len_t = target_ids.size() 

		word_embedding_i = self.WordEmbedding(input_ids) # (B,S_i,D)
		word_embedding_t = self.WordEmbedding(target_ids) # (B,S_t,D)

		input_pos = torch.arange(seq_len_i).unsqueeze(0).expand(batch_size,-1)
		target_pos = torch.arange(seq_len_t).unsqueeze(0).expand(batch_size,-1)
		encoder_ts = torch.arange(self.num_encoder_layers).unsqueeze(0)
		decoder_ts = torch.arange(self.num_encoder_layers,self.num_encoder_layers+self.num_decoder_layers).unsqueeze(0)

		if input_ids.is_cuda:
			encoder_ts = encoder_ts.cuda()
			input_pos = input_pos.cuda()
			target_pos = target_pos.cuda()
			decoder_ts  = decoder_ts .cuda()

		if isinstance(self.PositionEmbedding,list):
			pe_i,pe_t = self.PositionEmbedding
			position_embedding_i = pe_i(input_pos) # (B,S_i,D)
			position_embedding_t = pe_t(target_pos) # (B,S_t,D)
			
		else:
			position_embedding_i = self.PositionEmbedding(input_pos)
			position_embedding_t = self.PositionEmbedding(target_pos)		

		if self.is_share:
			time_embedding_e = self.TimeEmbedding(encoder_ts)
			time_embedding_d = self.TimeEmbedding(encoder_ts)
			time_embedding = torch.cat([time_embedding_e,time_embedding_d],dim = 1)			

		else:	
			all_ts = torch.cat([encoder_ts,decoder_ts],dim = -1)
			time_embedding = self.TimeEmbedding(all_ts)
			# time_embedding_i,time_embedding_t = torch.split(time_embedding,self.num_encoder_layers,dim = 1)
		output_emds_i = word_embedding_i+position_embedding_i
		output_emds_t = word_embedding_t+position_embedding_t

		return (output_emds_i,output_emds_t,time_embedding)

class AlbertEmbedding(nn.Module):
	def __init__(self,config):
		super(AlbertEmbedding, self).__init__()

		self.WordEmbedding = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=0)
		if config.position_embedding_mode == 'BERT':
			self.PositionEmbedding = nn.Embedding(config.max_position_embeddings,config.embedding_size)
			self.TimeEmbedding = nn.Embedding(config.num_encoder_layers,config.embedding_size)		
		else:
			self.PositionEmbedding = PositionalEmbedding(config.max_position_embeddings,config.embedding_size)
			self.TimeEmbedding = PositionalEmbedding(config.num_encoder_layers,config.embedding_size)
			
		self.num_encoder_layers = config.num_encoder_layers

	def forward(self,input_ids):
		'''
		Args:
			input_ids (B,S_i)

		Return:
			output_emds_i (B,S_i,D)			
			time_embedding_i (1,num_encoder_layers/num_decoder_layers,D)

		'''
		batch_size, seq_len_i = input_ids.size() 

		word_embedding_i = self.WordEmbedding(input_ids) # (B,S_i,D)

		input_pos = torch.arange(seq_len_i).unsqueeze(0).expand(batch_size,-1)
		encoder_ts = torch.arange(self.num_encoder_layers).unsqueeze(0)
		if input_ids.is_cuda:
			encoder_ts = encoder_ts.cuda()
			input_pos = input_pos.cuda()
	
		position_embedding_i = self.PositionEmbedding(input_pos)
		time_embedding_e = self.TimeEmbedding(encoder_ts)
		output_emds_i = word_embedding_i+position_embedding_i

		return (output_emds_i,time_embedding_e)


class MultiHeadAttention(nn.Module):
	def __init__(self,config):
		super(MultiHeadAttention,self).__init__()
		self.num_heads = config.num_heads
		self.key = nn.Linear(config.embedding_size,config.embedding_size)
		self.query = nn.Linear(config.embedding_size,config.embedding_size)
		self.value = nn.Linear(config.embedding_size,config.embedding_size)
		self.dropout = nn.Dropout(config.dropout_p)
		self.softmax = nn.Softmax(dim=-1)
		self.output = nn.Linear(config.embedding_size,config.embedding_size)
		

	def transpose_head_size(self,state):
		batch_size,seq_len,emb_size = state.size()
		state = state.view(batch_size,seq_len,self.num_heads,int(emb_size/self.num_heads)) #(B,S,H,Dh)
		return state.permute(0,2,1,3).contiguous()


	def forward(self,hidden_state,attention_mask,encoder_state = None):
		q = self.query(hidden_state) # (B,S,D)

		if encoder_state is not None:
			k = self.key(encoder_state)
			v = self.value(encoder_state)
		else:
			k = self.key(hidden_state) # (B,S,D)
			v = self.value(hidden_state) # (B,S,D)

		q = self.transpose_head_size(q) # (B,H,S,Dh)
		k = self.transpose_head_size(k)
		v = self.transpose_head_size(v)

		norm_term = q.size()[-1]

		attention_scores = torch.matmul(q,k.transpose(-1,-2)) #(B,H,S,S)
		attention_scores = attention_scores / math.sqrt(norm_term)
		attention_scores += attention_mask
		attention_scores = self.softmax(attention_scores)
		attention_scores = self.dropout(attention_scores)

		contexts = torch.matmul(attention_scores,v) #(B,H,S,Dh)
		contexts = contexts.permute(0,2,1,3).contiguous() #(B,S,H,Dh)
		batch_size,seq_len,*_ = contexts.size()
		contexts = contexts.view(batch_size,seq_len,-1) #(B,S,D)

		outputs = self.output(contexts)

		return outputs

class Residual(nn.Module):
	def __init__(self,config):
		super(Residual, self).__init__()
		self.LayerNorm = nn.LayerNorm(config.embedding_size,eps = 1e-12)
		self.dropout = nn.Dropout(config.dropout_p)
	def forward(self,states,outputs):
		outputs = self.dropout(outputs)
		outputs = self.LayerNorm(states + outputs)
		return outputs

class gelu(nn.Module):
	'''
	Reference is form https://arxiv.org/pdf/1606.08415.pdf
	'''
	def forward(self,x):
		return x*(torch.sigmoid(1.702*x))

class FeedForward(nn.Module):
	def __init__(self,config):
		super(FeedForward, self).__init__()
		activation_fn = gelu()#nn.GELU()
		self.layers = nn.Sequential(nn.Linear(config.embedding_size,config.embedding_size),
									nn.Dropout(config.dropout_p),
									activation_fn,
									nn.Linear(config.embedding_size,config.embedding_size),
									)
	def forward(self,hidden_state):
		for layer in self.layers:
			hidden_state = layer(hidden_state)
		return hidden_state

class Separable_Conv(nn.Module):
	def __init__(self,config):
		super(Separable_Conv, self).__init__()
		self.activation_fn = gelu()#nn.GELU()
		kernel_size = 3
		if (kernel_size-1)%2 == 0:
			padding_size = (int((kernel_size-1)/2),int((kernel_size-1)/2))
		else:
			padding_size = (int((kernel_size-1)/2)+1,int((kernel_size-1)/2))

		self.conv1 = nn.Conv1d(in_channels=config.embedding_size, 
							   out_channels=config.embedding_size,
							   kernel_size=kernel_size)
		self.conv2 = nn.Conv1d(in_channels=config.embedding_size, 
							   out_channels=config.embedding_size,
							   kernel_size=kernel_size)
		self.pad = nn.ConstantPad1d(padding_size, 0)
		self.dropout = nn.Dropout(config.dropout_p)

	def forward(self,hidden_state):
		'''
		Args:
			hidden_state (B,S,D)
		'''
		hidden_state = hidden_state.permute(0,2,1) # (B,D,S)
		hidden_state = self.pad(hidden_state) # (B,D,S+2)
		hidden_state = self.conv1(hidden_state) # (B,D,S)
		hidden_state = self.dropout(hidden_state)
		hidden_state = self.activation_fn(hidden_state)
		hidden_state = self.pad(hidden_state) # (B,D,S+2)
		hidden_state = self.conv2(hidden_state) # (B,D,S)
		hidden_state = hidden_state.permute(0,2,1) # (B,S,D)
		return hidden_state



class Transition(nn.Module):
	def __init__(self,config):
		super(Transition, self).__init__()
		# the activation function is ReLU from the original Universal Transformer paper
		activation_fn = gelu()#nn.GELU()
		if config.transition_type == 'fc':
			self.layer = FeedForward(config)
		else: # 'conv'
			self.layer = Separable_Conv(config)
		self.residual_connect = Residual(config)

	def forward(self,hidden_state):
		outputs = self.layer(hidden_state)
		outputs = self.residual_connect(hidden_state,outputs)
		return outputs

class Attention(nn.Module):
	def __init__(self,config):
		super(Attention, self).__init__()
		self.attention_layer = MultiHeadAttention(config)
		self.residual_connect = Residual(config)
	def forward(self,hidden_state,attention_mask,encoder_state = None):
		outputs = self.attention_layer(hidden_state,attention_mask,encoder_state)
		outputs = self.residual_connect(hidden_state,outputs)
		return outputs

class ACT(nn.Module):
	def __init__(self,config,is_decoder = False):
		super(ACT, self).__init__()
		self.LayerNorm = nn.LayerNorm(config.embedding_size,eps = 1e-12)
		self.dropout = nn.Dropout(config.dropout_p)
		self.sigmoid = nn.Sigmoid()
		self.halting_unit = nn.Linear(config.embedding_size,1)
		self.act_epsilon = config.act_epsilon
		self.is_decoder = is_decoder
		self.self_attention = Attention(config)
		if self.is_decoder:
			self.encoder_decoder_layer = Attention(config)
		self.transition = Transition(config)
		self.act_type = config.act_type

	def forward(self, 
				state, 
				attention_mask, 
				time_emb, 
				max_step, 
				encoder_state = None,
				encoder_state_mask = None,
				step = 0):
		'''
		Args:
			state (B,S,D)
			attention_mask (S,S)
			time_emb (1,NEL+NDL,D)
			encoder_state (B,S_i,D)
			encoder_state_mask (S_i,S)
			max_step: max number of layers
			step: the beginning of the step

		Return:
			pre_states (B,S,D)
			num_updates (B,S)
			remainders (B,S)

		Reference from https://arxiv.org/pdf/1603.08983)
		'''
		device = state.device
		batch_size,seq_len,_ = state.size()
		halting_probs = torch.zeros((batch_size,seq_len),device = device)
		remainders = torch.zeros((batch_size,seq_len),device = device)
		num_updates = torch.zeros((batch_size,seq_len),device = device)
		pre_states = torch.zeros_like(state,device = device)

		x = state

		while (step < max_step) & (((halting_probs<(1-self.act_epsilon)).any()).item()) :
			x = x + time_emb[:,step,:].unsqueeze(0)
			x = self.LayerNorm(x)
			x = self.dropout(x)

			x = self.self_attention(x,attention_mask)
			if self.is_decoder:
				x = self.self_attention(x,encoder_state_mask,encoder_state)
			x = self.transition(x)

			# ACT
			not_halting = (halting_probs < 1-self.act_epsilon).float()
			halting_probs_cur = (self.sigmoid(self.halting_unit(x))).squeeze(-1) # (B,D)

			# only sum the probability which less than 1
			halting_probs_sum = halting_probs + (halting_probs_cur*not_halting) 

			# (Remainder) only sum to 1 which larger than the threshold
			remainders_cur = (1 - halting_probs)*((halting_probs_sum > (1-self.act_epsilon)).float())
			remainders = remainders + remainders_cur
			halting_probs = halting_probs + remainders_cur
			
			# (Halting) keep accumulating the probability which less than threshold
			halting_probs = halting_probs + (halting_probs_cur*((halting_probs_sum <= (1-self.act_epsilon))).float())
			update_weights = halting_probs_cur*((halting_probs <= 1-self.act_epsilon).float()) + remainders_cur
			num_updates = num_updates + ((halting_probs <= 1-self.act_epsilon).float()) + ((remainders_cur > 0).float())

			if self.act_type == 'accumulated':
				pre_states = pre_states + ((update_weights.unsqueeze(-1))*x)
			else:
				pre_states = ((1 - update_weights)*pre_states) + (update_weights*x)
		
			step += 1

		return (pre_states,num_updates,remainders,step)


class UniversalTransformer(nn.Module):
	def __init__(self,config):
		super(UniversalTransformer, self).__init__()
		self.embedding = UTEmbedding(config)
		self.act_encoder = ACT(config)
		self.act_decoder = ACT(config,is_decoder = True)
		self.pooler = nn.Linear(config.embedding_size,config.embedding_size)
		self.pooler_activation = nn.Tanh()
		self.LayerNorm = nn.LayerNorm(config.embedding_size)
		self.classify = nn.Linear(config.embedding_size,config.vocab_size)
		self.max_step_e = config.num_encoder_layers
		self.max_step_d = config.num_decoder_layers
		self.apply(self.init_weight)

	def init_weight(self,module):
		if isinstance(module, (nn.Linear, nn.Embedding)):
			module.weight.data.normal_(mean=0.0, std=0.02)
		elif isinstance(module, nn.LayerNorm):
			module.bias.data.zero_()
			module.weight.data.fill_(1.0)
		if isinstance(module, nn.Linear) and module.bias is not None:
			module.bias.data.zero_()
		# TO DO: conv weight and bias 

	def forward(self,input_ids,target_ids,input_mask,target_mask,encoder_decoder_mask):
		'''
		Args:
			input_ids (B,S_i)
			target_ids (B,S_t)
			input_mask (S_i,S_i)
			target_mask (S_t,S_t)
			encoder_decoder_mask (S_t,S_i)
		'''
		input_emb, target_emb , time_emb = self.embedding(input_ids,target_ids)
		encoder_outputs,num_updates_e,remainders_e,step = self.act_encoder(input_emb,
								   							   			   input_mask,
								   							   			   time_emb,
								   							   			   max_step = self.max_step_e)
		decoder_outputs,num_updates_d,remainders_d,_ = self.act_decoder(target_emb,
								   										target_mask,
								   										time_emb,
								   										max_step = step + self.max_step_d,
								   										encoder_state = encoder_outputs,
								   										encoder_state_mask = encoder_decoder_mask,
								   										step = step)
		
		active_pos_i = input_ids != 0
		active_pos_t = target_ids != 0

		num_updates_total = num_updates_e[active_pos_i].sum() + num_updates_d[active_pos_t].sum()
		remainders_total = remainders_e[active_pos_i].sum() + remainders_d[active_pos_t].sum()
		ponder_cost = num_updates_total + remainders_total

		# (B,S,D) --> (B,S,V)
		outputs = self.pooler(decoder_outputs)
		outputs = self.pooler_activation(outputs)
		outputs = self.LayerNorm(outputs)
		outputs = self.classify(outputs) 

		return outputs,ponder_cost


class UniversalTransformerDecoder(nn.Module):
	def __init__(self,config):
		super(UniversalTransformerDecoder, self).__init__()
		self.embedding = AlbertEmbedding(config)
		self.act_encoder = ACT(config)
		self.pooler = nn.Linear(config.embedding_size,config.embedding_size)
		self.pooler_activation = nn.Tanh()
		self.LayerNorm = nn.LayerNorm(config.embedding_size)
		self.classify = nn.Linear(config.embedding_size,config.vocab_size)
		self.max_step_e = config.num_encoder_layers

	def forward(self,input_ids,input_mask):
		'''
		Args:
			input_ids (B,S_i)
			input_mask (S_i,S_i)

		'''
		input_emb, time_emb = self.embedding(input_ids)
		encoder_outputs,num_updates_e,remainders_e,step = self.act_encoder(input_emb,
								   							   			   input_mask,
								   							   			   time_emb,
								   							   			   max_step = self.max_step_e)
		

		# (B,S,D) --> (B,S,V)
		outputs = self.pooler(encoder_outputs)
		outputs = self.pooler_activation(outputs)
		outputs = self.LayerNorm(outputs)
		outputs = self.classify(outputs) 

		return outputs,num_updates_e,remainders_e

