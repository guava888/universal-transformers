from utils import parse_args,config_setting,FutureMask
from model.model import UniversalTransformerDecoder
import torch
import torch.nn as nn
import time
from data.utils import HPchar2int_v2
from torch.distributions.categorical import Categorical
import re

# argument setting
args = parse_args()
model_config = config_setting(args)

# model 
model = UniversalTransformerDecoder(model_config)

# resume model
checkpoint = torch.load(args.resume_path)
model.load_state_dict(checkpoint['state_dict'])

# device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = model.to(device)

# parameter
temperature = args.temperature
softmax = nn.Softmax(0)
char2int = HPchar2int_v2()
int2char = {v:k for k,v in char2int.items()}

# inference
input_sent = '哈利走進霍格華茲'
input_ids = list(map(char2int.get,list(input_sent)))


model.eval()
with torch.no_grad():
	while len(input_ids) < args.max_len:
		
		input_tensor = torch.LongTensor(input_ids).unsqueeze(0).to(device)
		masks_tensor = FutureMask(input_tensor).to(device)

		outputs,_ = model(input_ids=input_tensor,
						  input_mask=masks_tensor)

		outputs = outputs[0,-1,:]/temperature
		outputs = softmax(outputs)
		sampler = Categorical(outputs)
		input_ids.append(sampler.sample().cpu().item())


input_ids = list(map(int2char.get,input_ids))
target_chars = ''.join(input_ids)
target_chars = re.sub('\[NL\]','\n',target_chars)

print(target_chars)