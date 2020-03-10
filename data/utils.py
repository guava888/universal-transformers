import os
import re
import random


def strQ2B(ustring):
    ss = []
    for s in ustring:
        rstring = ""
        for uchar in s:
            inside_code = ord(uchar)
            if inside_code == 12288:
                inside_code = 32
            elif (inside_code >= 65281 and inside_code <= 65374):
                inside_code -= 65248
            rstring += chr(inside_code)
        ss.append(rstring)
    return ''.join(ss)


def HPchar2int(char_dict)->dict:
	'''
	In HPchar2int_v2, adding a token to represent as new line
	'''
	char2int = {}
	with open(char_dict,'r',encoding = 'utf-8') as f:
		for i,line in enumerate(f.readlines()):
			char2int[line.rstrip()] = i + 1
	char2int['[UNK]'] = len(char2int)
	char2int['[PAD]'] = 0
	char2int['[CLS]'] = len(char2int)
	char2int['[NL]'] = len(char2int)
	return char2int

	
def load_corpus(data_root,char_dict):
	char2int = HPchar2int(char_dict)
	sents = []
	data_list = os.listdir(data_root)
	temp = []
	for file in data_list:
		path = os.path.join(data_root,file)
		s = 0
		with open(path,'r',encoding = 'utf-8') as f:
			for line in f.readlines():
				if s == 0:
					s += 1
					continue
				line = line.rstrip()
				line = re.sub(' ','',line)
				line = strQ2B(line)	
				sents.append(line)
	
	results = []
	temp = []
	while len(sents)>0:
		sent = sents[0]
		if len(sent) > 500:
			temp = [list(map(char2int.get,x))+[char2int['[NL]']] for x in temp]
			temp = sum(temp,[])
			results.append(temp)
			temp = []

			sent = list(map(char2int.get,sent)) + [char2int['[NL]']]
			results.append(sent)

			sents = sents[1:]

		temp_len = len(''.join(temp)) + (len(temp)) if temp else 0
		if (temp_len + len(sent)) < 500:
			temp.append(sent)
			rand_number = random.random()
			if rand_number > 0.6:
				temp = [list(map(char2int.get,x))+[char2int['[NL]']] for x in temp]
				temp = sum(temp,[])
				results.append(temp)
				temp = []
			sents = sents[1:]
		else:
			temp = [list(map(char2int.get,x))+[char2int['[NL]']] for x in temp]
			temp = sum(temp,[])
			results.append(temp)
			temp = []

	return results
