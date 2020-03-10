import os
from collections import Counter
import re
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type = str, default = './data/corpus_v2')
    parser.add_argument('--output_file', type = str, default = './data/HPchars.txt')
   
    return parser.parse_args()

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

def load_corpus(data_root:str)->list:
    data_list = os.listdir(data_root)
    word_dict = []
    for path in data_list:
    	path = os.path.join(data_root,path)
    	with open(path,'r',encoding = 'utf-8') as f:
    		for line in f.readlines():
    			line = line.rstrip()
    			line = re.sub(' ','',line)
    			line = strQ2B(line)
    			chars = list(line)
    			word_dict += chars
    			
    		f.close()

    return word_dict

def write_file(output_file:str):
    dict_file = open(output_file,'w',encoding = 'utf-8')
    for char in sorted(list(set(word_dict))):
    	dict_file.write(char+'\n')


if __name__ == '__main__':
    args = parse_args()
    data_root = args.data_root
    output_file = args.output_file
    word_dict = load_corpus(data_root)
    write_file(output_file)
    


