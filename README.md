# Genenrate text using Universal Transformers Decoder

An implementation of [Universal Transformers](https://arxiv.org/abs/1807.03819) with pytorch.
In this project, I attempt to experiment the performance with small model on small dataset, so I using the universal transformer decoder to generate text.
In detail, I trained model from scratch by using amount 5.92 MB chinese corpus. 

# Requirements
- Pytorch
- TensorboardX

# Prepare data
#### Step 1: Collect data
Using web crawling to collect the data. Due to the copyright, I do not push the code and the data.
#### Step 2: Create dictionary of corpus
Generate character-based dictionary from corpus as the tokenizer:
 ```sh
$ python preprocess.py --data_root <data_root_folder> 
                        --output_file <output_file_name>
```

# Train & Inference
To run the experiment with default parameters:
```sh
$ python main.py
```
To inference the result by resume model:
```sh
$ python main_inf.py --input_sent <sent> 
```

# Result

```sh
$ python main_inf_v3.py --input_sent 哈利和妙麗牽著手走進禮堂
Generate ...
哈利和妙麗牽著手走進禮堂,朝禮堂門走去。
可是非常高興的聲音。哈利關上了門,馬份坐著貓頭鷹棚屋的門走到了廚房。門廳裡傳來一個很大的聲音,因為他們和麥教授一直擁擠滿到什麼地走進禮堂。門都低低地扭頭看著哈
```