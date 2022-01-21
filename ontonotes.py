# %%
from CC.loaders import *
import pickle
from tqdm import *
from CC.loaders.utils import *
import json
from CC.pre_trained import NERPreTrainer

# %%
args = {
    'num_epochs': 30,
    'num_gpus': [0, 1, 2, 3],
    'bert_config_file_name': './model/chinese_wwm_ext/bert_config.json',
    'pretrained_file_name': './model/chinese_wwm_ext/pytorch_model.bin',
    'max_seq_length': 512,
    'max_scan_num': 1500000,
    'train_file': './data/SuperNER/pre_train.json',
    'eval_file': './data/SuperNER/pre_dev.json',
    'test_file': './data/SuperNER/pre_test.json',
    'bert_vocab_file': './model/chinese_wwm_ext/vocab.txt',
    'tag_file': './data/SuperNER/tags_list.txt',
    'loader_name': 'le_loader',
    "word_embedding_file": "./data/tencent/word_embedding.txt",
    "word_vocab_file": "./data/tencent/tencent_vocab.txt",
    "word_vocab_file_with_tag": "./data/tencent/tencent_vocab_with_tag.json",
    "default_tag": "O",
    'batch_size': 32,
    'eval_batch_size': 64,
    'do_shuffle': True,
    'task_name': 'Pre_trained',
    "use_gpu": True,
    "debug": True,
    "tag_rules": {
        "ORG": "组织",
        "LOC": "地点",
        "PER": "人",
        "Time": "时间",
        "Thing": "物品",
        "Metric": "测量单位",
        "Abstract": "抽象",
        "Physical": "身体部位",
        "Term": "学期",
        "company": "企业",
        "name": "名字",
        "game": "游戏",
        "movie": "电影",
        "position": "职位",
        "address": "地址",
        "government": "政府",
        "scene": "景点",
        "book": "书名",
    }
}

pre_trainer = NERPreTrainer(**args)

for i in pre_trainer():
    a = i