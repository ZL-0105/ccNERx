# %%
import json
from CC.predicter import NERPredict
from CC.trainer import NERTrainer
# os.environ['CUDA_VISIBLE_DEVICES']='1'
# %%
args = {
    'num_epochs': 30,
    'num_gpus': [0],
    'bert_config_file_name': './model/chinese_wwm_ext/bert_config.json',
    'pretrained_file_name': './model/chinese_wwm_ext/pytorch_model.bin',
    # 'pretrained_file_name': './save_pretrained/FN_Pretrained/Bert_95250/pytorch_model.bin',
    # 'pretrained_file_name': './save_pretrained/FN_Pretrained_3/Bert_95250/pytorch_model.bin',
    # 'pretrained_file_name': './save_pretrained/FN_Pretrained_4/Bert_95250/pytorch_model.bin',
    'hidden_dim': 300,
    # 'max_scan_num': 1000000,
    'train_file': './data/FN/fj+sc/train.csv',
    'eval_file': './data/FN/fj/dev.csv',
    'test_file': './data/FN/fj/test.csv',
    'bert_vocab_file': './model/chinese_wwm_ext/vocab.txt',
    'tag_file': './data/FN/tags_list.txt',
    'output_eval': True,
    'loader_name': 'cn_loader',
    # "word_embedding_file":"./data/tencent/word_embedding.txt",
    # "word_vocab_file":"./data/tencent/tencent_vocab.txt",
    "default_tag":"O",
    'batch_size': 4,
    'eval_batch_size': 64,
    'do_shuffle': True,
    # 'pattern': ',	O', 
    'model_name': 'Bert',
    'task_name': 'FN-sc+fj_test-fj'
}
# %%
trainer = NERTrainer(**args)

for i in trainer():
    a = i



# %%
args = {
    'num_epochs': 30,
    'num_gpus': [0],
    'bert_config_file_name': './model/chinese_wwm_ext/bert_config.json',
    'pretrained_file_name': './model/chinese_wwm_ext/pytorch_model.bin',
    # 'pretrained_file_name': './save_pretrained/FN_Pretrained/Bert_95250/pytorch_model.bin',
    # 'pretrained_file_name': './save_pretrained/FN_Pretrained_3/Bert_95250/pytorch_model.bin',
    # 'pretrained_file_name': './save_pretrained/FN_Pretrained_4/Bert_95250/pytorch_model.bin',
    'hidden_dim': 300,
    # 'max_scan_num': 1000000,
    'train_file': './data/FN/fj+sc/train.csv',
    'eval_file': './data/FN/sc/dev.csv',
    'test_file': './data/FN/sc/test.csv',
    'bert_vocab_file': './model/chinese_wwm_ext/vocab.txt',
    'tag_file': './data/FN/tags_list.txt',
    'output_eval': True,
    'loader_name': 'cn_loader',
    # "word_embedding_file":"./data/tencent/word_embedding.txt",
    # "word_vocab_file":"./data/tencent/tencent_vocab.txt",
    "default_tag":"O",
    'batch_size': 4,
    'eval_batch_size': 64,
    'do_shuffle': True,
    # 'pattern': ',	O', 
    'model_name': 'Bert',
    'task_name': 'FN-sc+fj_test-sc'
}
# %%
trainer = NERTrainer(**args)

for i in trainer():
    a = i






# %%
# predict
args = {
    'num_epochs': 30,
    'num_gpus': [0, 1, 2, 3],
    'bert_config_file_name': './model/chinese_wwm_ext/bert_config.json',
    'pretrained_file_name': 'data/bert/chinese_wwm_ext/pytorch_model.bin',
    'hidden_dim': 300,
    'max_seq_length': 150,
    'max_scan_num': 1000000,

    'train_file': './data/weibonew/train.json',
    'eval_file': './data/weibonew/dev.json',
    'test_file': './data/weibonew/test.json',
    'bert_vocab_file': './model/chinese_wwm_ext/vocab.txt',
    'tag_file': './data/weibonew/labels.txt',

    'output_eval': True,

    'loader_name': 'le_loader',
    
    "word_embedding_file":"./data/tencent/word_embedding.txt",
    "word_vocab_file":"./data/tencent/tencent_vocab.txt",
    "default_tag":"O",

    'batch_size': 8,
    'eval_batch_size': 64,

    'do_shuffle': True,

    'model_name': 'LEBert',
    'lstm_crf_model_file': './save_model/Super_LEBert/lstm_crf/lstm_crf_14910.pth',
    'task_name': 'FN-predict-FJ'
}

# %%
predict = NERPredict(**args)
predict(["坐落于福州的福州大学ACM研究生团队, 在帅气幽默的傅仰耿老师带领下, 正在紧张刺激的开发一套全新的神秘系统。"])




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
    'loader_name': 'lex_loader',
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
    'num_gpus': [0, 1, 2],
    'bert_config_file_name': './model/chinese_wwm_ext/bert_config.json',
    'pretrained_file_name': './model/chinese_wwm_ext/pytorch_model.bin',
    'max_seq_length': 512,
    # 'max_scan_num': 1000000,
    'train_file': './data/FN/pre_train.csv',
    # 'eval_file': './data/weibo_L/dev.json',
    # 'test_file': './data/weibo_L/test.json',
    'bert_vocab_file': './model/chinese_wwm_ext/vocab.txt',
    'tag_file': './data/weibo/pretrained_labels.txt',
    
    'loader_name': 'pre_loader',
    
    # "word_embedding_file": "./data/tencent/word_embedding.txt",
    # "word_vocab_file": "./data/tencent/tencent_vocab.txt",
    # "word_vocab_file_with_tag": "./data/tencent/tencent_vocab_with_tag.json",#这个文件没有

    "default_tag": "O",
    'batch_size': 16,
    'eval_batch_size': 64,
    'do_shuffle': True,
    'task_name': 'FN_Pretrained_5',
    "use_gpu": True,
    "debug": True,
    # "tag_rules": {
    #     "PER.NOM": "人的象征",
    #     "LOC.NAM": "地点",
    #     "PER.NAM": "人",
    #     "GPE.NAM": "政治实体",
    #     "ORG.NAM": "组织",
    #     "ORG.NOM": "组织的象征",
    #     "LOC.NOM": "地点的象征",
    #     "GPE.NOM": "政治实体的象征",
    #     "ORG": "组织",
    #     "LOC": "地点",
    #     "PER": "人",
    #     "Time": "时间",
    #     "Thing": "物品",
    #     "Metric": "测量单位",
    #     "Abstract": "抽象",
    #     "Physical": "身体部位",
    #     "Term": "学期",
    #     "company": "企业",
    #     "name": "名字",
    #     "game": "游戏",
    #     "movie": "电影",
    #     "position": "职位",
    #     "address": "地址",
    #     "government": "政府",
    #     "scene": "景点",
    #     "book": "书名"
    # }
}

pre_trainer = NERPreTrainer(**args)

for i in pre_trainer():
    a = i

# %%
# 1 文本实体、实体类型和标签都mask了
# 2 加上了负样例
# 3 文本实体不mask
# 4 文本都不mask 只mask标签
# 5 label没有-100