# %%
import json
from CC.predicter import NERPredict
from CC.trainer import NERTrainer

# %%
args = {
    'num_epochs': 30,
    'num_gpus': [0, 1],
    'bert_config_file_name': './model/chinese_wwm_ext/bert_config.json',
    # 'pretrained_file_name': './model/chinese_wwm_ext/pytorch_model.bin',
    # 'pretrained_file_name': './save_pretrained/FN_Pretrained/Bert_95250/pytorch_model.bin',
    'pretrained_file_name': './save_pretrained/FN_Pretrained_4/Bert_95250/pytorch_model.bin',
    'hidden_dim': 300,
    # 'max_scan_num': 1000000,
    'train_file': './data/FN/train.csv',
    'eval_file': './data/FN/dev.csv',
    'test_file': './data/FN/test.csv',
    'bert_vocab_file': './model/chinese_wwm_ext/vocab.txt',
    'tag_file': './data/FN/tags_list.txt',
    'output_eval': True,
    'loader_name': 'cn_loader',
    # "word_embedding_file":"./data/tencent/word_embedding.txt",
    # "word_vocab_file":"./data/tencent/tencent_vocab.txt",
    "default_tag":"O",
    'batch_size': 64,
    'eval_batch_size': 64,
    'do_shuffle': True,
    'model_name': 'Bert',
    'task_name': 'FN-train-FJ_4'
}
# %%
trainer = NERTrainer(**args)

for i in trainer():
    a = i

# %%
# 我的prompt：
# 北京和福州都是一座城市，北京是一个地名，福州是一个地名

# 北京和福州都是一座城市，北京是一个地名
# 北京和福州都是一座城市，福州是一个地名

#%%
args = {
    'num_epochs': 30,
    'num_gpus': [0, 1, 2, 3],
    'bert_config_file_name': './model/chinese_wwm_ext/bert_config.json',
    'pretrained_file_name': './model/chinese_wwm_ext/pytorch_model.bin',
    # 'pretrained_file_name': './save_pretrained/FN_Pretrained/Bert_95250/pytorch_model.bin',
    # 'pretrained_file_name': './save_pretrained/FN_Pretrained_3/Bert_95250/pytorch_model.bin',
    # 'pretrained_file_name': './save_pretrained/FN_Pretrained_4/Bert_95250/pytorch_model.bin',
    # 'pretrained_file_name': './save_pretrained/FN_Pretrained_5/Bert_95250/pytorch_model.bin',
    'hidden_dim': 300,
    'max_seq_length': 512,
    'max_scan_num': 1000000,
    'train_file': './data/FN/fj-json/train.csv',
    'eval_file': './data/FN/fj-json/dev.csv',
    'test_file': './data/FN/fj-json/test.csv',
    'bert_vocab_file': './model/chinese_wwm_ext/vocab.txt',
    'tag_file': './data/FN/tags_list.txt',
    'loader_name': 'le_loader',
    'output_eval':True,
    "word_embedding_file":"./data/tencent/word_embedding.txt",
    "word_vocab_file":"./data/tencent/tencent_vocab.txt",
    # "word_vocab_file_with_tag": "./data/tencent/tencent_vocab_with_tag.json",
    "default_tag":"O",
    'batch_size': 32,
    'eval_batch_size': 64,
    'do_shuffle': True,
    "use_gpu": True,
    "debug": True,
    'model_name': 'LEBert',
    'task_name': 'FN-fj-LeBert'
}
trainer = NERTrainer(**args)

for i in trainer():
    a = i

# %%

# 预测Train.json
# weibo train.json
from CC.predicter import NERPredict
import json

# 使用了预训练模型
args = {
    'num_epochs': 30,
    'num_gpus': [0, 1, 2, 3],
    'bert_config_file_name': './model/chinese_wwm_ext/bert_config.json',
    'pretrained_file_name': './model/chinese_wwm_ext/pytorch_model.bin',
    'hidden_dim': 300,
    'max_seq_length': 150,
    'max_scan_num': 1000000,
    'train_file': 'data/ccks/train-json.csv',
    # 'eval_file': './data/ccks/dev.csv',
    # 'test_file': './data/ccks/test.csv',
    'eval_file': './data/FN/sc-json/dev.csv',
    'test_file': './data/FN/sc-json/test.csv',
    'bert_vocab_file': './model/chinese_wwm_ext/vocab.txt',
    'tag_file': 'data/ccks/ccks_tags_list.txt',
    'output_eval': True,
    'loader_name': 'le_loader',
    "word_embedding_file": "./data/tencent/word_embedding.txt",
    "word_vocab_file": "./data/tencent/tencent_vocab.txt",
    "default_tag": "O",
    'batch_size': 32,
    'eval_batch_size': 64,
    'do_shuffle': True,
    'model_name': 'LEBert',
    'task_name': 'ccks_predict_model'
}

args["lstm_crf_model_file"] = "save_model/ccks_predict_model/lstm_crf/lstm_crf_1320.pth"
args["bert_model_file"] = "save_model/ccks_predict_model/LEBert/LEBert_1320.pth"
predict = NERPredict(**args)

filename = "data/FN/sc-super/train.json"

batch_size = 40
index = 0
sentences = []

with open("data/FN/sc-super/train_super.json", "w", encoding="utf-8") as out:
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            text = data["text"]
            
            sentences.append(text)
            index += 1
            if index % batch_size == batch_size-1:
                for s, label in predict(sentences):
                    # print(index)
                    assert len(s[:args["max_seq_length"]-2])==len(label)
                    out.write(f"""{json.dumps({"text":s[:args["max_seq_length"]-2],"label":label},ensure_ascii=False)}\n""")
                sentences = []
                out.flush()
        if len(sentences)>0:
            for s, label in predict(sentences):
                assert len(s[:args["max_seq_length"]])==len(label)
                out.write(f"""{json.dumps({"text":s[:args["max_seq_length"]-2],"label":label},ensure_ascii=False)}\n""")
# %%
from tools.to_json import conll_to_json
conll_to_json('./data/ccks/train.csv', './data/ccks/train-json.csv', split_tag='\n\n')

# %%
conll_to_json('./data/FN/sc/train.csv', './data/FN/sc-json/train.csv', split_tag='\n\n')
conll_to_json('./data/FN/sc/test.csv', './data/FN/sc-json/test.csv', split_tag='\n\n')
conll_to_json('./data/FN/sc/dev.csv', './data/FN/sc-json/dev.csv', split_tag='\n\n')
# %%
conll_to_json('./data/FN/fj/train.csv', './data/FN/fj-json/train.csv', split_tag='\n\n')
conll_to_json('./data/FN/fj/test.csv', './data/FN/fj-json/test.csv', split_tag='\n\n')
conll_to_json('./data/FN/fj/dev.csv', './data/FN/fj-json/dev.csv', split_tag='\n\n')