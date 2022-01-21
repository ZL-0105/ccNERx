# %%
from transformers import BertTokenizer, BertForMaskedLM, BertConfig
from dataloaders import *
from tqdm import tqdm

config = BertConfig.from_json_file('../model/chinese_wwm_ext/bert_config.json')
tokenizer = BertTokenizer.from_pretrained('../model/chinese_wwm_ext/vocab.txt')
model = BertForMaskedLM.from_pretrained('../model/chinese_wwm_ext/pytorch_model.bin',config = config)
# text = "Replace me by any text you'd like."
train_data = NERDataset(tokenizer=tokenizer, file_name="../data/FN/text.csv")
train_loader = DataLoader(train_data, batch_size=4)
train_iter = tqdm(train_loader)

for sentences, attn_masks, tags in train_iter:
    #  encoded_input = tokenizer(sentences, return_tensors='pt')
    encoded_input = sentences
    print(sentences)
output = model(**encoded_input)



# %%
