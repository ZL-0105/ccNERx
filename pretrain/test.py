# %%
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig, BertForMaskedLM
import torch
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from dataloaders import *

def cuda(inputX):
    if type(inputX) == tuple:
        if torch.cuda.is_available():
            result = []
            for item in inputX:
                result.append(item.cuda())
            return result
        return inputX
    else:
        if torch.cuda.is_available():
            return inputX.cuda()
        return inputX

def save_model(model, epoch, dataset_name, model_name, save_offset=0):
    _dir = '../save_model/model/{}/{}'.format(dataset_name, model_name)
    if not os.path.isdir(_dir):
        os.makedirs(_dir)
    torch.save(model, '{}/epoch_{}.pth'.format(_dir, epoch + 1 + save_offset))

resume_path=False
num_epochs=200
lr=1e-4
gpu=[0, 1]
batch_size=4

config = BertConfig.from_json_file('../model/chinese_wwm_ext/bert_config.json')
tokenizer = BertTokenizer.from_pretrained('../model/chinese_wwm_ext/vocab.txt')
model = BertForMaskedLM.from_pretrained('../model/chinese_wwm_ext/pytorch_model.bin',config = config)

train_data = NERDataset(tokenizer=tokenizer, file_name="../data/FN/text.csv")
train_loader = DataLoader(train_data, batch_size)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.cuda()
model = torch.nn.DataParallel(model, device_ids=gpu).cuda()

if not resume_path == False:
    print('Accessing Resume PATH: {} ...\n'.format(resume_path))
    model_dict = torch.load(resume_path).module.state_dict()
    model.module.load_state_dict(model_dict)

model.to(device)

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.)

Epoch_loss = []
# output_file = open("./sentence.txt", "w", encoding="utf8")
for epoch in range(num_epochs):
    train_count = 0
    train_loss = 0
    # train_loss = []

    train_iter = tqdm(train_loader)
    
    for sentences, attn_masks, tags in train_iter:
        model.train()
        # res1 = sentences.cpu().numpy()
        # res2 = attn_masks.cpu().numpy()
        # res3 = tags.cpu().numpy()
        # print(res1)
        # np.savetxt("sentences.csv",res1)
        # np.savetxt("attn_masks.csv",res2)
        # np.savetxt("tags.csv",res3)
        # print(attn_masks)
        # print(tags)
        # sentences = open("./sentences.txt", "r", encoding="utf8")

        # sentences = cuda(sentences)
        # attn_masks = cuda(attn_masks)
        # tags = cuda(tags)
        sentences = cuda(sentences)
        attn_masks = cuda(attn_masks)
        tags = cuda(tags)
        
        # print("sentence",sentences.size())
        # print("attn_mask",attn_masks.size())
        # print("tags",tags.size())
        
        
        model.zero_grad()
        
        outputs = model(input_ids = sentences, attention_mask=attn_masks, labels =tags)
        loss, prediction_scores = outputs[:2]
        loss = loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # train_loss.append(loss.data.item()) #loss用数组来计算的
        train_loss += loss.data.item()
        train_count += 1

        train_iter.set_description('Train: {}/{}'.format(epoch + 1, num_epochs))
        train_iter.set_postfix(train_loss=train_loss / train_count, cur_pred_scores=prediction_scores.mean().data.item())
    Epoch_loss.append(np.mean(train_loss))

    if resume_path == False:
        save_model(model, epoch, 'FN', 'bert', 0)
    else:
        save_model(model, epoch, 'FN', 'bert', int(resume_path.split('/')[-1].split('_')[1].split('.')[0]))

# %%
