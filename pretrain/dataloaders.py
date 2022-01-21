import torch
import random
from torch.utils.data import TensorDataset, DataLoader, Dataset

class NERDataset(Dataset):
    # 平均长度716
    def __init__(self, tokenizer, file_name, padding_length=512, shuffle=True):
        self.tokenizer = tokenizer
        self.padding_length = padding_length
        self.max_masked_num = int(padding_length * 0.1)
        self.masked_idx = self.tokenizer('[MASK]', add_special_tokens=False)['input_ids'][0]
        self.ori_list = self.load_pre_train(file_name)
        if shuffle:
            random.shuffle(self.ori_list)
        
        
    def load_pre_train(self, file_name):
        with open(file_name, encoding='utf-8') as f:
            ori_list = f.read().split('\n')
        if ori_list[-1] == '':
            ori_list = ori_list[:-1]
        return ori_list
        

    def __len__(self):
        return len(self.ori_list)


    def __getitem__(self, idx):
        line = self.ori_list[idx]
        line = line.strip().split('\t')
        s1, s2 = line[0],  line[1]

        T = self.tokenizer(s1,s2, add_special_tokens=True, max_length=self.padding_length, padding='max_length', truncation=True)
        sentence = T['input_ids']
        attn_mask = T['attention_mask']

        index_arr = [i for i in range(len(sentence))]
        index_arr = index_arr[1:]
        
        # 随机标识
        random.shuffle(index_arr)
        
        # mask err num
        index_arr = index_arr[:int(len(index_arr) * 0.15)]
        masked_arr = index_arr[:int(len(index_arr) * 0.8)]
        err_arr = index_arr[int(len(index_arr) * 0.8):int(len(index_arr) * 0.9)]
        
        # 把句子变成 tags
        tags = torch.tensor(sentence)

        # 除要预测的label之外其他设成-100
        for idx in index_arr:
            sentence[idx] = -100

        for idx in masked_arr:
            sentence[idx] = self.masked_idx

        for idx in err_arr:
            sentence[idx] = int(random.random() * 49800)

        attn_mask = torch.tensor(attn_mask)
        sentence = torch.tensor(sentence)
        
        # token_type_ids = torch.tensor(T['token_type_ids'])
        
        return sentence, attn_mask, tags

