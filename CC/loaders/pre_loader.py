from CC.loaders.utils import *
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch import tensor
from transformers import BertTokenizer
from tqdm import *
from typing import *
from ICCSupervised.ICCSupervised import IDataLoader
import json
import numpy as np
import random
from distutils.util import strtobool


class PreLoader(IDataLoader):
    def __init__(self, **args):
        KwargsParser(debug=True) \
            .add_argument("batch_size", int, defaultValue=4) \
            .add_argument("train_file", str) \
            .add_argument("tag_file", str) \
            .add_argument("bert_vocab_file", str) \
            .add_argument("output_eval", bool, defaultValue=True) \
            .add_argument("add_seq_vocab", bool, defaultValue=False) \
            .add_argument("max_seq_length", int, defaultValue=256) \
            .add_argument("default_tag", str, defaultValue="O") \
            .add_argument("use_test", bool, defaultValue=False) \
            .add_argument("do_shuffle", bool, defaultValue=False) \
            .add_argument("do_predict", bool, defaultValue=False) \
            .add_argument("task_name", str) \
            .add_argument("debug", bool, defaultValue=False) \
            .parse(self, **args)

        self.read_data_set()
        self.verify_data()
        self.process_data()


    def read_data_set(self):
        self.data_files = [self.train_file]
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_vocab_file)

    def verify_data(self):
        pass

    def process_data(self):
        if self.use_test:
            self.myData_test = PreBertDataSet(self.data_files[2], **vars(self))
            self.dataiter_test = DataLoader(
                self.myData_test, batch_size=self.test_batch_size)
        else:
            self.myData = PreBertDataSet(self.data_files[0], **vars(self))
            self.dataiter = DataLoader(self.myData, batch_size=self.batch_size)

    def __call__(self):
        if self.use_test:
            return {
                'test_set': self.myData_test,
                'test_iter': self.dataiter_test,
            }
        if self.output_eval:
            return {
                'train_set': self.myData,
                'train_iter': self.dataiter,
            }
        else:
            return {
                'train_set': self.myData,
                'train_iter': self.dataiter,
            }


class PreBertDataSet(Dataset):
    def __init__(self, dataset_file: str, **args):
        self.file = dataset_file
        for name in args.keys():
            setattr(self, name, args[name])
        self.__init_dataset()

    # 转换成 ids
    def convert_embedding(self, item):
            prompts = [] # 原文+prompt 被msak后
            prompt_masks = [] # 原文+prompt 对应的masks数组
            # prompt_tags = [] # 原文+prompt的tags
            prompt_origins = [] # 原文+prompt
            word = []
            labels = []
            exist_prompt = set()
            
            text = item
            text = text.strip().split('\t')
            s1, s2 = text[0],  text[1]
            s1 = s1[:self.max_seq_length-2]
            s1_lenght = len(s1)

            random_num = random.randint(1,s1_lenght-4)
            s1_pormpt = s1[random_num:random_num+4]
            max_len_text = len(s1) + 2
            
            
            

            # s2 prompt 读入s2
            s2_pro = s2.split(';')
            for sen in s2_pro:
                if len(sen+"是一个异常关键词;") + max_len_text > self.max_seq_length:
                    continue
                else:
                    prompt_origins += list(sen)+list("是一个异常关键词;")
                    # prompt_masks += list(0 for i in sen) + list(1 for i in "是一个") + list(0 for i in "异常关键词") + [0]
                    prompt_masks += list(1 for i in sen) + list(1 for i in "是一个") + list(0 for i in "异常关键词") + [0]
                    # prompts += list('[MASK]' for i in sen) + list("是一个") + list('[MASK]' for i in "异常关键词") + [";"]
                    prompts += list(sen) + list("是一个异常关键词") + [";"]
                    max_len_text += len(list(sen)+list("是一个异常关键词;"))
            
            # 随机加入一个负样本
            if len(s1_pormpt+"不是一个异常关键词;") + max_len_text <= self.max_seq_length:    
                prompt_origins += list(s1_pormpt)+list("不是一个异常关键词;")
                # prompt_masks += list(0 for i in s1_pormpt) + list(1 for i in "不是一个") + list(0 for i in "异常关键词") + [0]
                prompt_masks += list(1 for i in s1_pormpt) + list(1 for i in "不是一个") + list(0 for i in "异常关键词") + [0]
                # prompts += list('[MASK]' for i in s1_pormpt) + list("不是一个") + list('[MASK]' for i in "异常关键词") + [";"]
                # prompts += list(s1_pormpt) + list("不是一个") + list('[MASK]' for i in "异常关键词") + [";"]
                prompts += list(s1_pormpt) + list("不是一个异常关键词") + [";"]
                max_len_text += len(list(s1_pormpt)+list("不是一个异常关键词;"))

            
            # resolve input
            text = ["[CLS]"] + list(s1) + ["[SEP]"] + prompts
            origin_text = text[:]
            text_origin_length = len(s1) + 2

            input_ids = self.tokenizer.convert_tokens_to_ids(text)
            origin_text = self.tokenizer.convert_tokens_to_ids(origin_text)
            
            
            
            labels = [-100 for _ in range(text_origin_length)]
            for m, token_id in zip(prompt_masks, origin_text):
                if m == 0:
                    labels.append(token_id)
                else:
                    # labels.append(-100)
                    labels.append(token_id)

            np_input_ids = np.zeros(self.max_seq_length, dtype=np.int)
            np_input_ids[:len(input_ids)] = input_ids

            np_token_type_ids = np.ones(self.max_seq_length, dtype=np.int)
            np_token_type_ids[:text_origin_length] = 0

            np_attention_mask = np.ones(self.max_seq_length, dtype=np.int)
            np_attention_mask[text_origin_length:len(text)] = prompt_masks


            np_labels = np.zeros(self.max_seq_length, dtype=np.int)
            np_labels[:len(labels)] = labels
            np_labels[len(labels):] = -100



            return np_input_ids, np_token_type_ids, np_attention_mask, np_labels
        # TODO: predict

        # raise NotImplemented("do_predict not implement")

    def __init_dataset(self):
        line_total = FileUtil.count_lines(self.file)
        self.input_token_ids = []
        self.token_type_ids = []
        self.attention_mask = []
        self.input_labels = []

        for line in tqdm(FileUtil.line_iter(self.file), desc=f"load dataset from {self.file}", total=line_total):
            line = line.strip()
            # data: Dict[str, List[Any]] = json.loads(line)

            data = line

            input_token_ids, token_type_ids, attention_mask,  input_labels = self.convert_embedding(
                data)

            self.input_token_ids.append(input_token_ids)
            self.token_type_ids.append(token_type_ids)
            self.attention_mask.append(attention_mask)
            self.input_labels.append(input_labels)

        self.size = len(self.input_token_ids)
        self.input_token_ids = np.array(self.input_token_ids)
        self.token_type_ids = np.array(self.token_type_ids)
        self.attention_mask = np.array(self.attention_mask)
        self.input_labels = np.array(self.input_labels)

        self.indexes = [i for i in range(self.size)]
        if self.do_shuffle:
            random.shuffle(self.indexes)

    def __getitem__(self, idx):
        idx = self.indexes[idx]
        return {
            'input_ids': tensor(self.input_token_ids[idx]),
            'attention_mask': tensor(self.attention_mask[idx]),
            'token_type_ids': tensor(self.token_type_ids[idx]),
            'input_labels': tensor(self.input_labels[idx]),
        }

    def __len__(self):
        return self.size
