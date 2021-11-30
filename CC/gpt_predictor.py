import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from ICCSupervised.ICCSupervised import IPredict
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, get_linear_schedule_with_warmup
from tqdm import tqdm
from torch.autograd import Variable


class Predictor(IPredict):
    def __init__(self,
                 tokenizer,
                 model_dir,
                 padding_length=128,
                 resume_path=False,
                 gpu=[0],
                 repeat_punish=1.0,
                 top_k=8,
                 top_p=0.5):
        self.tokenizer = tokenizer
        self.padding_length = padding_length
        self.model_init(model_dir)
        self.repeat_punish = repeat_punish
        self.top_k = top_k
        self.top_p = top_p

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        if not resume_path == False:
            print('Accessing Resume PATH: {} ...\n'.format(resume_path))
            model_dict = torch.load(resume_path).module.state_dict()
            self.model.load_state_dict(model_dict)

        if torch.cuda.is_available() and self.model_cuda == False:
            self.model = torch.nn.DataParallel(self.model,
                                               device_ids=gpu).cuda()
            self.model_cuda = True
            self.model.to(device)

        self.model.to(device)

    def model_init(self, model_dir):
        self.model = GPT2LMHeadModel.from_pretrained(model_dir)
        self.model_cuda = False

    def data_process(self, *X):
        T = self.tokenizer(*X, add_special_tokens=True, max_length=self.padding_length, truncation=True)
        input_ids = torch.tensor(T['input_ids'])
        attn_mask = torch.tensor(T['attention_mask'])

        return {'input_ids': input_ids, 'attention_mask': attn_mask}

    def top_k_top_p_filtering(self,
                              logits,
                              top_k=0,
                              top_p=0.0,
                              filter_value=-float("Inf")):
        assert logits.dim() == 2
        top_k = min(top_k, logits[0].size(-1))  # Safety check
        if top_k > 0:
            # Remove all tokens with a probability less than the last token of the top-k
            # torch.topk返回值和索引
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1,
                                                                      None]
            logits[indices_to_remove] = filter_value

        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            # 按维度里元素顺序累加并按顺序输出累加值
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1),
                                            dim=-1)

            # Remove tokens with cumulative probability above the threshold
            # 计算出累加概率大于p的索引
            sorted_indices_to_remove = cumulative_probs > self.cuda(
                torch.tensor([[top_p]] * logits.shape[0]))
            # Shift the indices to the right to keep also the first token above the threshold
            # 右移索引确保第一个token不会被1到
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                ..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            # 这种操作会筛选出sorted_indices_to_remove为1的值
            for i, indices in enumerate(sorted_indices_to_remove):
                indices_to_remove = sorted_indices[i][indices]
                logits[i][indices_to_remove] = filter_value
        return logits

    # def top_k_top_p_filtering(self, logits, top_k=0, top_p=0.0, filter_value=-float("Inf")):
    #     """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    #     Args:
    #         logits: logits distribution shape (vocabulary size)
    #         top_k > 0: keep only top k tokens with highest probability (top-k filtering).
    #         top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
    #             Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    #     From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    #     """
    #     assert (
    #         logits.dim() == 1
    #     )  # batch size 1 for now - could be updated for more but the code would be less clear
    #     top_k = min(top_k, logits.size(-1))  # Safety check
    #     if top_k > 0:
    #         # Remove all tokens with a probability less than the last token of the top-k
    #         # torch.topk返回值和索引
    #         indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    #         logits[indices_to_remove] = filter_value

    #     if top_p > 0.0:
    #         sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    #         # 按维度里元素顺序累加并按顺序输出累加值
    #         cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    #         # Remove tokens with cumulative probability above the threshold
    #         # 计算出累加概率大于p的索引
    #         sorted_indices_to_remove = cumulative_probs > top_p
    #         # Shift the indices to the right to keep also the first token above the threshold
    #         # 右移索引确保第一个token不会被1到
    #         sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    #         sorted_indices_to_remove[..., 0] = 0

    #         # 这种操作会筛选出sorted_indices_to_remove为1的值
    #         indices_to_remove = sorted_indices[sorted_indices_to_remove]
    #         logits[indices_to_remove] = filter_value
    #     return logits

    def __call__(self, *X):
        return self.predict(*X)

    def predict(self, *X):
        it = self.data_process(*X)
        for key in it.keys():
            it[key] = self.cuda(it[key].unsqueeze(0))
        with torch.no_grad():
            r = self.model(**it)
            pred = r.logits
            text = pred.max(-1)[1].tolist()

        return pred.cpu(), text, it['input_ids'].tolist()

    def predict_continous(self, *X):
        it = self.data_process(*X, self.tokenizer.sep_token)
        for key in it.keys():
            it[key] = self.cuda(it[key].unsqueeze(0))
        with torch.no_grad():
            while it['input_ids'].shape[1] < 300:
                r = self.model(**it)
                pred = r.logits
                next_token_logits = pred[0][-1]
                for id in it['input_ids'][0]:
                    next_token_logits[id] /= self.repeat_punish
                filtered_logits = self.top_k_top_p_filtering(
                    next_token_logits.unsqueeze(0),
                    top_k=self.top_k,
                    top_p=self.top_p)[0]
                next_token = torch.multinomial(torch.softmax(filtered_logits,
                                                             dim=-1),
                                               num_samples=1)
                it['input_ids'] = torch.cat(
                    (it['input_ids'], next_token.unsqueeze(0)), dim=1)
                it['attention_mask'] = torch.cat(
                    (it['attention_mask'], next_token.gt(0).unsqueeze(0)),
                    dim=1)
                yield pred.cpu(), it['input_ids'].tolist()

    def predict_continous_x(self, *X):
        it = self.tokenizer(X,
                            padding="max_length",
                            add_special_tokens=True,
                            max_length=self.padding_length,
                            truncation=True,
                            return_tensors="pt")
        with torch.no_grad():
            while True:
                r = self.model(**it)
                indexes = torch.sum(it["attention_mask"], dim=-1)
                select_logits = []
                exit_loop = True
                # TODO: torch 是否有内置函数
                for i, idx in enumerate(indexes):
                    # idx out of range?
                    if idx < it["input_ids"][0].shape[0]:
                        select_logits.append(r.logits[i][idx])
                        exit_loop = False
                    else:
                        select_logits.append(r.logits[i][idx - 1])
                if exit_loop:
                    break
                logits = torch.stack(select_logits)
                # 施加重复惩罚
                # TODO: 如何使用torch转换
                for i, input_ids in enumerate(it["input_ids"]):
                    for id in input_ids[:indexes[i]]:
                        logits[i][id] /= self.repeat_punish
                filtered_logits = self.top_k_top_p_filtering(logits, 8, 0.5)

                next_token = torch.multinomial(torch.softmax(filtered_logits,
                                                            dim=-1),
                                            num_samples=1)
                for i, idx in enumerate(indexes):
                    if idx < it["input_ids"][0].shape[0]:
                        it["input_ids"][i][idx] = next_token[i]
                        it["attention_mask"][i][idx] = 1
                yield r.logits.cpu(), it["input_ids"].tolist()

    def predict_ccner_x(self, input_str=None, input_dict=None):
        '''
        Only for CCNERX
        :param input_str: can be a str or a list of str.
        :param input_dict: the tokenized dict.
        '''
        if input_str is not None:
            it = self.tokenizer(input_str,
                                padding="max_length",
                                add_special_tokens=True,
                                max_length=self.padding_length,
                                truncation=True,
                                return_tensors="pt")
        else:
            it = input_dict
        for key in it.keys():
            it[key] = self.cuda(it[key])
        max_length = self.padding_length * 2
        with torch.no_grad():
            while it['input_ids'].shape[1] < max_length:
                r = self.model(**it)
                logits = r.logits[:, -1]
                filtered_logits = self.top_k_top_p_filtering(logits, 8, 0.5)

                next_tokens = torch.multinomial(torch.softmax(filtered_logits,
                                                            dim=-1),
                                            num_samples=1)
                next_tokens.cuda()
                it['input_ids'] = torch.cat(
                    (it['input_ids'], next_tokens), dim=1)
                if 'attention_mask' in it.keys():
                    it['attention_mask'] = torch.cat(
                        (it['attention_mask'], next_tokens.gt(self.tokenizer.pad_token_id)),
                        dim=1)
                if 'token_type_ids' in it.keys():
                    it['token_type_ids'] = torch.cat(
                        (it['token_type_ids'], (next_tokens != -999).long()),
                        dim=1)
                yield r.logits.cpu(), it["input_ids"].tolist()

    def predict_directly(self, input_str=None, input_dict=None):
        '''
        Only for CCNERX
        :param input_str: can be a str or a list of str.
        :param input_dict: the tokenized dict.
        '''
        if input_str is not None:
            it = self.tokenizer(input_str,
                                padding="max_length",
                                add_special_tokens=True,
                                max_length=self.padding_length,
                                truncation=True,
                                return_tensors="pt")
        else:
            it = input_dict
        for key in it.keys():
            it[key] = self.cuda(it[key])
        with torch.no_grad():
            r = self.model(**it)
            logits = r.logits
            pred_scores = torch.softmax(logits, dim=-1)
            pred_ids = pred_scores.max(-1)[1]
        return logits.cpu(), pred_ids.tolist()

    def cuda(self, inputX):
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