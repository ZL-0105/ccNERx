# %%
import random

with open('data/CDD/conll/train.txt', encoding='utf-8') as f:
    ori_list = f.read().split('\n')
    if ori_list[-1] == '':
        ori_list = ori_list[:-1]

random.shuffle(ori_list)


train = ori_list[:500]
test = ori_list[500:]
# dev = ori_list[1238:]
with open('data/CDD/0.5k/conll/train.txt', encoding='utf-8', mode='a+') as f:
    for item in train:
        f.write(item + '\n')

with open('data/CDD/0.5k/conll/else.txt', encoding='utf-8', mode='a+') as f:
    for item in test:
        f.write(item + '\n')
        
# with open('data/ccks/dev.csv', encoding='utf-8', mode='a+') as f:
#     for item in dev:
#         f.write(item + '\n')

# %%
import random

with open('data/CDD/train.json', encoding='utf-8') as f:
    ori_list = f.read().split('\n')
    if ori_list[-1] == '':
        ori_list = ori_list[:-1]

random.shuffle(ori_list)


train = ori_list[:2000]
test = ori_list[2000:]
# dev = ori_list[1238:]
with open('data/CDD/2k/json/train.json', encoding='utf-8', mode='a+') as f:
    for item in train:
        f.write(item + '\n')

with open('data/CDD/2k/json/else.json', encoding='utf-8', mode='a+') as f:
    for item in test:
        f.write(item + '\n')
        
# with open('data/ccks/dev.csv', encoding='utf-8', mode='a+') as f:
#     for item in dev:
#         f.write(item + '\n')
# %%
