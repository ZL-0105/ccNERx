# %%
import random

with open('data/ccks/ccks+FN_FJ/FN_FJ_train.json', encoding='utf-8') as f:
    ori_list = f.read().split('\n')
    if ori_list[-1] == '':
        ori_list = ori_list[:-1]

random.shuffle(ori_list)


train = ori_list[:1379]
test = ori_list[1379:]
# dev = ori_list[1238:]
with open('data/ccks/ccks+FN_FJ/FN_FJ_train_400.json', encoding='utf-8', mode='a+') as f:
    for item in train:
        f.write(item + '\n')

with open('data/ccks/ccks+FN_FJ/2.json', encoding='utf-8', mode='a+') as f:
    for item in test:
        f.write(item + '\n')
        
# with open('data/ccks/dev.csv', encoding='utf-8', mode='a+') as f:
#     for item in dev:
#         f.write(item + '\n')
# %%
