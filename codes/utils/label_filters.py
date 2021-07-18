import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from tqdm import tqdm
import os
from codes.deepDream import *
import json

train_dataset = datasets.CIFAR10('../data', train=True, transform=transforms.ToTensor(), download=False)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
# test_dataset = datasets.CIFAR10('./data', train=False, transform=transforms.ToTensor(), download=False)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 每一类样本取 100 个
sample_num = 100
category_num = 10

samples = {}
for i, data in tqdm(enumerate(train_loader, 1)):
    img, label = data
    label = int(label)
    if label not in samples:
        samples[label] = []
    if len(samples[label]) >= sample_num:
        continue
    samples[label].append(img)
    flag = 0
    for j in range(category_num):
        try:
            if len(samples[j]) >= sample_num:
                flag += 1
        except:
            break
    if flag == category_num:
        break

# for k, v in samples.items():
#     print(len(samples[k]))

# 注意这里直接提取 DeepDream - 需要注意那里面的结构
model = DeepDream()
filter_clusters = json.load(open('/home/hwliu/workspace/network/CAM/codes/utils/filter_means_hanwen.json'))
cluser_references = {}
cluster_filter_prefer = {}
for i in range(10): cluster_filter_prefer[str(i)] = []
filter_label = {}
idx = 0
for k, filters in filter_clusters.items():
    cluser_references[k] = {}
    for filter in filters:
        filter_preference = {}
        for label, sps in samples.items():
            y = 0
            for sample in sps:
                # y += torch.sum(abs(model.label_img_hanwen(sample, label)))
                y += torch.sum(abs(model.label_img_hanwen(sample, label)[filter]))
                # 原来的错误代码 y += abs(torch.sum(model.label_img(sample, label)[0][filter]))
            y /= 100
            filter_preference[label] = float(y)
        filter_preference['prefer'] = max(filter_preference, key=lambda i: filter_preference[i])
        filter_label[filter] = filter_preference
        cluser_references[k][filter] = filter_preference
        cluster_filter_prefer[str(filter_preference['prefer'])].append(filter)
        print('filter', idx, filter, filter_preference['prefer'])
        idx += 1

# d = open('cluster_hanwen_{}.json'.format(sample_num), 'w')
# try:
#     json.dump(cluser_references, d)
#     # { 9: [ {1:,0.9, ...}, {  }, {  } ], [], ...      }
# except:
#     for k, v in cluser_references.items():
#         d.write(k)
#         d.write('\t')
#         d.write(str(v))
#         d.write('\n')
# d.close()

d = open('cluster_filter_hanwen_{}.json'.format(sample_num), 'w')
json.dump(cluster_filter_prefer, d)
d.close()

# d = open('filter_hanwen_{}.json'.format(sample_num), 'w')
# try:
#     json.dump(filter_label, d)
#     # { 9: [ {1:,0.9, ...}, {  }, {  } ], [], ...      }
# except:
#     for k, v in filter_label.items():
#         d.write(k)
#         d.write('\t')
#         d.write(str(v))
#         d.write('\n')
# d.close()

#
# filter_num = 512
# filter_label = {}
# for filter in range(0, filter_num):
#     for label, sps in samples.items():
#         y = 0
#         for sample in sps:
#             y += torch.sum(abs(model.label_img_hanwen(sample, label)))
#             # 原来的错误代码 y += abs(torch.sum(model.label_img(sample, label)[0][filter]))
#         y /= 100
#         filter_preference[label] = float(y)
#     filter_preference['prefer'] = max(filter_preference, key=lambda i: filter_preference[i])
#     filter_label[filter] = filter_preference
#
# d = open('filter_hanwen.json', 'w')
# try:
#     json.dump(filter_label, d)
#     # { 9: [ {1:,0.9, ...}, {  }, {  } ], [], ...      }
# except:
#     for k, v in filter_label.items():
#         d.write(k)
#         d.write('\t')
#         d.write(str(v))
#         d.write('\n')
# d.close()
