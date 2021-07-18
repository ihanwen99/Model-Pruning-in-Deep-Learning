from torchvision import datasets
from tqdm import tqdm
import os
from codes.deepDream import *
import json
from torch.utils.data import DataLoader

train_dataset = datasets.CIFAR10('../../data', train=True, transform=transforms.ToTensor(), download=False)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
# test_dataset = datasets.CIFAR10('./data', train=False, transform=transforms.ToTensor(), download=False)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
sample_num = 10

samples = {}
for i, data in tqdm(enumerate(train_loader, 1)):
    img, label = data
    newimg = np.zeros((32, 32, 3))
    for j in range(3):
        newimg[:, :, j] = img[0][j, :, :] * 255
    # print(newimg)
    img = Image.fromarray((np.array(newimg)).astype('uint8'), 'RGB')
    label = int(label)
    # img.save('sampleImgOutput/{}_{}.png'.format(label, i))
    img.save('/home/hwliu/workspace/network/CAM/sampleImgOutput/{}_{}.png'.format(label, i))

    if label not in samples:
        samples[label] = []
    if len(samples[label]) >= sample_num:
        continue
    samples[label].append(img)
    flag = 0
    for j in range(sample_num):
        try:
            if len(samples[j]) >= sample_num:
                flag += 1
        except:
            break
    if flag == sample_num:
        break
