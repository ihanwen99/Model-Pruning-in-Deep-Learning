# from deepDream import *
# !
# !
# !
# !
# !
# 注意

from deepDream_conv7 import *
import os

'''
dream all 1000 labels, multiple copies of a label by changing learning rate and  gaussian kernel
'''

output_dir = '/home/hwliu/workspace/network/CAM/output/new_vgg19_cifar_conv7/'


def main():
    nItrs = [1000]
    # lrs = [0.08,0.1,0.12,0.14]
    lrs = [0.14]
    # sigmas = [0.4.png,0.42,0.44,0.46,0.48,0.5]
    sigmas = [0.4]
    image_dict = {}

    model = DeepDream()
    for sigma in sigmas:
        model.createGaussianFilter(sigma=sigma)  # create gaussian filter
        for label in range(0, 512):
            for lr in lrs:
                for nItr in nItrs:
                    outputImage = model.dream(label=label, nItr=nItr, lr=lr)  # dream
                    fileName = "dream_" + str(label) + "_" + str(nItr) + "_" + str(lr) + "_" + str(sigma) + ".png"
                    print(fileName)
                    image_dict[fileName] = outputImage
                    # print(params)
            # if (label % 8 == 7):  # clear out by saving the output images
            if (label % 8 == 7):
                for name, image in image_dict.items():
                    model.save(image, output_dir + name)  # save the images

                image_dict.clear()  # clear the dictionary


def predict(img):
    model = DeepDream()
    res = list(model.predict(img)[0])
    print(res)
    return np.argmax(res)


def label_filters(img, label):
    model = DeepDream()
    res = model.label_img(img, label)
    return res


def batch_label():
    files = os.listdir('/home/hwliu/workspace/network/CAM/output/vgg19_cifar10_fc3/')
    filter_pres = {}
    for f in files:
        filter_pres[f] = {}
        for label in range(10):
            filter_pres[f][label] = \
                label_filters('/home/hwliu/workspace/network/CAM/output/vgg19_cifar10_fc3/' + f, label)[0, label, 0, 0]
        filter_pres[f]['pre'] = max(filter_pres[f], key=lambda i: filter_pres[f][i])
    for k, v in filter_pres.items():
        print(k, v)

def test_label_filters_hanwen():
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from torchvision import datasets
    from torchvision import transforms
    train_dataset = datasets.CIFAR10('../data', train=True, transform=transforms.ToTensor(), download=False)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    model = DeepDream()
    for i, data in tqdm(enumerate(train_loader, 1)):
        img, label = data
        result = model.label_img_hanwen(img, label)
        print(result)
        print(result.shape)
        break
    return result

def test():
    model = DeepDream()


if __name__ == "__main__":
    main()
    # test()
    # test_label_filters_hanwen()



