import os, codecs
import cv2
import numpy as np
from sklearn.cluster import KMeans


def get_file_name(path):
    '''''
    Args: path to list;  Returns: path with filenames
    '''
    filenames = os.listdir(path)
    path_filenames = []
    filename_list = []
    for file in filenames:
        if not file.startswith('.'):
            path_filenames.append(os.path.join(path, file))
            filename_list.append(file)

    return path_filenames


def knn_detect(file_list, cluster_nums, randomState=None):
    feature = []
    files = file_list
    # img = cv2.imread(files[0])
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # sift = cv2.xfeatures2d.SIFT_create()
    # kp = sift.detect(gray, None)
    # im = img.copy()
    # 对关键点进行绘图
    # ret = cv2.drawKeypoints(gray, kp, im)
    # cv2.imwrite('1.png', img)
    # cv2.imwrite('2.png', gray)
    # cv2.imwrite('3.png', im)

    for file in files:
        img = cv2.imread(file)
        img = img.flatten() # 150528
        feature.append(img)
    #     print(file)
    #     img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_CUBIC)
    #
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     print(gray.dtype)
    #     kp,des = sift.detectAndCompute(gray, None)  # 检测并计算描述符
    #     # Kp,des=sift.detectAndCompute(gray,None)#检测并计算描述符
    #     # des =sift.detect(gray, None)# sift.detectAndCompute(gray, None)
    #     # # # 找到后可以计算关键点的描述符
    #     # Kp, des = sift.compute(gray, des)
    #     if des is None:
    #         file_list.remove(file)
    #         continue
    #
    #     reshape_feature = des.reshape(-1, 1)
    #     features.append(reshape_feature[0].tolist())

    input_x = np.array(feature)
    print(input_x.shape)

    kmeans = KMeans(n_clusters=cluster_nums, random_state=randomState).fit(input_x)

    return kmeans.labels_, kmeans.cluster_centers_
    # return 0,0


def res_fit(filenames, labels):
    files = [file.split('/')[-1] for file in filenames]

    return dict(zip(files, labels))


def save(path, filename, data):
    file = os.path.join(path, filename)
    with codecs.open(file, 'w', encoding='utf-8') as fw:
        for f, l in data.items():
            fw.write("{}\t{}\n".format(f, l))


def main():
    path_filenames = get_file_name("../../output/new_vgg19_cifar_conv7")
    labels, cluster_centers = knn_detect(path_filenames, 10)

    res_dict = res_fit(path_filenames, labels)
    save('./', 'knn_res_hanwen_7.txt', res_dict)


if __name__ == "__main__":
    main()
