"""
feature_b.两个语句的编辑距离.
"""

import numpy as np
from tqdm import tqdm
# from feature_extract.cfg import *

feature_title = 'feature_b'
# filepath = FILE_PATH
featurepath = './../feature/{}.txt'.format(feature_title)


def string_distance(str1, str2):
    len1 = str1.__len__()
    len2 = str2.__len__()
    distance = np.zeros((len1 + 1, len2 + 1))

    for i in range(0, len1 + 1):
        distance[i, 0] = i
    for i in range(0, len2 + 1):
        distance[0, i] = i

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if str1[i - 1] == str2[j - 1]:
                cost = 0
            else:
                cost = 1
            distance[i, j] = min(distance[i - 1, j] + 1, distance[i, j - 1] + 1,
                                 distance[i - 1, j - 1] + cost)  # 分别对应删除、插入和替换

    return int(distance[len1, len2])


# if __name__ == '__main__':
def get_feature_b(filepath,save_dir):
    feature_b = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            str1 = line.split('\t')[0]
            str2 = line.split('\t')[1]

            result = string_distance(str1, str2)
            feature_b.append(result)

    featurepath = save_dir+'{}.txt'.format(filepath.split("/")[-1].split(".")[0] + "_" + feature_title)
    with open(featurepath, 'w', encoding='utf-8') as f:
        for i in tqdm(feature_b):
            f.write(str(i) + '\n')


