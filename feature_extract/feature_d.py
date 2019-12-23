"""
feature_d.两个语句的Jaccard相似度。
"""

import jieba
from tqdm import tqdm
# from feature_extract.cfg import *

feature_title = 'feature_d'
# filepath = FILE_PATH
# featurepath = './../feature/{}.txt'.format(feature_title)


def Jaccrad(model, reference):  # terms_reference为源句子，terms_model为候选句子
    terms_reference = jieba.cut(reference)  # 默认精准模式
    terms_model = jieba.cut(model)
    grams_reference = set(terms_reference)  # 去重；如果不需要就改为list
    grams_model = set(terms_model)
    temp = 0
    for i in grams_reference:
        if i in grams_model:
            temp = temp + 1
    fenmu = len(grams_model) + len(grams_reference) - temp  # 并集
    jaccard_coefficient = float(temp / fenmu)  # 交集
    return jaccard_coefficient


# if __name__ == '__main__':
def get_feature_d(filepath,save_dir):
    feature_d = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            str1 = line.split('\t')[0]
            str2 = line.split('\t')[1]

            jaccard_coefficient = Jaccrad(str1, str2)
            feature_d.append(jaccard_coefficient)

    featurepath = save_dir+'{}.txt'.format(filepath.split("/")[-1].split(".")[0] + "_" +feature_title)
    with open(featurepath, 'w', encoding='utf-8') as f:
        for i in tqdm(feature_d):
            f.write(str(i) + '\n')




