"""
feature_c.两个语句的n-gram相似性的特征.
"""

from tqdm import tqdm
import jieba
from nltk import ngrams
# from feature_extract.cfg import *

feature_title = 'feature_c'
# filepath = FILE_PATH
featurepath = './../feature/{}.txt'.format(feature_title)


def Ngram_distance(str1, str2, n=2):
    li1 = list(jieba.cut(str1))
    li2 = list(jieba.cut(str2))

    set1 = set(ngrams(li1, n))
    set2 = set(ngrams(li2, n))
    setx = set1 & set2

    len1 = len(set1)
    len2 = len(set2)
    lenx = len(setx)

    ngram_dist = len1 + len2 - 2 * lenx
    try:
        ngram_sim = 1 - ngram_dist / (len1 + len2)
    except ZeroDivisionError as ez:
        ngram_sim = 0

    return ngram_sim


# if __name__ == '__main__':
def get_feature_c(filepath,save_dir):
    feature_c = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            str1 = line.split('\t')[0]
            str2 = line.split('\t')[1]

            feature_c.append({"Unigram": Ngram_distance(str1, str2, 1),
                              "Bigram": Ngram_distance(str1, str2, 2),
                              "Trigram": Ngram_distance(str1, str2, 3)})

    featurepath = save_dir+'{}.txt'.format(filepath.split("/")[-1].split(".")[0] + "_" + feature_title)
    with open(featurepath, 'w', encoding='utf-8') as f:
        for i in tqdm(feature_c):
            f.write(str(i) + '\n')
