"""
feature_a.两个语句的长度上的差距.
"""
from tqdm import tqdm
# from feature_extract.cfg import *

feature_title = 'feature_a'
# filepath = FILE_PATH
featurepath = './../feature/{}.txt'.format(feature_title)

# if __nmae__ == "__main__":
def get_feature_a(filepath,save_dir):
    feature_a = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            str1 = line.split('\t')[0]
            str2 = line.split('\t')[1]

            l1 = len(str1)
            l2 = len(str2)

            feature_a.append(abs(l1 - l2))

    featurepath = save_dir+'{}.txt'.format(filepath.split("/")[-1].split(".")[0] + "_" +feature_title)
    with open(featurepath, 'w', encoding='utf-8') as f:
        for i in tqdm(feature_a):
            f.write(str(i) + '\n')
