import jieba
from data_process_2_1 import read_data,save_data,synword_and_samepinyin_data
from tqdm import tqdm
from data_process_2 import data_inverse
import pandas as pd
import os

def stopwordList(stopword_path):
    """创建停用词表/其他词表也可以"""
    stopwords = [line.strip() for line in open(stopword_path, encoding='UTF-8').readlines()]
    return stopwords

import re
def re_replace(content,otherwords,replace_word=""):
    """
    正则匹配开头和结尾的礼貌用语，如，请问，谢谢了
    :param content:
    :param otherwords:
    :param replace_word:
    :return:
    """
    resorted_otherwords = sorted(otherwords,key=lambda i:len(i),reverse=True)
    p_1 = ""
    for word in resorted_otherwords:
        p_1 += word
        p_1 += "|"
    p_1 = p_1.strip("|")
    new_content = re.sub(p_1,replace_word,content)
    return new_content

# 去停用词/语气词
def seg_depart(content,stopwords,otherword):
    """
    去停用词/语气词
    :param content: 需要进行处理的单个句子
    :param stopwords:  停用词表
    :param otherword: 针对该数据集统计的一些礼貌用语，如，请问，谢谢了
    :return:
    """
    # 对content进行分词
    content = re_replace(content,otherword)
    content_seg = jieba.cut(content.strip())
    new_content = ""
    for word in content_seg:
        if word in stopwords:
            continue
        else:
            new_content += word
    return new_content

def spread_list(sample_list):
    """用于将嵌套list展开成一维的"""
    list1 = []
    for sample in sample_list:
        len1 = len(sample)
        for i in range(len1):
            list1.append(sample[i])
    return list1


def compare_the_same_q_in_train_and_dev(train_path,dev_path,stopword_path,otherword_path):
    """
    计算测试集和训练集中相同问题的数目
    :param train_path:  训练集路径
    :param dev_path: 测试集路径
    :param stopword_path: 停用词路径
    :param otherword_path: 其他要删掉的词，如语气词等，及请问，咨询一下......
    :return:
    """
    stopwords = stopwordList(stopword_path)
    otherword = stopwordList(otherword_path)
    # stopwords += otherword
    train_data, train_true_data, train_false_data = read_data(train_path)
    train_data = [[sample[0],sample[1]] for sample in train_data]
    train_sample = spread_list(train_data)  # 只获取文本部分内容，不获取标签部分内容
    train_question = []
    for i in tqdm(range(len(train_sample))):
        train_question.append(seg_depart(train_sample[i],stopwords,otherword))
    dev_data = read_data(dev_path,dev=True)
    dev_sample = spread_list(dev_data)
    dev_question = []
    for i in tqdm(range(len(dev_sample))):
        dev_question.append(seg_depart(dev_sample[i],stopwords,otherword))

    same_number = 0   # 记录测试集和训练集中相同问题的数目
    for question in tqdm(dev_question):
        if question in train_question:
            same_number += 1
        else:
            continue
    return same_number

def compare_same_sample_in_train_and_test(train_data,test_data,save_same_sample_path,stopwords,otherword):
    """
    统计训练集和测试集中相同的样本
    :param train_data:
    :param test_data:
    :param save_same_sample
    :return: 返回具有相同样本的词
    """
    train_data_inverse, train_1, dev_2, test_3 = data_inverse(train_data,pattern=False)
    all_train_data = train_data + train_1
    all_train_data_not_label = [[item[0],item[1]] for item in all_train_data]
    train_data_compare = []
    for sample in tqdm(all_train_data_not_label):
        q1 = seg_depart(sample[0],stopwords,otherword)
        q2 = seg_depart(sample[1],stopwords,otherword)
        train_data_compare.append([q1,q2])
    test_data_compare = []
    for sample in tqdm(test_data):
        q1 = seg_depart(sample[0], stopwords,otherword)
        q2 = seg_depart(sample[1], stopwords,otherword)
        test_data_compare.append([q1, q2])

    same_sample = []
    for sample in tqdm(test_data_compare):
        if sample in train_data_compare:
            train_index = train_data_compare.index(sample)
            test_index = test_data_compare.index(sample)
            same_sample.append([test_index,test_data[test_index][0],test_data[test_index][1],all_train_data[train_index][0],
                                all_train_data[train_index][1],all_train_data[train_index][2]])
    same_sample_df = pd.DataFrame(same_sample,columns=["qid","test_q1","test_q2","train_q1","train_q2","label"])
    same_sample_df.to_csv(save_same_sample_path,sep="\t",index=None)
    return same_sample

def remove_stopwords_sample():
    """
    去掉停用词和礼貌用语，如，请问，谢谢了，之后的数据集，（train,test）
    :return: 输出去掉停用词之后得训练集和测试集
    """
    train_path = "./train_set.json"
    dev_path = "./test_set.csv"
    stopword_path = "./stopwords.txt"
    otherword_path = "./otherwords.txt"
    save_data_dir = "./tongji/"
    if not os.path.exists(save_data_dir):
        os.mkdir(save_data_dir)
    remove_stopwords_train_path = save_data_dir + "remove_stopwords_train_set.txt"
    remove_stopwords_test_path = save_data_dir + "remove_stopwords_test_set.txt"
    stopwords = stopwordList(stopword_path)
    otherword = stopwordList(otherword_path)
    # stopwords += otherword

    train_data, train_true_data, train_false_data = read_data(train_path)
    dev_data = read_data(dev_path, dev=True)

    remove_stopwords_train = []
    for sample in tqdm(train_data):
        q1 = seg_depart(sample[0],stopwords,otherword)
        q2 = seg_depart(sample[1],stopwords,otherword)
        remove_stopwords_train.append([q1,q2,sample[2]])

    remove_stopwords_test = []
    for sample in tqdm(dev_data):
        q1 = seg_depart(sample[0], stopwords,otherword)
        q2 = seg_depart(sample[1], stopwords,otherword)
        remove_stopwords_test.append([q1, q2])
    if not os.path.exists(remove_stopwords_train_path) and not os.path.exists(remove_stopwords_test_path):
        save_data(remove_stopwords_train, remove_stopwords_train_path, columns_num=3)
    save_data(remove_stopwords_test,remove_stopwords_test_path,columns_num=2)
    return remove_stopwords_train, remove_stopwords_test


from functools import reduce
def A_B_and_B_C(save_data_dir):
    """
    A->C为测试集中的样本
    若训练集中存在如下样本
    A->B 且 B->C 则可推出A->C的标签,规则如下：
    if A->B=True and B->C=True, then A->C = True
    else: A->C = False
    :return: 筛选出的数据,包含标签
    """
    remove_stopwords_train, remove_stopwords_test = remove_stopwords_sample()
    new_samples = []
    count = 0
    new_samples_save_path = save_data_dir + "A_B_and_B_C_sample.csv"
    print("A_B_and_B_C  ")
    all_combine_samples = []
    for index, sample in tqdm(enumerate(remove_stopwords_test)):
        A = sample[0]
        C = sample[1]
        A_samples_in_train = []
        C_samples_in_train = []
        for train_sample in remove_stopwords_train:
            if A in train_sample:
                A_samples_in_train.append(train_sample)
            if C in train_sample:
                C_samples_in_train.append(train_sample)
        func= lambda x,y:x if y in x else x+[y]
        A_samples_in_train = reduce(func,[[],]+A_samples_in_train)
        C_samples_in_train = reduce(func,[[],]+C_samples_in_train)
        if len(A_samples_in_train) != 0 and len(C_samples_in_train) != 0:
            for sample_1 in A_samples_in_train:
                for sample_2 in C_samples_in_train:
                    combine_sample = sample_1[0:2] + sample_2[0:2]  # 将两个样本中得句子进行合并
                    combine_sample = list(set(combine_sample))  # 去掉相同得question
                    if combine_sample not in all_combine_samples:
                        if len(combine_sample) == 1:
                            all_combine_samples.append(combine_sample)
                            label = 1
                            new_samples.append([index,A,C,label])
                            count += 1
                            # print(count)
                        elif len(combine_sample) == 2:
                            all_combine_samples.append(combine_sample)
                            label = 1
                            new_samples.append([index, A, C, label])
                            count += 1
                            # print(count)
                        elif len(combine_sample) == 3:
                            all_combine_samples.append(combine_sample)
                            if sample_1[2] == 1 and sample_2[2] == 1:
                                label = 1
                                new_samples.append([index, A, C, label])
                                count += 1
                                # print(count)
                            elif sample_1[2] == 0 and sample_2[2] == 0:
                                continue
                            else:
                                label = 0
                                new_samples.append([index, A, C, label])
                                count += 1
                                # print(count)
                        else:
                            continue
        else:
            continue

    func = lambda x, y: x if y in x else x + [y]
    new_samples = reduce(func, [[], ] + new_samples)
    data_111 = []
    for i in range(0, len(new_samples)):
        if i == 0:
            data_111.append(new_samples[i])
        else:
            if new_samples[i][0] != data_111[len(data_111)-1][0]:
                data_111.append(new_samples[i])
            else:
                continue

    new_samples = data_111
    new_samples_df = pd.DataFrame(new_samples,columns=["qid","q_1","q_2","label"])
    new_samples_df.to_csv(new_samples_save_path,sep="\t",index=None)
    print("the number of A_B_and_B_C_sample is {}".format(count))
    return new_samples


def get_sample_according_A_B_and_B_C():
    save_data_dir = "./tongji/"
    A_B_and_B_C(save_data_dir)

def get_same_sample_in_train_and_test():
    """计算测试集和训练集中相同样本的数目"""
    train_path = "./train_set.json"
    dev_path = "./test_set.csv"
    stopword_path = "./stopwords.txt"
    otherword_path = "./otherwords.txt"
    save_data_dir = "./tongji/"
    if not os.path.exists(save_data_dir):
        os.mkdir(save_data_dir)
    save_same_sample_path = "./tongji/same_sample_in_train_and_test.csv"
    stopwords = stopwordList(stopword_path)
    otherword = stopwordList(otherword_path)
    # stopwords += otherword
    train_data, train_true_data, train_false_data = read_data(train_path)
    dev_data = read_data(dev_path, dev=True)
    same_sample = compare_same_sample_in_train_and_test(train_data, dev_data, save_same_sample_path, stopwords,otherword)


def get_same_q_in_train_and_test():
    """计算测试集和训练集中相同问题的数目"""
    train_path = "./train_set.json"
    dev_path = "./test_set.csv"
    stopword_path = "./stopwords.txt"
    otherword_path =  "./otherwords.txt"
    same_sampleNumber = compare_the_same_q_in_train_and_dev(train_path, dev_path, stopword_path,otherword_path)
    print("same_sampleNumber:{}".format(same_sampleNumber))

def the_average_length_of_question_in_train_dataset():
    """
    统计训练集中，每个问题的平均长度
    :return:
    """
    train_path = "./train_set.json"
    train_data, train_true_data, train_false_data = read_data(train_path)
    all_length = 0
    all_question = 0
    for sample in train_data:
        all_length += len(sample[0])
        all_length += len(sample[1])
        all_question += 2
    average_length_question = all_length/all_question
    print("the_average_length_of_question_in_train_dataset is {}".format(average_length_question))
    return average_length_question

if __name__ == "__main__":
    # get_same_q_in_train_and_test()
    # get_same_sample_in_train_and_test()
    # remove_stopwords_sample()
    # get_sample_according_A_B_and_B_C()

    # the_average_length_of_question_in_train_dataset()

    # otherword_path = "./otherwords.txt"
    # otherword = stopwordList(otherword_path)
    # content = "请问一下，能帮我下嘛？谢谢了"
    # print(re_replace(content,otherword))

