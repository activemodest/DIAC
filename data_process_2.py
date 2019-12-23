import json
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm   # 进度条工具包
from pyhanlp import HanLP   # 调入自然语言处理工具包
import pandas as pd
from data_process_2_1 import read_data,save_data,synword_and_samepinyin_data
import os
import jieba

def data_inverse(data,dev=False,pattern=True,mode=1,slicing_portion=[0.3,0.33]):
    """
    对样本中的两个问题的顺序，进行翻转,即[q1,q2,label]->[q2,q1,label]
    :param data: 需要进行翻转的数据集
    :param dev: 判断是否是dev数据集，dev数据集不包含标签
    :param pattern: 判断是否是带有pattern的数据
    :param mode:1:表示采取使用通配符替换关键词的pattern;
                2:表示分别留下不相同的词和相同的词
    :param slicing_portion: 切分成train：dev:test的比例
    :return: 返回对样本进行翻转后的数据集
    """
    if pattern:
        if mode == 1:
            if dev:
                data_df = pd.DataFrame(data, columns=["q11", "q21", "q12", "q22"])
            else:
                data_df = pd.DataFrame(data, columns=["q11", "q21","q12","q22", "label"])
            # print("data_df.iloc[0:2]",data_df.iloc[0:2])
            cols = list(data_df)
            cols.insert(0, cols.pop(cols.index("q21")))
            cols.insert(2, cols.pop(cols.index("q22")))
            data_df_inverse = data_df.loc[:, cols]
            if dev:
                data_df_inverse.columns = ["q11", "q21", "q12", "q22"]
            else:
                data_df_inverse.columns = ["q11", "q21","q12","q22", "label"]
            # print("data_df_inverse.iloc[0:2]", data_df_inverse.iloc[0:2])
            if dev:
                final_data = pd.concat([data_df, data_df_inverse], axis=0, ignore_index=True)
                final_data = np.array(final_data).tolist()  # DataFrame->list
                return final_data
            else:
                final_data = pd.concat([data_df,data_df_inverse], axis=0,ignore_index=True)
                final_data = np.array(final_data).tolist()  # DataFrame->list
                train, test_and_dev = train_test_split(final_data, test_size=slicing_portion[0])  # 将数据集划分为train,test,dev
                test, dev = train_test_split(test_and_dev, test_size=slicing_portion[0])
        else:
            if dev:
                data_df = pd.DataFrame(data, columns=["q11", "q21", "q12", "q22","q31"])
            else:
                data_df = pd.DataFrame(data, columns=["q11", "q21", "q12", "q22","q31", "label"])
            # print("data_df.iloc[0:2]",data_df.iloc[0:2])
            cols = list(data_df)
            cols.insert(0, cols.pop(cols.index("q21")))
            cols.insert(2, cols.pop(cols.index("q22")))
            data_df_inverse = data_df.loc[:, cols]
            if dev:
                data_df_inverse.columns = ["q11", "q21", "q12", "q22","q31"]
            else:
                data_df_inverse.columns = ["q11", "q21", "q12", "q22","q31", "label"]
            # print("data_df_inverse.iloc[0:2]", data_df_inverse.iloc[0:2])
            if dev:
                final_data = pd.concat([data_df, data_df_inverse], axis=0, ignore_index=True)
                final_data = np.array(final_data).tolist()  # DataFrame->list
                return final_data
            else:
                final_data = pd.concat([data_df, data_df_inverse], axis=0, ignore_index=True)
                final_data = np.array(final_data).tolist()  # DataFrame->list
                train, test_and_dev = train_test_split(final_data,test_size=slicing_portion[0])  # 将数据集划分为train,test,dev
                test, dev = train_test_split(test_and_dev, test_size=slicing_portion[0])
    else:
        if dev:
            data_df = pd.DataFrame(data, columns=["q1", "q2"])
        else:
            data_df = pd.DataFrame(data, columns=["q1", "q2", "label"])
        # print("data_df.iloc[0:2]",data_df.iloc[0:2])
        cols = list(data_df)
        cols.insert(0, cols.pop(cols.index("q2")))
        data_df_inverse = data_df.loc[:, cols]
        if dev:
            data_df_inverse.columns = ["q1", "q2"]
        else:
            data_df_inverse.columns = ["q1", "q2", "label"]
        # print("data_df_inverse.iloc[0:2]", data_df_inverse.iloc[0:2])
        if dev:
            final_data = pd.concat([data_df, data_df_inverse], axis=0, ignore_index=True)
            final_data = np.array(final_data).tolist()  # DataFrame->list
            return final_data
        else:
            final_data = pd.concat([data_df,data_df_inverse], axis=0,ignore_index=True)
            final_data = np.array(final_data).tolist()  # DataFrame->list
            train, test_and_dev = train_test_split(final_data, test_size=slicing_portion[0])  # 将数据集划分为train,test,dev
            test, dev = train_test_split(test_and_dev, test_size=slicing_portion[0])
        #返回没有切分的数据集，和经过切分得到的训练集，验证集和测试集
    return final_data, train, dev, test

def stay_same_word(data,stopwords):
    """
    提取两个句子中相同的词汇，并进行拼接
    :param data: 待提取的数据集
    :param stopwords: 停用词表
    :return: 相同词汇拼成的句子的数据集
    """
    new_data = []
    for sample in data:
        question_1 = sample[0]
        question_2 = sample[1]
        question_1_seg = list(jieba.cut(question_1.strip()))
        question_2_seg = list(jieba.cut(question_2.strip()))
        same_word = []
        for word_1 in question_1_seg:
            if word_1 not in stopwords:
                if word_1 in question_2_seg:
                    same_word.append(word_1)
                    same_word.append("|")
                else:
                    continue
            else:
                continue
        # if len(same_word) == 0:
        #     same_word.append("")
        same_word_content = "".join(same_word)
        new_data.append(same_word_content)
    return new_data

def stay_different_word(data,stopwords):
    """
    提取两个句子中相同的词汇，并进行拼接
    :param data: 待提取的数据集
    :param stopwords: 停用词表
    :return: 相同词汇拼成的句子的数据集
    """
    new_data = []
    for sample in data:
        question_1 = sample[0].replace(" ","")
        question_2 = sample[1].replace(" ","")
        question_1_seg = list(jieba.cut(question_1.strip()))
        question_2_seg = list(jieba.cut(question_2.strip()))
        q1_different_word = []
        q2_different_word = []
        for word_1 in question_1_seg:
            if word_1 not in stopwords:
                if word_1 in question_2_seg:
                    pass
                else:
                    q1_different_word.append(word_1)
                    q1_different_word.append("|")
        # if len(q1_different_word) == 0:
        #     q1_different_word.append("")
        q1_different_word_content = "".join(q1_different_word)
        for word_2 in question_2_seg:
            # print(word_2)
             if word_2 not in stopwords:
                if word_2 not in question_1_seg:
                    q2_different_word.append(word_2)
                    q2_different_word.append("|")
                else:
                    pass
        # if len(q2_different_word) == 0:
        #     q2_different_word.append("")
        q2_different_word_content = "".join(q2_different_word)
        new_sample = [q1_different_word_content,q2_different_word_content]
        new_data.append(new_sample)
    return new_data


def get_data_pattern(data,dev=False,mode=1):
    """
    使用通配符替代每个样本[q1,q2,label]，中q1和q2的相同的名词
    :param data: 要进行处理的数据
    :param dev: 处理的数据是否是dev数据集
    :param mode: 1:表示采取使用通配符替换关键词的pattern;
                2:表示分别留下不相同的词和相同的词
    :return:
    """
    universal_character = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
    # 通配符，用来替换q1和q2中相同的名词
    pattern_data = [] # 没有使用同音字和近义词替换的原始数据
    noun_list = ["n", "nb", "nba", "nbc", "nbp", "nf", "ng", "nh", "nhd", "nhm", "ni", "nic", "nis", "nit", "nl", "nm",
                 "nmc", "nnd", "nnt","nr", "nr1", "nr2", "nrf", "nri", "ns", "nsf", "nt", "ntc", "ntcb", "ntcf", "ntch",
                 "nth", "nto","nts", "ntu", "nx", "nz", "rr","r","rz"]
    # 对样本进行patter的抽取
    if mode == 1:
        for sample in tqdm(data):
            tagging_q1 = HanLP.segment(sample[0])
            tagging_q2 = HanLP.segment(sample[1])
            word_tagging_q1 = ['%s/%s' %(term.word, term.nature) for term in tagging_q1]
            word_q1 = [term.word for term in tagging_q1]
            word_tagging_q2 = ['%s/%s' % (term.word, term.nature) for term in tagging_q2]
            word_q2 = [term.word for term in tagging_q2]
            # print(word_tagging_q1)
            # print(word_tagging_q2)
            index_equal = [[word_tagging_q1.index(x),word_tagging_q2.index(x)]
                           for x in word_tagging_q1 if x in word_tagging_q2 if x.split("/")[-1] in noun_list]
            # print("index",index_equal)
            for i in range(len(index_equal)):
                word_q1[index_equal[i][0]] = universal_character[i]
                word_q2[index_equal[i][1]] = universal_character[i]
            q1 = ''.join(word for word in word_q1)
            q2 = ''.join(word for word in word_q2)
            if dev:
                pattern_data.append([q1,q2])
            else:
                pattern_data.append([q1,q2,sample[2]])
        final_data = []
        for i in range(len(data)):
            # 如果测试集则没有标签
            if dev:
                final_data.append([data[i][0], data[i][1], pattern_data[i][0], pattern_data[i][1]])
            else:
                final_data.append([data[i][0],data[i][1],pattern_data[i][0],pattern_data[i][1],pattern_data[i][2]])
        # 数据的格式[q1,q2,q1',q2',label]或者[q1,q2,q1',q2']（dev的情况下）
        return final_data, pattern_data

    else:
        stopword_path = "./stopwords.txt"
        otherword_path = "./otherwords.txt"
        stopwords = [line.strip() for line in open(stopword_path, encoding='UTF-8').readlines()]
        otherword = [line.strip() for line in open(otherword_path, encoding='UTF-8').readlines()]
        stopwords += otherword
        # stopwords.append(" ")
        different_word_data = stay_different_word(data,stopwords)
        same_word_data = stay_same_word(data, stopwords)
        final_data = []
        pattern_data = []
        for i in range(len(data)):
            # 如果测试集则没有标签
            pattern_data.append([different_word_data[i][0], different_word_data[i][1],same_word_data[i]])
            if dev:
                final_data.append([data[i][0], data[i][1], different_word_data[i][0], different_word_data[i][1],
                                   same_word_data[i]])
            else:
                final_data.append([data[i][0], data[i][1], different_word_data[i][0], different_word_data[i][1],
                                   same_word_data[i],data[i][2]])
        # 数据的格式[q1,q2,q1',q2',label]或者[q1,q2,q1',q2']（dev的情况下）
        return final_data, pattern_data

def get_the_final_data():
    """
    获得最终版本的数据,数据格式为[q11,q21,q12,q22,label]
    :return:
    """
    save_data_dir = "./final_data/final_data_2/"
    if not os.path.exists(save_data_dir):
        os.mkdir(save_data_dir)
    cilinpath = "./cilin.txt"
    file_path_json = "./dataset/train_set.json"
    same_pinyin_file = "./same_pinyin.txt"
    chinese_word_freq_file = "./chinese-words.txt"
    save_data_synwords_and_samepinyin = save_data_dir + "data_replace_by_synwords_and_samepinyin.txt"
    data, true_data, false_data = read_data(file_path_json)
    data_eva = synword_and_samepinyin_data(true_data, save_data_synwords_and_samepinyin, cilinpath, same_pinyin_file,
                                         chinese_word_freq_file)   # 进行数据增强
    new_data = data_eva + data   # 包含增强后的数据集
    final_data,pattern_data = get_data_pattern(new_data)
    all_data, train, dev, test = data_inverse(final_data)

    all_data_path_txt = save_data_dir + "train_set.txt"
    train_path_txt = save_data_dir + "train.txt"
    test_path_txt = save_data_dir + "test.txt"
    dev_path_txt = save_data_dir + "dev.txt"
    save_data(all_data,all_data_path_txt)
    save_data(train,train_path_txt)
    save_data(test,test_path_txt)
    save_data(dev,dev_path_txt)

    # 生成测试集
    dev_csv_path = "./dataset/dev_set.csv"
    dev_txt_path = save_data_dir + "dev_set.txt"
    dev = read_data(dev_csv_path,dev=True)
    dev_data, pattern_dev = get_data_pattern(dev,dev=True)
    save_data(dev_data, dev_txt_path, columns_num=4)

def get_the_final_data_2(dev_samples=-5000):
    """
    获得最终版本的数据,数据格式为[q1,q2,label],并从原始训练集里切5000条数据作为测试集
    :return:
    """
    save_data_dir = "./final_data/final_data_7/"
    if not os.path.exists(save_data_dir):
        os.mkdir(save_data_dir)
    cilinpath = "./cilin.txt"
    file_path_json = "./dataset/train_set.json"
    same_pinyin_file = "./same_pinyin.txt"
    chinese_word_freq_file = "./chinese-words.txt"
    save_data_synwords_and_samepinyin = save_data_dir + "data_replace_by_synwords_and_samepinyin.txt"
    data, true_data, false_data = read_data(file_path_json)
    data_eva = synword_and_samepinyin_data(true_data, save_data_synwords_and_samepinyin, cilinpath, same_pinyin_file,
                                         chinese_word_freq_file)   # 进行数据增强
    new_data = data_eva + data   # 包含增强后的数据
    all_train_data = data_inverse(new_data,pattern=False)
    dev_data_from_train = new_data[dev_samples:]  # 从原始数据集里切500条数据出来作为验证集
    new_data = new_data[0:dev_samples]  # 剩下的数据作为训练集
    all_data, train, dev, test = data_inverse(new_data,pattern=False)
    # dev_data_from_train_1, dev_data_from_train_pattern = get_data_pattern(dev_data_from_train)

    all_train_data_path = save_data_dir + "all_train_data.txt"
    all_data_path_txt = save_data_dir + "train_set.txt"
    train_path_txt = save_data_dir + "train.txt"
    test_path_txt = save_data_dir + "test.txt"
    dev_path_txt = save_data_dir +"dev.txt"
    dev_from_train_path_txt = save_data_dir + "dev_split.txt"
    save_data(all_train_data,all_train_data_path,columns_num=3)
    save_data(all_data,all_data_path_txt,columns_num=3)
    save_data(train,train_path_txt,columns_num=3)
    save_data(test,test_path_txt,columns_num=3)
    save_data(dev,dev_path_txt,columns_num=3)
    save_data(dev_data_from_train, dev_from_train_path_txt,columns_num=3)

    # 生成测试集
    dev_csv_path = "./dataset/dev_set.csv"
    dev_txt_path = save_data_dir +"dev_set.txt"
    dev = read_data(dev_csv_path,dev=True)
    # dev_data, pattern_dev = get_data_pattern(dev,dev=True)
    save_data(dev, dev_txt_path, columns_num=2)

def get_the_final_data_3(dev_samples=-5000):
    """
    获得最终版本的数据,数据格式为[q11,q21,q12,q22,label],并从原始训练集里切5000条数据作为测试集,pattern为使用通配符替换相同词汇
    :return:
    """
    save_data_dir = "./final_data/final_data_10/"
    if not os.path.exists(save_data_dir):
        os.mkdir(save_data_dir)
    cilinpath = "./cilin.txt"
    file_path_json = "./dataset/train_set.json"
    same_pinyin_file = "./same_pinyin.txt"
    chinese_word_freq_file = "./chinese-words.txt"
    save_data_synwords_and_samepinyin = save_data_dir + "data_replace_by_synwords_and_samepinyin.txt"
    data, true_data, false_data = read_data(file_path_json)
    data_eva_true = synword_and_samepinyin_data(true_data, save_data_synwords_and_samepinyin, cilinpath, same_pinyin_file,
                                         chinese_word_freq_file,portition=0.2)   # 进行数据增强
    data_eva_false = synword_and_samepinyin_data(false_data, save_data_synwords_and_samepinyin, cilinpath,
                                                same_pinyin_file,
                                                chinese_word_freq_file, portition=0.3)  # 进行数据增强
    new_data = data_eva_true + data_eva_false + data   # 包含增强后的数据集
    all_train_data,all_train_data_pattern = get_data_pattern(new_data)
    all_train_data,all_train_train,all_train_dev,all_train_test = data_inverse(all_train_data)
    dev_data_from_train = new_data[dev_samples:]   # 从原始数据集里切500条数据出来作为验证集
    new_data = new_data[0:dev_samples]    # 剩下的数据作为训练集
    final_data,pattern_data = get_data_pattern(new_data)
    all_data, train, dev, test = data_inverse(final_data)
    dev_data_from_train_1,dev_data_from_train_pattern = get_data_pattern(dev_data_from_train)

    all_train_data_path = save_data_dir + "all_train_data.txt"
    all_data_path_txt = save_data_dir + "train_set.txt"
    train_path_txt = save_data_dir + "train.txt"
    test_path_txt = save_data_dir + "test.txt"
    dev_path_txt = save_data_dir + "dev.txt"
    dev_from_train_path_txt = save_data_dir + "dev_split.txt"
    save_data(all_train_data, all_train_data_path)
    save_data(all_data,all_data_path_txt)
    save_data(train,train_path_txt)
    save_data(test,test_path_txt)
    save_data(dev,dev_path_txt)
    save_data(dev_data_from_train_1,dev_from_train_path_txt)

    # 生成测试集
    dev_csv_path = "./dataset/test_set.csv"
    dev_txt_path = save_data_dir + "test_set.txt"
    dev = read_data(dev_csv_path,dev=True)
    dev_data, pattern_dev = get_data_pattern(dev,dev=True)
    save_data(dev_data, dev_txt_path, columns_num=4)

def get_the_final_data_4(dev_samples=-5000):
    """
    获得最终版本的数据,数据格式为[q11,q21,q12,q22,q31,label],并从原始训练集里切5000条数据作为测试集
    数据增强加到了10000条
    q12:q1中与q2不同的词汇
    q22:q2中与q1不同的词汇
    q31:q1与q2相同的词汇
    :return:
    """
    save_data_dir = "./final_data/final_data_6/"
    if not os.path.exists(save_data_dir):
        os.mkdir(save_data_dir)
    cilinpath = "./cilin.txt"
    file_path_json = "./dataset/train_set.json"
    same_pinyin_file = "./same_pinyin.txt"
    chinese_word_freq_file = "./chinese-words.txt"
    save_data_synwords_and_samepinyin = save_data_dir + "data_replace_by_synwords_and_samepinyin.txt"
    data, true_data, false_data = read_data(file_path_json)
    data_eva = synword_and_samepinyin_data(true_data, save_data_synwords_and_samepinyin, cilinpath, same_pinyin_file,
                                         chinese_word_freq_file,portition=0.1)   # 进行数据增强
    new_data = data_eva + data   # 包含增强后的数据集
    print(len(new_data))
    all_train_data,all_train_data_pattern = get_data_pattern(new_data,mode=2)
    all_train_data,all_train_train,all_train_dev,all_train_test = data_inverse(all_train_data,mode=2)
    print(len(all_train_data))
    dev_data_from_train = new_data[dev_samples:]   # 从原始数据集里切500条数据出来作为验证集
    new_data = new_data[0:dev_samples]    # 剩下的数据作为训练集
    final_data,pattern_data = get_data_pattern(new_data,mode=2)
    all_data, train, dev, test = data_inverse(final_data,mode=2)
    dev_data_from_train_1,dev_data_from_train_pattern = get_data_pattern(dev_data_from_train,mode=2)

    all_train_data_path = save_data_dir + "all_train_data.txt"
    all_data_path_txt = save_data_dir + "train_set.txt"
    train_path_txt = save_data_dir + "train.txt"
    test_path_txt = save_data_dir + "test.txt"
    dev_path_txt = save_data_dir + "dev.txt"
    dev_from_train_path_txt = save_data_dir + "dev_split.txt"
    save_data(all_train_data,all_train_data_path,columns_num=6)
    save_data(all_data,all_data_path_txt,columns_num=6)
    save_data(train,train_path_txt,columns_num=6)
    save_data(test,test_path_txt,columns_num=6)
    save_data(dev,dev_path_txt,columns_num=6)
    save_data(dev_data_from_train_1,dev_from_train_path_txt,columns_num=6)

    # 生成测试集
    dev_csv_path = "./dataset/dev_set.csv"
    dev_txt_path = save_data_dir + "dev_set.txt"
    dev = read_data(dev_csv_path,dev=True)
    dev_data, pattern_dev = get_data_pattern(dev,dev=True,mode=2)
    save_data(dev_data, dev_txt_path, columns_num=5)

def get_the_final_data_5(dev_samples=-5000):
    """
    获得最终版本的数据,数据格式为[q1,q2,label],并从原始训练集里切5000条数据作为测试集
    注意：只包含增强后的数据，不包含原始数据
    :return:
    """
    save_data_dir = "./final_data/final_data_9/"
    if not os.path.exists(save_data_dir):
        os.mkdir(save_data_dir)
    cilinpath = "./cilin.txt"
    file_path_json = "./dataset/train_set.json"
    same_pinyin_file = "./same_pinyin.txt"
    chinese_word_freq_file = "./chinese-words.txt"
    save_data_synwords_and_samepinyin = save_data_dir + "data_replace_by_synwords_and_samepinyin.txt"
    data, true_data, false_data = read_data(file_path_json)
    data_eva = synword_and_samepinyin_data(data, save_data_synwords_and_samepinyin, cilinpath, same_pinyin_file,
                                         chinese_word_freq_file,portition=1)   # 进行数据增强
    # new_data = data_eva + data   # 包含增强后的数据
    new_data = data_eva
    all_train_data,all_train_data_1,all_train_data_2,all_train_dat_3 = data_inverse(new_data,pattern=False)
    dev_data_from_train = new_data[dev_samples:]  # 从原始数据集里切500条数据出来作为验证集
    new_data = new_data[0:dev_samples]  # 剩下的数据作为训练集
    all_data, train, dev, test = data_inverse(new_data,pattern=False)
    # dev_data_from_train_1, dev_data_from_train_pattern = get_data_pattern(dev_data_from_train)

    all_train_data_path = save_data_dir + "all_train_data.txt"
    all_data_path_txt = save_data_dir + "train_set.txt"
    train_path_txt = save_data_dir + "train.txt"
    test_path_txt = save_data_dir + "test.txt"
    dev_path_txt = save_data_dir +"dev.txt"
    dev_from_train_path_txt = save_data_dir + "dev_split.txt"
    save_data(all_train_data,all_train_data_path,columns_num=3)
    save_data(all_data,all_data_path_txt,columns_num=3)
    save_data(train,train_path_txt,columns_num=3)
    save_data(test,test_path_txt,columns_num=3)
    save_data(dev,dev_path_txt,columns_num=3)
    save_data(dev_data_from_train, dev_from_train_path_txt,columns_num=3)

    # 生成测试集
    dev_csv_path = "./dataset/test_set.csv"
    dev_txt_path = save_data_dir +"test_set.txt"
    dev = read_data(dev_csv_path,dev=True)
    save_data_synwords_and_samepinyin_for_dev = save_data_dir + "data_replace_by_synwords_and_samepinyin_for_dev.txt"
    data_eva = synword_and_samepinyin_data(dev, save_data_synwords_and_samepinyin_for_dev, cilinpath, same_pinyin_file,
                                           chinese_word_freq_file, columns_num=2,portition=1)  # 进行数据增强
    # dev_data, pattern_dev = get_data_pattern(dev,dev=True)
    save_data(dev, dev_txt_path, columns_num=2)


if __name__ == "__main__":
    """调用以下不同函数，可以获得不同格式的数据集，具体内容可查看每个函数的注释"""
    get_the_final_data_3()
    # get_the_final_data_2()
    # get_the_final_data_4()
    # get_the_final_data_5()



