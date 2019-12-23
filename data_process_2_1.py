import pycorrector
import pandas as pd
import numpy as np
import json
import sys
from pyhanlp import HanLP   # 调入自然语言处理工具包
import random
from tqdm import tqdm
random.seed(1)   # 设置随机种子

def read_data(data_path,dev=False):
    """
    从文件中读取数据，并转化格式
    :param data_json_path:  待读取的数据的格式
    :dev: 判断读取的数据文件是否为测试集，测试集的数据集没有标签，因此要单独进行处理
    :return: new_data, true_data, false_data，数据格式为[[q1,q2,label]...]或[[q1,q2,label]...]
    """
    #读取json格式的数据
    if data_path.split(".")[-1] == "json":
        with open(data_path,"r",encoding="UTF-8") as f:   #读取json格式的文件
            all_data = json.load(f)
        new_data = []
        for sample in all_data:
            new_data.append([sample["q1"],sample["q2"],sample["label"]])
        true_data = [sample for sample in new_data if sample[2]==1]    # 获取正样本
        false_data = [sample for sample in new_data if sample[2]==0]    # 获取负样本
    #读取csv格式的数据
    elif data_path.split(".")[-1] == "csv":
        data_df = pd.read_csv(data_path, sep="\t")
        new_data = np.array(data_df).tolist()   # 将DataFrame格式转化为list格式
        true_data = [sample for sample in new_data if sample[2] == 1]  # 获取正样本
        false_data = [sample for sample in new_data if sample[2] == 0]  # 获取负样本
    # 其他格式的数据异常退出
    else:
        print("读取数据出错")
        sys.exit(0)   # 程序报错退出

    if dev:
        data_df = pd.read_csv(data_path, sep="\t")
        data_df.drop(["qid"], axis=1)
        data = np.array(data_df).tolist()
        dev = [[i[1], i[2]] for i in data]
        # dev = np.array(data_df).tolist()[:][1:]  # DataFrame->list
        return dev
    else:
        return new_data, true_data, false_data

def save_data(data,file_path,columns_num=5):
    """
    将数据写到文件中，并按指定格式保存
    :param data: 要存储的数据集，list格式
    :param file_path:
    :param columns_num: 要存储的数据集中样本的列数
    :return:
    """
    with open(file_path, "w", encoding="UTF-8") as f:
        for sample in data:
            if columns_num == 2:
                rowtxt = "{}\t{}\t".format(sample[0], sample[1])
            elif columns_num == 3:
                rowtxt = "{}\t{}\t{}\t".format(sample[0], sample[1], sample[2])
            elif columns_num == 4:
                rowtxt = "{}\t{}\t{}\t{}\t".format(sample[0], sample[1], sample[2], sample[3])
            elif columns_num == 5:
                rowtxt = "{}\t{}\t{}\t{}\t{}\t".format(sample[0], sample[1], sample[2],sample[3], sample[4])
            elif columns_num == 6:
                rowtxt = "{}\t{}\t{}\t{}\t{}\t{}\t".format(sample[0], sample[1], sample[2], sample[3], sample[4],sample[5])
            else:
                print("write data to file error")
                sys.exit(0)  # 程序报错退出
            f.write(rowtxt)
            f.write("\n")

def get_keyword(content,keynum=2):
    """
    获取每个问题中的关键字,关键词的数目由keynum控制
    :param content: 一个句子
    :return:
    """
    keywordList = HanLP.extractKeyword(content,keynum)
    return keywordList

def construct_synwords(cilinpath):
    """
    根据哈工大的同义词词林（cilin.txt）构建同义词表，文件来自https://github.com/TernenceWind/replaceSynbycilin/blob/master/cilin.txt
    :param cilinpath:  同义词词林的路径
    :return:
    """
    synwords = []
    with open(cilinpath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            temp = line.strip()
            split = temp.split(' ')
            bianhao = split[0]
            templist = split[1:]
            if bianhao[-1] == '=':
                synwords.append(templist)
    return synwords

def replace_synwords(content,synwords):
    """
    使用同义词替换content中的关键词
    :param content:  需要进行同义词替换的句子，不是整个样本或者数据集
    :param synwords: 同义词词典
    :return:
    """
    segmentationList = HanLP.segment(content)
    # print(len(segmentationList))
    if len(set(segmentationList)) <= 2:
        keynum = 1
    elif len(segmentationList) > 2 and len(set(segmentationList)) <= 6:
        keynum = 2
    else:
        # keynum = int(len(set(segmentationList))/3)
        keynum = 4
    keywordList = get_keyword(content,keynum)   # 获取关键词
    # print(keywordList)

    segmentationList = [term.word for term in segmentationList]
    replace_word = {}
    #查询content中的关键词在同义词表中的近义词
    for word in keywordList:
        if word in segmentationList:
            for syn in synwords:
                # if word in syn:   # 设计替换规则
                if word == syn[0]:
                    if len(syn) == 1:
                        continue
                    else:
                        # 以最靠近word的词替换word
                        if syn.index(word) == 0:
                            replace_word[word] = (syn[1])
                        else:
                            replace_word[word] = (syn[syn.index(word)-1])
                else:
                    continue
        else:
            continue

    # 替换content中的同义词
    for i in range(len(segmentationList)):
        if segmentationList[i] in replace_word:
            segmentationList[i] = replace_word[segmentationList[i]]
        else:
            continue
    # 将content重新组合成句子
    content_new = "".join(segmentationList)
    # 返回经过替换后的content,即new_content
    return content_new

def get_same_pinyin_vocabulary(same_pinyin_file):
    """
    获得具有相同拼音的词表，文件来自https://github.com/shibing624/pycorrector/blob/master/pycorrector/data/same_pinyin.txt
    :param same_pinyin_file:
    :return: {"word1":samepinyin,"word2":samepinyin}
    """
    same_pinyin = {}
    with open(same_pinyin_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines[1:]:
            temp = line.strip("\n")
            split_1 = temp.split('\t')
            word_index = split_1[0]  # 词根
            # word_same_1 = split_1[1]   #同音同调
            # word_same_2 = split_1[2]   #同音异调
            # word_samePinyin = split_1[1]+split_1[2]
            sameWords = ""
            for i in split_1[1:]:  #拼接同音同调和同音异调词表
                sameWords += i
            same_pinyin[word_index] = list(sameWords)   # 将同音同调和同音异调放在同一个list中
            # same_pinyin[word_index] = [list(word_same_1),list(word_same_2)]   # 将同音同调和同音异调放在不同list中
    # 格式[word,freq]
    return same_pinyin

def get_word_freq(chinese_word_freq_file_path):
    '''
    读取word,frequency ,构建词典
    :param chinese_word_freq_file_path:   中文词频文件
    :return: {"word1":freq1,"word2":freq2}
    '''
    word_freq_vocab = {} # 词频字典,格式为[“word”:freq]
    with open(chinese_word_freq_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines[1:]:
            line = line.strip()
            word_freq = line.split(" ")
            if word_freq[0] not in word_freq_vocab:
                word_freq_vocab[word_freq[0]] = int(word_freq[1])  # 添加"word"：freq到词典中
            else:
                pass
    # print("word_freq_vocab", word_freq_vocab["火"])
    return word_freq_vocab

def replace_samePinyin(content,same_pinyin,word_freq_vocab,replace_num=1):
    """
    使用同音字替换content中关键词中，（替换规则为替换掉所有同音字出现频率最高的那个字）
    :param content:  要替换的文本
    :param same_pinyin: 相同拼音词汇表
    :param word_freq_vocab: 汉语字频率表
    :param replace_num: 要替换的数量，这个版本目前只考虑一个content中只替换一个字
    :return: 经过相同拼音替换掉的文本
    """
    segmentationList = HanLP.segment(content)
    word_list_of_content = list(content)
    # print(len(segmentationList))
    if len(set(segmentationList)) <= 2:
        keynum = 1
    elif len(segmentationList) > 2 and len(set(segmentationList)) <= 6:
        keynum = 2
    else:
        # keynum = int(len(set(segmentationList))/3)
        keynum = 4
    keywordList = get_keyword(content,keynum)   # 获取关键词
    key_character = []
    for word in keywordList:   # 提取关键词里的关键字
        key_character += list(word)
    key_character = list(set(key_character))   # 去掉重复的关键字
    key_character = [word for word in key_character if word in same_pinyin]# 先检查关键词中的所有字是否都出现在same_pinyin词汇表中
    word_freq = []
    for i in key_character:   # 统计关键字的频率
        samePinyin_list = same_pinyin[i]   # 获取相同拼音的所有字
        samePinyin_freq = []
        for j in samePinyin_list:
            if j in word_freq_vocab:
                samePinyin_freq.append(word_freq_vocab[j])
            else:
                samePinyin_freq.append(1)
        word_freq.append(samePinyin_list[samePinyin_freq.index(max(samePinyin_freq))])
    freq =[]
    if len(word_freq) != 0:
        for i in word_freq:
            if i in word_freq_vocab:
                freq.append(word_freq_vocab[i])
            else:
                freq.append(1)
        same_pinyin_HighFreq_word = word_freq[freq.index(max(freq))]
        replace_word = key_character[freq.index(max(freq))]
        replace_index = word_list_of_content.index(replace_word)
        word_list_of_content[replace_index] = same_pinyin_HighFreq_word
        new_content =  "".join(word_list_of_content)
        # print("smae_pinyin",same_pinyin["火"])
        return new_content
    else:
        return content


def synword_and_samepinyin_data(data,save_data_file,cilinpath,same_pinyin_file,chinese_word_freq_file,repalce_rule=True,columns_num=3,portition=0.05):
    """
    从data中取一定比例的样本，进行数据增强
    :param data: 数据集
    :param save_data_file:  保存增强后的数据集的文件
    :param cilinpath: 近义词的词表文件
    :param same_pinyin_file:  同音字的词表文件
    :param chinese_word_freq_file:  中文字频文件
    :param repalce_rule: True为从正样本中的q1和q2中随机选择一个作为待替换的文本
                        false则将q1和q2都进行同义词替换
    :param portition: 要替换的样本的比例
    :return: 经过数据增强后的数据集
    """
    synwords = construct_synwords(cilinpath) # 获取近义词词表
    same_pinyin_vocab = get_same_pinyin_vocabulary(same_pinyin_file)   # 获取相同拼音的词表
    word_freq_vocab = get_word_freq(chinese_word_freq_file)    # 获取汉字词频表
    data_df = pd.DataFrame(data)
    if portition != 1:
        samples = data_df.sample(frac=portition, replace=False,random_state=1)  # 随机选着一定比例的样本进行数据增强
    else:
        samples = data_df
    new_samples = np.array(samples).tolist()   # 将DataFrame转化为list
    # print(save_data_file)
    save_original_data_file = "." + "".join(save_data_file.split(".")[:-1]) + "_originalData." + save_data_file.split(".")[-1]

    save_data(new_samples, save_original_data_file, columns_num=columns_num)  # 将经过同义词替换的数据保存到文件
    random_synword_or_samepinyin = [random.randint(0, 1) for i in range(len(new_samples))]  # 用于随机选择使用相同拼音替换策略或者同义词替换策略
    new_data = []  # 保存经过同义词替换后得到的样本
    if repalce_rule:
        random_q1_or_q2 = [random.randint(0,1) for i in range(len(new_samples))]   # 用于随机选取一个句子做增强（替换成相同的拼音或者同义词）
        for i in tqdm(range(len(new_samples))):
            content = new_samples[i][random_q1_or_q2[i]]
            if random_synword_or_samepinyin[i] == 0: # 使用同义词替换策略
                new_content = replace_synwords(content, synwords)
            else:   # 使用相同拼音的词替代
                new_content = replace_samePinyin(content, same_pinyin_vocab, word_freq_vocab)
            backup_sample = new_samples[i]
            if new_content == content:
                continue
            else:   # 如果经过同义词替换后的句子与原句子不同，则得到新样本
                new_samples[i][random_q1_or_q2[i]] = new_content
                new_data.append(new_samples[i])

    else:
        for i in tqdm(range(len(new_samples))):
            content_1 = new_samples[0]
            content_2 = new_samples[1]
            if random_synword_or_samepinyin[i] == 0:  # 使用同义词替换策略
                new_content_1 = replace_synwords(content_1,synwords)
                new_content_2 = replace_synwords(content_2,synwords)
            else:
                new_content_1 = replace_samePinyin(content_1, same_pinyin_vocab, word_freq_vocab)
                new_content_2 = replace_samePinyin(content_2, same_pinyin_vocab, word_freq_vocab)
            backup_sample = new_samples[i]
            if new_content_1 == content_1:
                continue
            else:
                new_data.append([new_content_1,new_samples[i][1],new_samples[i][2]])
            if new_content_2 == content_2:
                continue
            else:
                new_data.append([new_samples[i][0],new_content_1, new_samples[i][2]])
    save_data(new_data, save_data_file, columns_num=columns_num)   # 将经过同义词替换的数据保存到文件
    # 返回经过同义词替换的数据集
    return new_data

def wrong_written_character():
    """错别字纠正，最终并没有用到这次数据增强中"""
    correct_sent, detail = pycorrector.correct("吸度后开车撞死人会判刑吗？怎么判刑")
    print("correct_sent:{},detail:{}".format(correct_sent,detail))

if __name__ == "__main__":
    cilinpath = "./cilin.txt"
    file_path_json = "./dataset/train_set.json"
    same_pinyin_file = "./same_pinyin.txt"
    chinese_word_freq_file = "./chinese-words.txt"
    save_data_synwords_and_samepinyin = "./data_replace_by_synwords_and_samepinyin/data_replace_by_synwords_and_samepinyin.txt"
    new_data, true_data, false_data = read_data(file_path_json)
    data_1 = synword_and_samepinyin_data(true_data, save_data_synwords_and_samepinyin, cilinpath, same_pinyin_file, chinese_word_freq_file)

