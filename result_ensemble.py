import pandas as pd
import numpy as np
from tqdm import tqdm
def read_data(path,qid=False):
    """
    读取csv文件
    :param path:
    :param qid: True or False，为False时不保留qid这一行
    :return:
    """
    data_df = pd.read_csv(path, sep="\t")
    # data_df.drop(["qid"], axis=1)
    data = np.array(data_df).tolist()
    if not qid:
        data = [i[1:] for i in data]
    return data

def ensemble():
    data_path = ["./sample_submission.csv","./wang.csv"]
    ensemble_data_path = "./wang_1.csv"
    ensemble_data = []
    all_data = []
    for path in data_path:
        data = read_data(path)
        all_data.append(data)
    for i in range(all_data[0]):
        count_1 = 0
        count_0 = 0
        for j in len(all_data):
            if all_data[j][i] == 1:
                count_1 += 1
            else:
                count_0 += 0
        if count_1 > count_0:
            ensemble_data.append(1)
        elif count_1 < count_0:
            ensemble_data.append(0)
        else:   # 个数相同怎么处理===============------------------------------------
            ensemble_data.append(1)
    id = [i for i in range(1,len(ensemble_data)+1)]
    submition_data = list(zip(id,ensemble_data))
    submition_data_df = pd.DataFrame(submition_data,columns=["qid","label"])
    submition_data_df.to_csv(ensemble_data_path,index=0)

def ensembale_accord_rule():
    model_data_path = "./result_submit/B0265stage2(7638).csv"
    rule_recorrect_path = {"A_B_and_B_C_sample":"./tongji/A_B_and_B_C_sample.csv",
                           "same_sample_in_train_and_test":"./tongji/same_sample_in_train_and_test.csv"}
    recorrect_sample_path = "./tongji/recorrect_sample_accord_rule_"+model_data_path.split("/")[-1]
    all_model_data = read_data(model_data_path,qid=True)
    same_sample_in_train_and_test = read_data(rule_recorrect_path["same_sample_in_train_and_test"],qid=True)
    same_sample_in_train_and_test_qid = [item[0] for item in same_sample_in_train_and_test]
    A_B_and_B_C_sample = read_data(rule_recorrect_path["A_B_and_B_C_sample"],qid=True)
    A_B_and_B_C_sample_qid = [item[0] for item in A_B_and_B_C_sample]
    qid_diff = list(set(A_B_and_B_C_sample_qid).difference(set(same_sample_in_train_and_test_qid)))
    qid_diff_label = [item[3] for item in A_B_and_B_C_sample if item[0] in qid_diff]
    count = 0
    recorrect_sample = []
    for sample in tqdm(same_sample_in_train_and_test):
        if sample[5] == all_model_data[sample[0]][1]:
                continue
        else:
            recorrect_sample.append([sample[0],all_model_data[sample[0]][1],sample[5]])
            all_model_data[sample[0]][1] = sample[5]
            count += 1
    all_model_data = pd.DataFrame(all_model_data)
    recorrect_all_model_data_path =  "./tongji/recorrect_all_model_data_"+model_data_path.split("/")[-1]
    all_model_data.to_csv(recorrect_all_model_data_path, sep="\t", index=None)

    # for i in tqdm(range(len(qid_diff))):
    #     if qid_diff_label[i] == 0:
    #         continue
    #     else:
    #         if qid_diff_label[i] == all_model_data[qid_diff[i]][1]:
    #             continue
    #         else:
    #             recorrect_sample.append([qid_diff[i], all_model_data[qid_diff[i]][1], qid_diff_label[i]])
    #             all_model_data[qid_diff[i]][1] = qid_diff_label[i]
    #             count += 1

    recorrect_sample_df = pd.DataFrame(recorrect_sample,columns=["qid","model_lable","rule_label"])
    recorrect_sample_df.to_csv(recorrect_sample_path, sep="\t", index=None)
    print("count:{}".format(count))




if __name__ == "__main__":
    # ensemble()
    ensembale_accord_rule()