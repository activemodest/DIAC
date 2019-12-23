from feature_a import get_feature_a
from feature_b import get_feature_b
from feature_c import get_feature_c
from feature_d import get_feature_d
import os
from tqdm import tqdm

def get_statistic_feature():
    save_dir = "./statistic_feature_for_final_data_4/"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    get_file_dir = "./final_data_4/"
    data_set = [get_file_dir+"train_set.txt",get_file_dir+"dev_set.txt",get_file_dir+"dev_split.txt",
                get_file_dir+"dev.txt",get_file_dir+"train.txt",get_file_dir+"dev.txt"]
    for data_path in tqdm(data_set):
        get_feature_a(data_path, save_dir)
        get_feature_b(data_path, save_dir)
        get_feature_c(data_path, save_dir)
        get_feature_d(data_path, save_dir)

if __name__ == "__main__":
    get_statistic_feature()